#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <iomanip>

// NOTE: This is a standalone, hand-written COFF (.obj) parser for Windows/MSVC-style object files.
// It prints headers, sections, relocations, symbols, and resolves long names via the string table.
// Comments are in English per user request.
// Compile (MSVC):  cl /std:c++17 /EHsc COFFObjParser.cpp
// Compile (Clang on Windows): clang++ -std=c++17 -o coffparse.exe COFFObjParser.cpp
// Usage: coffparse.exe path\to\file.obj

#pragma pack(push, 1)
struct IMAGE_FILE_HEADER {
	uint16_t Machine;                 // Target machine
	uint16_t NumberOfSections;        // Section count
	uint32_t TimeDateStamp;           // Timestamp
	uint32_t PointerToSymbolTable;    // File offset to COFF symbol table
	uint32_t NumberOfSymbols;         // Number of symbols
	uint16_t SizeOfOptionalHeader;    // 0 for .obj
	uint16_t Characteristics;         // Flags
};

struct IMAGE_SECTION_HEADER {
	char     Name[8];                 // Section name or slash+offset
	uint32_t VirtualSize;             // Not meaningful for .obj, but present in MS format
	uint32_t VirtualAddress;          // Not meaningful for .obj
	uint32_t SizeOfRawData;           // Raw data size
	uint32_t PointerToRawData;        // File offset to raw data
	uint32_t PointerToRelocations;    // File offset to relocation entries
	uint32_t PointerToLinenumbers;    // Deprecated
	uint16_t NumberOfRelocations;     // Relocation entry count
	uint16_t NumberOfLinenumbers;     // Deprecated
	uint32_t Characteristics;         // Section flags
};

// COFF relocation entry (MSVC):
struct IMAGE_RELOCATION {
	uint32_t VirtualAddress;          // Offset within the section to fix up
	uint32_t SymbolTableIndex;        // Index into the COFF symbol table
	uint16_t Type;                    // Relocation type (machine-specific)
};

// COFF symbol table entry: 18 bytes per entry
struct IMAGE_SYMBOL {
	union {
		char ShortName[8];            // Either 8-char name or zero + offset into string table
		struct {
			uint32_t Zeroes;
			uint32_t Offset;         // Offset into string table when Zeroes == 0
		} LongName;
	} N;
	uint32_t Value;                   // Value/offset within section
	int16_t  SectionNumber;           // Section index or special (0=UNDEF, -1=ABS, -2=DEBUG)
	uint16_t Type;                    // Symbol type (least used today)
	uint8_t  StorageClass;            // External, static, function, etc.
	uint8_t  NumberOfAuxSymbols;      // Count of following aux records
};
#pragma pack(pop)

// Utility: read entire file into memory
static bool ReadFileBytes(const std::string& path, std::vector<uint8_t>& out) {
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs) return false;
	ifs.seekg(0, std::ios::end);
	std::streamsize size = ifs.tellg();
	if (size < 0) return false;
	out.resize(static_cast<size_t>(size));
	ifs.seekg(0, std::ios::beg);
	if (!ifs.read(reinterpret_cast<char*>(out.data()), size)) return false;
	return true;
}

// Utility: bounds check when viewing a structure inside a byte vector
template <typename T>
const T* View(const std::vector<uint8_t>& buf, size_t offset) {
	if (offset + sizeof(T) > buf.size()) return nullptr;
	return reinterpret_cast<const T*>(buf.data() + offset);
}

// Utility: raw pointer view (length-checked)
static const uint8_t* ViewSpan(const std::vector<uint8_t>& buf, size_t offset, size_t length) {
	if (offset + length > buf.size()) return nullptr;
	return buf.data() + offset;
}

// Resolve section name: either inline or slash-number (offset into string table)
static std::string ResolveSectionName(const IMAGE_SECTION_HEADER& sh,
	const uint8_t* strtab,
	uint32_t strtab_size) {
	// If name begins with '/', it may be "/<decimal_offset>" into string table
	if (sh.Name[0] == '/') {
		char* endptr = nullptr;
		long ofs = std::strtol(sh.Name + 1, &endptr, 10);
		if (endptr && *endptr == '\0' && ofs > 0 && static_cast<uint32_t>(ofs) < strtab_size) {
			const char* s = reinterpret_cast<const char*>(strtab + ofs);
			return std::string(s);
		}
	}
	// Otherwise treat as 8-char possibly non-null-terminated
	char name[9] = { 0 };
	std::memcpy(name, sh.Name, 8);
	return std::string(name);
}

// Resolve symbol name from IMAGE_SYMBOL and string table
static std::string ResolveSymbolName(const IMAGE_SYMBOL& sym,
	const uint8_t* strtab,
	uint32_t strtab_size) {
	if (sym.N.LongName.Zeroes == 0 && sym.N.LongName.Offset != 0) {
		uint32_t ofs = sym.N.LongName.Offset;
		if (ofs < strtab_size) {
			const char* s = reinterpret_cast<const char*>(strtab + ofs);
			return std::string(s);
		}
		else {
			return std::string("<bad_strtab_ofs>");
		}
	}
	else {
		// 8-byte short name, may be not zero-terminated
		char name[9] = { 0 };
		std::memcpy(name, sym.N.ShortName, 8);
		return std::string(name);
	}
}

// Decode some common Machine values
static const char* MachineToString(uint16_t m) {
	switch (m) {
	case 0x014c: return "I386 (x86)";
	case 0x8664: return "AMD64 (x64)";
	case 0x01c0: return "ARM";
	case 0x01c4: return "ARMv7";
	case 0xAA64: return "ARM64";
	default: return "Unknown";
	}
}

// Decode some special SectionNumber values
static const char* SectionNumberToString(int16_t sn) {
	switch (sn) {
	case 0: return "UNDEF";
	case -1: return "ABS";
	case -2: return "DEBUG";
	default: return nullptr; // real section index
	}
}

// Decode some StorageClass values (partial)
static const char* StorageClassToString(uint8_t sc) {
	switch (sc) {
	case 2: return "EXTERNAL";
	case 3: return "STATIC";
	case 101: return "FUNCTION";
	case 105: return "FILE";
	default: return "";
	}
}

// Pretty hex dump helper (limited)
static void HexDump(const uint8_t* data, size_t len, size_t max_bytes = 64) {
	size_t n = std::min(len, max_bytes);
	for (size_t i = 0; i < n; ++i) {
		if (i % 16 == 0) std::cout << "\n    ";
		std::cout << std::hex << std::setw(2) << std::setfill('0')
			<< static_cast<int>(data[i]) << ' ';
	}
	std::cout << std::dec << (len > max_bytes ? "\n    ..." : "") << "\n";
}

int main(int argc, char** argv) {
	std::string obj_file = R"(D:\code\mycode\Test\win32_test\x64\Debug\win32_test.obj)";
	//if (argc < 2) {
	//	std::cerr << "Usage: coffparse <path-to-obj>\n";
	//	return 1;
	//}

	std::vector<uint8_t> buf;
	if (!ReadFileBytes(obj_file, buf)) {
		std::cerr << "Failed to read file: " << argv[1] << "\n";
		return 1;
	}

	if (buf.size() < sizeof(IMAGE_FILE_HEADER)) {
		std::cerr << "File too small to be a COFF object.\n";
		return 1;
	}

	// COFF header starts at file beginning in .obj
	const IMAGE_FILE_HEADER* fh = View<IMAGE_FILE_HEADER>(buf, 0);
	if (!fh) { std::cerr << "Failed to read COFF header.\n"; return 1; }

	std::cout << "=== COFF FILE HEADER ===\n";
	std::cout << "Machine             : 0x" << std::hex << fh->Machine << std::dec
		<< " (" << MachineToString(fh->Machine) << ")\n";
	std::cout << "NumberOfSections    : " << fh->NumberOfSections << "\n";
	std::cout << "TimeDateStamp       : 0x" << std::hex << fh->TimeDateStamp << std::dec << "\n";
	std::cout << "PtrToSymbolTable    : 0x" << std::hex << fh->PointerToSymbolTable << std::dec << "\n";
	std::cout << "NumberOfSymbols     : " << fh->NumberOfSymbols << "\n";
	std::cout << "SizeOfOptionalHeader: " << fh->SizeOfOptionalHeader << "\n";
	std::cout << "Characteristics     : 0x" << std::hex << fh->Characteristics << std::dec << "\n\n";

	// Section headers follow immediately after the file header + optional header (usually 0 in .obj)
	size_t sec_hdr_off = sizeof(IMAGE_FILE_HEADER) + fh->SizeOfOptionalHeader;
	std::vector<IMAGE_SECTION_HEADER> sections;
	sections.reserve(fh->NumberOfSections);

	for (uint16_t i = 0; i < fh->NumberOfSections; ++i) {
		const IMAGE_SECTION_HEADER* sh = View<IMAGE_SECTION_HEADER>(buf, sec_hdr_off + i * sizeof(IMAGE_SECTION_HEADER));
		if (!sh) { std::cerr << "Failed to read section header #" << (i + 1) << "\n"; return 1; }
		sections.push_back(*sh);
	}

	// Symbol table and string table
	const uint32_t symtab_ofs = fh->PointerToSymbolTable;
	const uint32_t nsyms = fh->NumberOfSymbols;

	const uint8_t* symtab_ptr = nullptr;
	const IMAGE_SYMBOL* sym_first = nullptr;

	uint32_t strtab_size = 0;
	const uint8_t* strtab_ptr = nullptr;

	if (symtab_ofs != 0 && nsyms > 0) {
		size_t symtab_bytes = static_cast<size_t>(nsyms) * sizeof(IMAGE_SYMBOL);
		symtab_ptr = ViewSpan(buf, symtab_ofs, symtab_bytes);
		if (!symtab_ptr) { std::cerr << "Symbol table out of range.\n"; return 1; }
		sym_first = reinterpret_cast<const IMAGE_SYMBOL*>(symtab_ptr);

		// String table immediately follows the symbol table. First 4 bytes = total size of string table.
		size_t strtab_ofs = symtab_ofs + symtab_bytes;
		if (auto psize = View<uint32_t>(buf, strtab_ofs)) {
			strtab_size = *psize;
			if (strtab_size >= 4) {
				strtab_ptr = ViewSpan(buf, strtab_ofs + 4, strtab_size - 4);
				if (!strtab_ptr) { std::cerr << "String table out of range.\n"; return 1; }
			}
		}
	}

	std::cout << "=== SECTIONS (" << sections.size() << ") ===\n";
	for (size_t i = 0; i < sections.size(); ++i) {
		const auto& sh = sections[i];
		std::string sname = ResolveSectionName(sh, strtab_ptr, strtab_size);
		std::cout << "[#" << (i + 1) << "] Name='" << sname << "'\n";
		std::cout << "     SizeOfRawData      : 0x" << std::hex << sh.SizeOfRawData << std::dec << " (" << sh.SizeOfRawData << ")\n";
		std::cout << "     PtrToRawData       : 0x" << std::hex << sh.PointerToRawData << std::dec << "\n";
		std::cout << "     PtrToRelocations   : 0x" << std::hex << sh.PointerToRelocations << std::dec << "\n";
		std::cout << "     NumberOfRelocations: " << sh.NumberOfRelocations << "\n";
		std::cout << "     Characteristics    : 0x" << std::hex << sh.Characteristics << std::dec << "\n";

		// Show a small hexdump of section raw data if any
		if (sh.PointerToRawData && sh.SizeOfRawData) {
			const uint8_t* data = ViewSpan(buf, sh.PointerToRawData, sh.SizeOfRawData);
			if (data) {
				std::cout << "     RawData (first bytes): ";
				HexDump(data, sh.SizeOfRawData, 48);
			}
		}
		std::cout << "\n";
	}

	// Relocations per section
	std::cout << "=== RELOCATIONS ===\n";
	for (size_t i = 0; i < sections.size(); ++i) {
		const auto& sh = sections[i];
		if (sh.NumberOfRelocations == 0) continue;
		std::cout << "Section #" << (i + 1) << " relocations (count=" << sh.NumberOfRelocations << "):\n";
		size_t relo_bytes = static_cast<size_t>(sh.NumberOfRelocations) * sizeof(IMAGE_RELOCATION);
		const IMAGE_RELOCATION* relos = reinterpret_cast<const IMAGE_RELOCATION*>(ViewSpan(buf, sh.PointerToRelocations, relo_bytes));
		if (!relos) { std::cout << "  <relocation table out of range>\n"; continue; }

		for (uint16_t r = 0; r < sh.NumberOfRelocations; ++r) {
			const auto& re = relos[r];
			std::cout << "  [" << r << "] VA=0x" << std::hex << re.VirtualAddress << std::dec
				<< "  SymIdx=" << re.SymbolTableIndex
				<< "  Type=0x" << std::hex << re.Type << std::dec;
			// Try to resolve symbol name for clarity
			if (sym_first && re.SymbolTableIndex < fh->NumberOfSymbols) {
				const IMAGE_SYMBOL* s = sym_first + re.SymbolTableIndex;
				std::string sname = ResolveSymbolName(*s, strtab_ptr, strtab_size);
				std::cout << "  ('" << sname << "')";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}

	// Symbols
	std::cout << "=== SYMBOLS (" << nsyms << ") ===\n";
	if (sym_first && nsyms > 0) {
		uint32_t i = 0;
		while (i < nsyms) {
			const IMAGE_SYMBOL* s = sym_first + i;
			std::string name = ResolveSymbolName(*s, strtab_ptr, strtab_size);

			std::cout << "[#" << i << "] name='" << name << "'";
			const char* snstr = SectionNumberToString(s->SectionNumber);
			if (snstr) {
				std::cout << "  Sec=" << snstr;
			}
			else {
				std::cout << "  Sec=#" << s->SectionNumber;
			}
			std::cout << "  Val=0x" << std::hex << s->Value << std::dec
				<< "  Type=0x" << std::hex << s->Type << std::dec
				<< "  StorageClass=" << (int)s->StorageClass << " (" << StorageClassToString(s->StorageClass) << ")"
				<< "  AuxCount=" << (int)s->NumberOfAuxSymbols
				<< "\n";

			// If there are auxiliary symbols, skip and optionally print a brief note
			for (uint8_t a = 0; a < s->NumberOfAuxSymbols; ++a) {
				uint32_t aux_idx = i + 1 + a;
				if (aux_idx >= nsyms) break;
				const IMAGE_SYMBOL* aux = sym_first + aux_idx;
				// We do not attempt to decode all aux formats here; just show raw 18 bytes.
				const uint8_t* raw = reinterpret_cast<const uint8_t*>(aux);
				std::cout << "   AUX[#" << (int)a << "]:";
				for (int b = 0; b < 18; ++b) {
					std::cout << ' ' << std::hex << std::setw(2) << std::setfill('0') << (int)raw[b];
				}
				std::cout << std::dec << "\n";
			}

			i += 1 + s->NumberOfAuxSymbols;
		}
	}

	// String table summary
	if (strtab_ptr && strtab_size >= 4) {
		std::cout << "\n=== STRING TABLE ===\n";
		std::cout << "Total size: " << strtab_size << " bytes (including 4-byte size header)\n";
		// Optionally preview first bytes
		size_t preview = std::min<size_t>(strtab_size - 4, 64);
		std::cout << "Preview:";
		HexDump(strtab_ptr, preview, preview);
	}

	std::cout << "\nDone.\n";
	return 0;
}
