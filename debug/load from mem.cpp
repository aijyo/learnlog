// pe_dryrun_loader.cpp
// Educational, SAFE "dry-run" manual mapping demo for PE DLLs.
// Implements steps 1-7 and 9 of a manual loader WITHOUT executing code:
// 1) Parse headers
// 2) Reserve a simulated image buffer (not executable)
// 3) Copy headers/sections
// 4) Apply base relocations INTO the simulated buffer
// 5) Resolve imports and WRITE the resolved addresses INTO the simulated IAT buffer
// 6) Compute desired section protections (do not call VirtualProtect)
// 7) Enumerate TLS metadata ONLY (do not execute callbacks)
// 9) Compute the VA of OEP and print it (do not call it)
//
// IMPORTANT:
// - No pages are made executable.
// - No entry point or TLS callback is invoked.
// - No changes to the current process's live IAT.
// This is for learning the structures & flow only.
//
// Build: cl /std:c++17 /W4 /EHsc pe_dryrun_loader.cpp

#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#pragma comment(lib, "kernel32.lib")

// ---------- Utility: simple file reader ----------
static bool ReadAllBytes(const wchar_t* path, std::vector<uint8_t>& out) {
    HANDLE h = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER sz{}; GetFileSizeEx(h, &sz);
    if (sz.QuadPart <= 0 || sz.QuadPart > (1ll<<31)) { CloseHandle(h); return false; }
    out.resize(static_cast<size_t>(sz.QuadPart));
    DWORD got=0;
    BOOL ok = ReadFile(h, out.data(), (DWORD)out.size(), &got, nullptr);
    CloseHandle(h);
    return ok && got == out.size();
}

static inline bool InRange(size_t total, size_t off, size_t len) {
    return off <= total && total - off >= len;
}

template<typename T>
static bool ReadChecked(const uint8_t* base, size_t total, size_t off, T& out) {
    if (!InRange(total, off, sizeof(T))) return false;
    std::memcpy(&out, base + off, sizeof(T));
    return true;
}

template<typename T>
static std::wstring Hex(T v) {
    wchar_t buf[32];
    swprintf(buf, 32, L"0x%llX", (unsigned long long)v);
    return buf;
}

// ---------- RVA->FileOff helper ----------
static DWORD RvaToFileOff(const IMAGE_NT_HEADERS* nt, const IMAGE_SECTION_HEADER* sh0, DWORD shn, DWORD rva) {
    if (rva < nt->OptionalHeader.SizeOfHeaders) return rva;
    for (DWORD i=0;i<shn;++i) {
        const auto& sh = sh0[i];
        DWORD va   = sh.VirtualAddress;
        DWORD vlen = sh.Misc.VirtualSize ? sh.Misc.VirtualSize : sh.SizeOfRawData;
        if (rva >= va && rva < va + vlen) {
            return sh.PointerToRawData + (rva - va);
        }
    }
    return 0;
}

// ---------- Protection pretty-printer ----------
static void ExplainProt(DWORD ch, DWORD& wantProt) {
    // Map IMAGE_SCN flags to an intended protection (for explanation only).
    bool X = (ch & IMAGE_SCN_MEM_EXECUTE) != 0;
    bool R = (ch & IMAGE_SCN_MEM_READ)    != 0;
    bool W = (ch & IMAGE_SCN_MEM_WRITE)   != 0;

    if (X && W && R) wantProt = PAGE_EXECUTE_READWRITE;
    else if (X && R) wantProt = PAGE_EXECUTE_READ;
    else if (R && W) wantProt = PAGE_READWRITE;
    else if (R)      wantProt = PAGE_READONLY;
    else wantProt = PAGE_NOACCESS;
}

// ---------- Dry-run loader ----------
struct SimImage {
    // Simulated image buffer (RW only; never executable in this demo)
    std::vector<uint8_t> buf;
    uint8_t* base() { return buf.data(); }
    size_t   size() const { return buf.size(); }
};

static bool DryRunLoad(const std::vector<uint8_t>& file) {
    if (file.size() < sizeof(IMAGE_DOS_HEADER)) {
        std::wcerr << L"[!] Too small\n"; return false;
    }
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(file.data());
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) { std::wcerr << L"[!] Not MZ\n"; return false; }

    if (!InRange(file.size(), (size_t)dos->e_lfanew, sizeof(DWORD) + sizeof(IMAGE_FILE_HEADER))) {
        std::wcerr << L"[!] Bad e_lfanew\n"; return false;
    }
    DWORD sig = *reinterpret_cast<const DWORD*>(file.data() + dos->e_lfanew);
    if (sig != IMAGE_NT_SIGNATURE) { std::wcerr << L"[!] Not PE\n"; return false; }

    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(file.data() + dos->e_lfanew);
    bool is64 = (nt->OptionalHeader.Magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC);
    const auto* sh0 = IMAGE_FIRST_SECTION(nt);
    DWORD shn = nt->FileHeader.NumberOfSections;

    if (!InRange(file.size(), (size_t)((const uint8_t*)sh0 - file.data()), shn * sizeof(IMAGE_SECTION_HEADER))) {
        std::wcerr << L"[!] Truncated section table\n"; return false;
    }

    std::wcout << L"=== PE DLL dry-run ===\n";
    std::wcout << L"PE32+: " << (is64?L"yes":L"no")
               << L"  Sections: " << shn
               << L"\nImageBase: " << Hex(is64
                   ? (unsigned long long)reinterpret_cast<const IMAGE_NT_HEADERS64*>(nt)->OptionalHeader.ImageBase
                   : (unsigned long long)nt->OptionalHeader.ImageBase)
               << L"  SizeOfImage: " << Hex(nt->OptionalHeader.SizeOfImage)
               << L"  EntryRVA: " << Hex(nt->OptionalHeader.AddressOfEntryPoint) << L"\n";

    // Step 2: reserve simulated image buffer
    SimImage sim;
    sim.buf.resize(nt->OptionalHeader.SizeOfImage, 0u);

    // Step 3: copy headers + sections
    std::wcout << L"\n[1-3] Copy headers & sections...\n";
    size_t hdrSz = std::min<size_t>(nt->OptionalHeader.SizeOfHeaders, file.size());
    std::memcpy(sim.base(), file.data(), hdrSz);

    for (DWORD i=0;i<shn;++i) {
        const auto& sh = sh0[i];
        char name[9]={0}; std::memcpy(name, sh.Name, 8);
        std::wcout << L"  " << name
                   << L"  RVA=" << Hex(sh.VirtualAddress)
                   << L"  VSz=" << Hex(sh.Misc.VirtualSize)
                   << L"  RawOff=" << Hex(sh.PointerToRawData)
                   << L"  RawSz=" << Hex(sh.SizeOfRawData) << L"\n";

        if (sh.SizeOfRawData == 0) continue;
        if (!InRange(file.size(), sh.PointerToRawData, sh.SizeOfRawData)) {
            std::wcout << L"    [!] Raw out-of-bounds, skip\n"; continue;
        }
        if (!InRange(sim.size(), sh.VirtualAddress, sh.SizeOfRawData)) {
            std::wcout << L"    [!] Sim buffer out-of-bounds, skip\n"; continue;
        }
        std::memcpy(sim.base() + sh.VirtualAddress, file.data() + sh.PointerToRawData, sh.SizeOfRawData);
    }

    // Step 4: apply relocations (simulate a different base to force work)
    std::wcout << L"\n[4] Relocations...\n";
    unsigned long long imgBase = is64
        ? reinterpret_cast<const IMAGE_NT_HEADERS64*>(nt)->OptionalHeader.ImageBase
        : nt->OptionalHeader.ImageBase;
    unsigned long long mappedBase = imgBase + 0x200000; // simulated target base
    long long delta = (long long)(mappedBase - imgBase);

    const auto& ddRel = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC];
    if (ddRel.VirtualAddress && ddRel.Size) {
        DWORD relOff = RvaToFileOff(nt, sh0, shn, ddRel.VirtualAddress);
        if (!relOff || !InRange(file.size(), relOff, ddRel.Size)) {
            std::wcout << L"  [!] Reloc directory inaccessible, skip\n";
        } else {
            size_t pos = relOff, end = relOff + ddRel.Size;
            size_t patched = 0, blocks = 0;

            while (pos + sizeof(IMAGE_BASE_RELOCATION) <= end) {
                auto br = reinterpret_cast<const IMAGE_BASE_RELOCATION*>(file.data() + pos);
                if (br->SizeOfBlock == 0) break;
                DWORD pageRVA = br->VirtualAddress;
                size_t n = (br->SizeOfBlock - sizeof(IMAGE_BASE_RELOCATION)) / sizeof(WORD);
                auto entries = reinterpret_cast<const WORD*>(file.data() + pos + sizeof(IMAGE_BASE_RELOCATION));

                for (size_t i=0;i<n;++i) {
                    WORD e = entries[i];
                    WORD type = e >> 12;
                    WORD off  = e & 0x0FFF;
                    DWORD rva = pageRVA + off;
                    if (!InRange(sim.size(), rva, (type==IMAGE_REL_BASED_DIR64)?8:4)) continue;

                    uint8_t* p = sim.base() + rva;
                    if (type == IMAGE_REL_BASED_HIGHLOW) {
                        DWORD v; std::memcpy(&v, p, 4);
                        v = (DWORD)((long long)v + delta);
                        std::memcpy(p, &v, 4);
                        ++patched;
                    } else if (type == IMAGE_REL_BASED_DIR64) {
                        unsigned long long v; std::memcpy(&v, p, 8);
                        v = (unsigned long long)((long long)v + delta);
                        std::memcpy(p, &v, 8);
                        ++patched;
                    } else if (type == IMAGE_REL_BASED_ABSOLUTE) {
                        // skip
                    }
                }
                ++blocks;
                pos += br->SizeOfBlock;
            }
            std::wcout << L"  blocks=" << blocks << L"  patched=" << patched << L"\n";
        }
    } else {
        std::wcout << L"  No relocation directory\n";
    }

    // Step 5: resolve imports and WRITE addresses into the simulated IAT
    std::wcout << L"\n[5] Imports (resolve + write to simulated IAT)...\n";
    const auto& ddImp = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (ddImp.VirtualAddress && ddImp.Size) {
        DWORD impOff = RvaToFileOff(nt, sh0, shn, ddImp.VirtualAddress);
        if (!impOff || !InRange(file.size(), impOff, sizeof(IMAGE_IMPORT_DESCRIPTOR))) {
            std::wcout << L"  [!] Import directory inaccessible\n";
        } else {
            size_t cur = impOff;
            while (true) {
                if (!InRange(file.size(), cur, sizeof(IMAGE_IMPORT_DESCRIPTOR))) break;
                auto desc = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(file.data() + cur);
                if (desc->OriginalFirstThunk == 0 && desc->FirstThunk == 0) break;

                DWORD nameOff = RvaToFileOff(nt, sh0, shn, desc->Name);
                const char* dll = nameOff ? reinterpret_cast<const char*>(file.data() + nameOff) : "<bad>";
                std::wcout << L"  DLL: " << (dll ? dll : "<bad>") << L"\n";

                HMODULE h = dll ? LoadLibraryA(dll) : nullptr;
                if (!h) std::wcout << L"    [!] LoadLibrary failed, gle=" << GetLastError() << L"\n";

                DWORD oftOff = RvaToFileOff(nt, sh0, shn, desc->OriginalFirstThunk);
                DWORD ftOff  = RvaToFileOff(nt, sh0, shn, desc->FirstThunk);
                if (!oftOff || !ftOff) { std::wcout << L"    [!] Bad thunk RVAs\n"; cur += sizeof(*desc); continue; }

                // IMPORTANT: we PATCH the simulated image's IAT bytes (not the process IAT)
                if (is64) {
                    size_t idx = 0;
                    while (InRange(file.size(), oftOff + idx*sizeof(ULONGLONG), sizeof(ULONGLONG)) &&
                           InRange(sim.size(),  desc->FirstThunk + idx*sizeof(ULONGLONG), sizeof(ULONGLONG))) {
                        ULONGLONG thunk = *reinterpret_cast<const ULONGLONG*>(file.data() + oftOff + idx*sizeof(ULONGLONG));
                        if (!thunk) break;
                        FARPROC addr = nullptr;
                        if (thunk & IMAGE_ORDINAL_FLAG64) {
                            WORD ord = (WORD)(thunk & 0xFFFF);
                            if (h) addr = GetProcAddress(h, MAKEINTRESOURCEA(ord));
                            std::wcout << L"    Ordinal #" << ord << L" -> " << Hex((uintptr_t)addr) << L"\n";
                        } else {
                            DWORD nameRVA = (DWORD)thunk;
                            DWORD nmOff = RvaToFileOff(nt, sh0, shn, nameRVA);
                            const IMAGE_IMPORT_BY_NAME* ibn = nmOff && InRange(file.size(), nmOff, sizeof(WORD)+1)
                                ? reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(file.data() + nmOff) : nullptr;
                            const char* fn = ibn ? (const char*)ibn->Name : nullptr;
                            if (h && fn) addr = GetProcAddress(h, fn);
                            std::wcout << L"    " << (fn?fn:"<bad>") << L" -> " << Hex((uintptr_t)addr) << L"\n";
                        }
                        // Write into simulated IAT (DIR64 pointer)
                        std::memcpy(sim.base() + desc->FirstThunk + idx*sizeof(ULONGLONG),
                                    &addr, sizeof(ULONGLONG));
                        ++idx;
                    }
                } else {
                    size_t idx = 0;
                    while (InRange(file.size(), oftOff + idx*sizeof(DWORD), sizeof(DWORD)) &&
                           InRange(sim.size(),  desc->FirstThunk + idx*sizeof(DWORD), sizeof(DWORD))) {
                        DWORD thunk = *reinterpret_cast<const DWORD*>(file.data() + oftOff + idx*sizeof(DWORD));
                        if (!thunk) break;
                        FARPROC addr = nullptr;
                        if (thunk & IMAGE_ORDINAL_FLAG32) {
                            WORD ord = (WORD)(thunk & 0xFFFF);
                            if (h) addr = GetProcAddress(h, MAKEINTRESOURCEA(ord));
                            std::wcout << L"    Ordinal #" << ord << L" -> " << Hex((uintptr_t)addr) << L"\n";
                        } else {
                            DWORD nmOff = RvaToFileOff(nt, sh0, shn, thunk);
                            const IMAGE_IMPORT_BY_NAME* ibn = nmOff && InRange(file.size(), nmOff, sizeof(WORD)+1)
                                ? reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(file.data() + nmOff) : nullptr;
                            const char* fn = ibn ? (const char*)ibn->Name : nullptr;
                            if (h && fn) addr = GetProcAddress(h, fn);
                            std::wcout << L"    " << (fn?fn:"<bad>") << L" -> " << Hex((uintptr_t)addr) << L"\n";
                        }
                        // Write into simulated IAT (HIGHLOW pointer)
                        std::memcpy(sim.base() + desc->FirstThunk + idx*sizeof(DWORD),
                                    &addr, sizeof(DWORD));
                        ++idx;
                    }
                }

                cur += sizeof(IMAGE_IMPORT_DESCRIPTOR);
            }
        }
    } else {
        std::wcout << L"  No imports\n";
    }

    // Step 6: section protection planning (explain only)
    std::wcout << L"\n[6] Section protection plan (no VirtualProtect here):\n";
    for (DWORD i=0;i<shn;++i) {
        const auto& sh = sh0[i];
        DWORD prot = 0; ExplainProt(sh.Characteristics, prot);
        char name[9]={0}; std::memcpy(name, sh.Name, 8);
        std::wcout << L"  " << name << L" -> ";
        switch (prot) {
            case PAGE_EXECUTE_READWRITE: std::wcout << L"PAGE_EXECUTE_READWRITE"; break;
            case PAGE_EXECUTE_READ:      std::wcout << L"PAGE_EXECUTE_READ"; break;
            case PAGE_READWRITE:         std::wcout << L"PAGE_READWRITE"; break;
            case PAGE_READONLY:          std::wcout << L"PAGE_READONLY"; break;
            default:                     std::wcout << L"PAGE_NOACCESS"; break;
        }
        std::wcout << L"\n";
    }

    // Step 7: TLS metadata enumeration (do NOT execute)
    std::wcout << L"\n[7] TLS metadata (enumeration only; no callbacks executed):\n";
    const auto& ddTls = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
    if (ddTls.VirtualAddress && ddTls.Size) {
        DWORD tlsOff = RvaToFileOff(nt, sh0, shn, ddTls.VirtualAddress);
        if (tlsOff) {
            if (is64) {
                if (InRange(file.size(), tlsOff, sizeof(IMAGE_TLS_DIRECTORY64))) {
                    const auto* tls = reinterpret_cast<const IMAGE_TLS_DIRECTORY64*>(file.data() + tlsOff);
                    std::wcout << L"  TLS callbacks VA list: " << Hex(tls->AddressOfCallBacks) << L"\n";
                }
            } else {
                if (InRange(file.size(), tlsOff, sizeof(IMAGE_TLS_DIRECTORY32))) {
                    const auto* tls = reinterpret_cast<const IMAGE_TLS_DIRECTORY32*>(file.data() + tlsOff);
                    std::wcout << L"  TLS callbacks VA list: " << Hex(tls->AddressOfCallBacks) << L"\n";
                }
            }
        } else {
            std::wcout << L"  [!] TLS directory not readable via file offset\n";
        }
    } else {
        std::wcout << L"  No TLS\n";
    }

    // Step 9: compute OEP VA (do NOT call)
    unsigned long long oepVA = mappedBase + nt->OptionalHeader.AddressOfEntryPoint;
    std::wcout << L"\n[9] Entry point (NOT calling): RVA=" << Hex(nt->OptionalHeader.AddressOfEntryPoint)
               << L"  simulated VA=" << Hex(oepVA) << L"\n";

    std::wcout << L"\n[Done] Dry-run completed safely. No code executed.\n";
    return true;
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) {
        std::wcout << L"Usage: " << argv[0] << L" <path-to-dll>\n";
        return 1;
    }
    std::vector<uint8_t> file;
    if (!ReadAllBytes(argv[1], file)) { std::wcerr << L"[!] Failed to read file\n"; return 2; }
    if (!DryRunLoad(file))            { std::wcerr << L"[!] Dry-run failed\n";     return 3; }
    return 0;
}
