#pragma once

#include <cstdint>
#include <string>
#include <fstream>
#include <vector>

namespace utils
{
	class RenderFileBuffer
	{
	public:
		RenderFileBuffer();
		virtual ~RenderFileBuffer();

		int Open(const std::string& filename, bool readCyclic);
		int Close();
		bool IsOk() const;

		// input: expect len
		// return data ptr && real length of data
		const uint8_t* Read(std::uint32_t& len);
		std::uint64_t Length();
	protected:
		bool _readCyclic = false;
		std::uint64_t _bufferLength = 0;
		std::uint64_t _fileLength = 0;
		std::uint64_t _curPosition = 0;				//_file.tellg()
		std::vector<std::uint8_t> _buffer;
		std::string _fileName;
		std::ifstream _file;
	};

}
