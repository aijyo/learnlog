#include "./render_buffer.h"

namespace utils
{

	RenderFileBuffer::RenderFileBuffer()
	{

	}

	RenderFileBuffer::~RenderFileBuffer()
	{
		Close();
	}

	int RenderFileBuffer::Open(const std::string& filename, bool readCyclic)
	{
		int result = 0;

		do
		{
			_file.open(filename, std::ios::binary | std::ios::in);
			if (!_file.is_open())
			{
				result = 1;
				break;
			}

			// get file length
			_file.seekg(0, std::ios::end);
			_fileLength = _file.tellg();
			_file.seekg(0, std::ios::beg);
			_curPosition = 0;

			_readCyclic = readCyclic;

		} while (false);

		return result;
	}

	int RenderFileBuffer::Close()
	{
		int result = 0;

		_buffer.clear();
		_file.close();

		return result;
	}

	bool RenderFileBuffer::IsOk() const
	{
		return _file.is_open() && _file.good() && _fileLength != 0;
	}

	// input: expect len
	// return data ptr && real length of data
	const uint8_t* RenderFileBuffer::Read(std::uint32_t& len)
	{
		const uint8_t* result = nullptr;

		do
		{

			if (!(_file.is_open()
				&& _file.good()) || 0 == _fileLength)
			{
				break;
			}

			if (_buffer.size() < len)
				_buffer.resize(len/*+1*/);

			auto pBuf = (char*)_buffer.data();
			std::uint64_t pos = 0;
			std::uint64_t readLen = 0;

			while (readLen < len)
			{
				auto leftLength = _fileLength - _curPosition;
				auto readCount = std::min(leftLength, (std::uint64_t)len - readLen);
				_file.read(pBuf + pos, readCount);

				// update pos
				_curPosition += readCount;
				pos += readCount;
				readLen += readCount;

				if (!_readCyclic)
					break;

				// read cyclic
				//if (_file.eof())
				//{
				//}
				// read all
				if (_curPosition == _fileLength)
				{
					_file.seekg(0, std::ios::beg);
					_curPosition = 0;
				}
			}
			result = (uint8_t*)pBuf;
			len = readLen;
		} while (false);
		return result;
	}
	std::uint64_t RenderFileBuffer::Length()
	{
		return _fileLength;
	}
}
