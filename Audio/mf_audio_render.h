https://github.com/sipsorcery/mediafoundationsamples

#include <stdio.h>
#include <iostream>
#include <string>
#include <tchar.h>
#include <mfapi.h>
#include <mfplay.h>
#include <mfreadwrite.h>
#include <mmdeviceapi.h>
#include <mferror.h>

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfplay.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

#define MEDIA_FILE_PATH L"D:/code/gitcode/mediafoundationsamples/MediaFiles/Macroform_-_Simplicity.mp3"
#define AUDIO_DEVICE_INDEX 0 // Select the first audio rendering device returned by the system.


#define CHECK_HR(hr, msg) if (hr != S_OK) { printf(msg); printf(" Error: %.2X.\n", hr); goto done; }

#define CHECKHR_GOTO(x, y) if(FAILED(x)) goto y

#define INTERNAL_GUID_TO_STRING( _Attribute, _skip ) \
if (Attr == _Attribute) \
{ \
	pAttrStr = #_Attribute; \
	C_ASSERT((sizeof(#_Attribute) / sizeof(#_Attribute[0])) > _skip); \
	pAttrStr += _skip; \
	goto done; \
} \

template <class T> void SAFE_RELEASE(T** ppT)
{
    if (*ppT)
    {
        (*ppT)->Release();
        *ppT = NULL;
    }
}

template <class T> inline void SAFE_RELEASE(T*& pT)
{
    if (pT != NULL)
    {
        pT->Release();
        pT = NULL;
    }
}


/**
* Attempts to print out a list of all the audio output devices
* available on the system.
* @@Returns S_OK if successful or an error code if not.
*
* Remarks:
* See https://docs.microsoft.com/en-us/windows/win32/medfound/streaming-audio-renderer.
*/
HRESULT ListAudioOutputDevices()
{
    HRESULT hr = S_OK;

    IMMDeviceEnumerator* pEnum = NULL;      // Audio device enumerator.
    IMMDeviceCollection* pDevices = NULL;   // Audio device collection.
    IMMDevice* pDevice = NULL;              // An audio device.
    UINT deviceCount = 0;

    // Create the device enumerator.
    hr = CoCreateInstance(
        __uuidof(MMDeviceEnumerator),
        NULL,
        CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator),
        (void**)&pEnum
    );

    // Enumerate the rendering devices.
    hr = pEnum->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &pDevices);
    CHECK_HR(hr, "Failed to enumerate audio end points.");

    hr = pDevices->GetCount(&deviceCount);
    CHECK_HR(hr, "Failed to get audio end points count.");

    std::cout << "Audio output device count " << deviceCount << "." << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        LPWSTR wstrID = NULL;                   // Device ID.

        hr = pDevices->Item(i, &pDevice);
        CHECK_HR(hr, "Failed to get device for ID.");

        hr = pDevice->GetId(&wstrID);
        CHECK_HR(hr, "Failed to get name for device.");

        std::wcout << "Audio output device " << i << ": " << wstrID << "." << std::endl;

        CoTaskMemFree(wstrID);
    }

done:

    SAFE_RELEASE(pEnum);
    SAFE_RELEASE(pDevices);
    SAFE_RELEASE(pDevice);

    return hr;
}

#ifndef IF_EQUAL_RETURN
#define IF_EQUAL_RETURN(param, val) if(val == param) return #val
#endif

LPCSTR GetGUIDNameConst(const GUID& guid)
{
    IF_EQUAL_RETURN(guid, MF_MT_MAJOR_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_MAJOR_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_SUBTYPE);
    IF_EQUAL_RETURN(guid, MF_MT_ALL_SAMPLES_INDEPENDENT);
    IF_EQUAL_RETURN(guid, MF_MT_FIXED_SIZE_SAMPLES);
    IF_EQUAL_RETURN(guid, MF_MT_COMPRESSED);
    IF_EQUAL_RETURN(guid, MF_MT_SAMPLE_SIZE);
    IF_EQUAL_RETURN(guid, MF_MT_WRAPPED_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_NUM_CHANNELS);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_SAMPLES_PER_SECOND);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_FLOAT_SAMPLES_PER_SECOND);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_AVG_BYTES_PER_SECOND);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_BLOCK_ALIGNMENT);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_BITS_PER_SAMPLE);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_VALID_BITS_PER_SAMPLE);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_SAMPLES_PER_BLOCK);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_CHANNEL_MASK);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_FOLDDOWN_MATRIX);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_PEAKREF);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_PEAKTARGET);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_AVGREF);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_AVGTARGET);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_PREFER_WAVEFORMATEX);
    IF_EQUAL_RETURN(guid, MF_MT_AAC_PAYLOAD_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION);
    IF_EQUAL_RETURN(guid, MF_MT_FRAME_SIZE);
    IF_EQUAL_RETURN(guid, MF_MT_FRAME_RATE);
    IF_EQUAL_RETURN(guid, MF_MT_FRAME_RATE_RANGE_MAX);
    IF_EQUAL_RETURN(guid, MF_MT_FRAME_RATE_RANGE_MIN);
    IF_EQUAL_RETURN(guid, MF_MT_PIXEL_ASPECT_RATIO);
    IF_EQUAL_RETURN(guid, MF_MT_DRM_FLAGS);
    IF_EQUAL_RETURN(guid, MF_MT_PAD_CONTROL_FLAGS);
    IF_EQUAL_RETURN(guid, MF_MT_SOURCE_CONTENT_HINT);
    IF_EQUAL_RETURN(guid, MF_MT_VIDEO_CHROMA_SITING);
    IF_EQUAL_RETURN(guid, MF_MT_INTERLACE_MODE);
    IF_EQUAL_RETURN(guid, MF_MT_TRANSFER_FUNCTION);
    IF_EQUAL_RETURN(guid, MF_MT_VIDEO_PRIMARIES);
    IF_EQUAL_RETURN(guid, MF_MT_CUSTOM_VIDEO_PRIMARIES);
    IF_EQUAL_RETURN(guid, MF_MT_YUV_MATRIX);
    IF_EQUAL_RETURN(guid, MF_MT_VIDEO_LIGHTING);
    IF_EQUAL_RETURN(guid, MF_MT_VIDEO_NOMINAL_RANGE);
    IF_EQUAL_RETURN(guid, MF_MT_GEOMETRIC_APERTURE);
    IF_EQUAL_RETURN(guid, MF_MT_MINIMUM_DISPLAY_APERTURE);
    IF_EQUAL_RETURN(guid, MF_MT_PAN_SCAN_APERTURE);
    IF_EQUAL_RETURN(guid, MF_MT_PAN_SCAN_ENABLED);
    IF_EQUAL_RETURN(guid, MF_MT_AVG_BITRATE);
    IF_EQUAL_RETURN(guid, MF_MT_AVG_BIT_ERROR_RATE);
    IF_EQUAL_RETURN(guid, MF_MT_MAX_KEYFRAME_SPACING);
    IF_EQUAL_RETURN(guid, MF_MT_DEFAULT_STRIDE);
    IF_EQUAL_RETURN(guid, MF_MT_PALETTE);
    IF_EQUAL_RETURN(guid, MF_MT_USER_DATA);
    IF_EQUAL_RETURN(guid, MF_MT_AM_FORMAT_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG_START_TIME_CODE);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG2_PROFILE);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG2_LEVEL);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG2_FLAGS);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG_SEQUENCE_HEADER);
    IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_SRC_PACK_0);
    IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_CTRL_PACK_0);
    IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_SRC_PACK_1);
    IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_CTRL_PACK_1);
    IF_EQUAL_RETURN(guid, MF_MT_DV_VAUX_SRC_PACK);
    IF_EQUAL_RETURN(guid, MF_MT_DV_VAUX_CTRL_PACK);
    IF_EQUAL_RETURN(guid, MF_MT_ARBITRARY_HEADER);
    IF_EQUAL_RETURN(guid, MF_MT_ARBITRARY_FORMAT);
    IF_EQUAL_RETURN(guid, MF_MT_IMAGE_LOSS_TOLERANT);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG4_SAMPLE_DESCRIPTION);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG4_CURRENT_SAMPLE_ENTRY);
    IF_EQUAL_RETURN(guid, MF_MT_ORIGINAL_4CC);
    IF_EQUAL_RETURN(guid, MF_MT_ORIGINAL_WAVE_FORMAT_TAG);

    // Media types

    IF_EQUAL_RETURN(guid, MFMediaType_Audio);
    IF_EQUAL_RETURN(guid, MFMediaType_Video);
    IF_EQUAL_RETURN(guid, MFMediaType_Protected);
    IF_EQUAL_RETURN(guid, MFMediaType_SAMI);
    IF_EQUAL_RETURN(guid, MFMediaType_Script);
    IF_EQUAL_RETURN(guid, MFMediaType_Image);
    IF_EQUAL_RETURN(guid, MFMediaType_HTML);
    IF_EQUAL_RETURN(guid, MFMediaType_Binary);
    IF_EQUAL_RETURN(guid, MFMediaType_FileTransfer);

    IF_EQUAL_RETURN(guid, MFVideoFormat_AI44); //     FCC('AI44')
    IF_EQUAL_RETURN(guid, MFVideoFormat_ARGB32); //   D3DFMT_A8R8G8B8 
    IF_EQUAL_RETURN(guid, MFVideoFormat_AYUV); //     FCC('AYUV')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DV25); //     FCC('dv25')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DV50); //     FCC('dv50')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DVH1); //     FCC('dvh1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DVSD); //     FCC('dvsd')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DVSL); //     FCC('dvsl')
    IF_EQUAL_RETURN(guid, MFVideoFormat_H264); //     FCC('H264')
    IF_EQUAL_RETURN(guid, MFVideoFormat_I420); //     FCC('I420')
    IF_EQUAL_RETURN(guid, MFVideoFormat_IYUV); //     FCC('IYUV')
    IF_EQUAL_RETURN(guid, MFVideoFormat_M4S2); //     FCC('M4S2')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MJPG);
    IF_EQUAL_RETURN(guid, MFVideoFormat_MP43); //     FCC('MP43')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MP4S); //     FCC('MP4S')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MP4V); //     FCC('MP4V')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MPG1); //     FCC('MPG1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MSS1); //     FCC('MSS1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MSS2); //     FCC('MSS2')
    IF_EQUAL_RETURN(guid, MFVideoFormat_NV11); //     FCC('NV11')
    IF_EQUAL_RETURN(guid, MFVideoFormat_NV12); //     FCC('NV12')
    IF_EQUAL_RETURN(guid, MFVideoFormat_P010); //     FCC('P010')
    IF_EQUAL_RETURN(guid, MFVideoFormat_P016); //     FCC('P016')
    IF_EQUAL_RETURN(guid, MFVideoFormat_P210); //     FCC('P210')
    IF_EQUAL_RETURN(guid, MFVideoFormat_P216); //     FCC('P216')
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB24); //    D3DFMT_R8G8B8 
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB32); //    D3DFMT_X8R8G8B8 
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB555); //   D3DFMT_X1R5G5B5 
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB565); //   D3DFMT_R5G6B5 
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB8);
    IF_EQUAL_RETURN(guid, MFVideoFormat_UYVY); //     FCC('UYVY')
    IF_EQUAL_RETURN(guid, MFVideoFormat_v210); //     FCC('v210')
    IF_EQUAL_RETURN(guid, MFVideoFormat_v410); //     FCC('v410')
    IF_EQUAL_RETURN(guid, MFVideoFormat_WMV1); //     FCC('WMV1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_WMV2); //     FCC('WMV2')
    IF_EQUAL_RETURN(guid, MFVideoFormat_WMV3); //     FCC('WMV3')
    IF_EQUAL_RETURN(guid, MFVideoFormat_WVC1); //     FCC('WVC1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y210); //     FCC('Y210')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y216); //     FCC('Y216')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y410); //     FCC('Y410')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y416); //     FCC('Y416')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y41P);
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y41T);
    IF_EQUAL_RETURN(guid, MFVideoFormat_YUY2); //     FCC('YUY2')
    IF_EQUAL_RETURN(guid, MFVideoFormat_YV12); //     FCC('YV12')
    IF_EQUAL_RETURN(guid, MFVideoFormat_YVYU);

    IF_EQUAL_RETURN(guid, MFAudioFormat_PCM); //              WAVE_FORMAT_PCM 
    IF_EQUAL_RETURN(guid, MFAudioFormat_Float); //            WAVE_FORMAT_IEEE_FLOAT 
    IF_EQUAL_RETURN(guid, MFAudioFormat_DTS); //              WAVE_FORMAT_DTS 
    IF_EQUAL_RETURN(guid, MFAudioFormat_Dolby_AC3_SPDIF); //  WAVE_FORMAT_DOLBY_AC3_SPDIF 
    IF_EQUAL_RETURN(guid, MFAudioFormat_DRM); //              WAVE_FORMAT_DRM 
    IF_EQUAL_RETURN(guid, MFAudioFormat_WMAudioV8); //        WAVE_FORMAT_WMAUDIO2 
    IF_EQUAL_RETURN(guid, MFAudioFormat_WMAudioV9); //        WAVE_FORMAT_WMAUDIO3 
    IF_EQUAL_RETURN(guid, MFAudioFormat_WMAudio_Lossless); // WAVE_FORMAT_WMAUDIO_LOSSLESS 
    IF_EQUAL_RETURN(guid, MFAudioFormat_WMASPDIF); //         WAVE_FORMAT_WMASPDIF 
    IF_EQUAL_RETURN(guid, MFAudioFormat_MSP1); //             WAVE_FORMAT_WMAVOICE9 
    IF_EQUAL_RETURN(guid, MFAudioFormat_MP3); //              WAVE_FORMAT_MPEGLAYER3 
    IF_EQUAL_RETURN(guid, MFAudioFormat_MPEG); //             WAVE_FORMAT_MPEG 
    IF_EQUAL_RETURN(guid, MFAudioFormat_AAC); //              WAVE_FORMAT_MPEG_HEAAC 
    IF_EQUAL_RETURN(guid, MFAudioFormat_ADTS); //             WAVE_FORMAT_MPEG_ADTS_AAC 

    return NULL;
}

/**
* Helper function to get a user friendly description for a media type.
* Note that there may be properties missing or incorrectly described.
* @param[in] pMediaType: pointer to the media type to get a description for.
* @@Returns A string describing the media type.
*
* Potential improvements https://docs.microsoft.com/en-us/windows/win32/medfound/media-type-debugging-code.
*/
std::string GetMediaTypeDescription(IMFMediaType* pMediaType)
{
    HRESULT hr = S_OK;
    GUID MajorType;
    UINT32 cAttrCount;
    LPCSTR pszGuidStr;
    std::string description;
    WCHAR TempBuf[200];

    if (pMediaType == NULL)
    {
        description = "<NULL>";
        goto done;
    }

    hr = pMediaType->GetMajorType(&MajorType);
    CHECKHR_GOTO(hr, done);

    //pszGuidStr = STRING_FROM_GUID(MajorType);
    pszGuidStr = GetGUIDNameConst(MajorType);
    if (pszGuidStr != NULL)
    {
        description += pszGuidStr;
        description += ": ";
    }
    else
    {
        description += "Other: ";
    }

    hr = pMediaType->GetCount(&cAttrCount);
    CHECKHR_GOTO(hr, done);

    for (UINT32 i = 0; i < cAttrCount; i++)
    {
        GUID guidId;
        MF_ATTRIBUTE_TYPE attrType;

        hr = pMediaType->GetItemByIndex(i, &guidId, NULL);
        CHECKHR_GOTO(hr, done);

        hr = pMediaType->GetItemType(guidId, &attrType);
        CHECKHR_GOTO(hr, done);

        //pszGuidStr = STRING_FROM_GUID(guidId);
        pszGuidStr = GetGUIDNameConst(guidId);
        if (pszGuidStr != NULL)
        {
            description += pszGuidStr;
        }
        else
        {
            LPOLESTR guidStr = NULL;

            CHECKHR_GOTO(StringFromCLSID(guidId, &guidStr), done);
            auto wGuidStr = std::wstring(guidStr);
            description += std::string(wGuidStr.begin(), wGuidStr.end()); // GUID's won't have wide chars.

            CoTaskMemFree(guidStr);
        }

        description += "=";

        switch (attrType)
        {
        case MF_ATTRIBUTE_UINT32:
        {
            UINT32 Val;
            hr = pMediaType->GetUINT32(guidId, &Val);
            CHECKHR_GOTO(hr, done);

            description += std::to_string(Val);
            break;
        }
        case MF_ATTRIBUTE_UINT64:
        {
            UINT64 Val;
            hr = pMediaType->GetUINT64(guidId, &Val);
            CHECKHR_GOTO(hr, done);

            if (guidId == MF_MT_FRAME_SIZE)
            {
                description += "W:" + std::to_string(HI32(Val)) + " H: " + std::to_string(LO32(Val));
            }
            else if (guidId == MF_MT_FRAME_RATE)
            {
                // Frame rate is numerator/denominator.
                description += std::to_string(HI32(Val)) + "/" + std::to_string(LO32(Val));
            }
            else if (guidId == MF_MT_PIXEL_ASPECT_RATIO)
            {
                description += std::to_string(HI32(Val)) + ":" + std::to_string(LO32(Val));
            }
            else
            {
                //tempStr.Format("%ld", Val);
                description += std::to_string(Val);
            }

            //description += tempStr;

            break;
        }
        case MF_ATTRIBUTE_DOUBLE:
        {
            DOUBLE Val;
            hr = pMediaType->GetDouble(guidId, &Val);
            CHECKHR_GOTO(hr, done);

            //tempStr.Format("%f", Val);
            description += std::to_string(Val);
            break;
        }
        case MF_ATTRIBUTE_GUID:
        {
            GUID Val;
            const char* pValStr;

            hr = pMediaType->GetGUID(guidId, &Val);
            CHECKHR_GOTO(hr, done);

            //pValStr = STRING_FROM_GUID(Val);
            pValStr = GetGUIDNameConst(Val);
            if (pValStr != NULL)
            {
                description += pValStr;
            }
            else
            {
                LPOLESTR guidStr = NULL;
                CHECKHR_GOTO(StringFromCLSID(Val, &guidStr), done);
                auto wGuidStr = std::wstring(guidStr);
                description += std::string(wGuidStr.begin(), wGuidStr.end()); // GUID's won't have wide chars.

                CoTaskMemFree(guidStr);
            }

            break;
        }
        case MF_ATTRIBUTE_STRING:
        {
            hr = pMediaType->GetString(guidId, TempBuf, sizeof(TempBuf) / sizeof(TempBuf[0]), NULL);
            if (hr == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER))
            {
                description += "<Too Long>";
                break;
            }
            CHECKHR_GOTO(hr, done);
            auto wstr = std::wstring(TempBuf);
            description += std::string(wstr.begin(), wstr.end()); // It's unlikely the attribute descriptions will contain multi byte chars.

            break;
        }
        case MF_ATTRIBUTE_BLOB:
        {
            description += "<BLOB>";
            break;
        }
        case MF_ATTRIBUTE_IUNKNOWN:
        {
            description += "<UNK>";
            break;
        }
        }

        description += ", ";
    }

done:

    return description;
}


/*
* Attempts to get an audio output sink for the specified device index.
* @param[in] deviceIndex: the audio output device to get the sink for.
* @param[out] ppAudioSink: if successful this parameter will be set with
*  the output sink.
* @@Returns S_OK if successful or an error code if not.
*/
HRESULT GetAudioOutputDevice(UINT deviceIndex, IMFMediaSink** ppAudioSink)
{
    HRESULT hr = S_OK;

    IMMDeviceEnumerator* pEnum = NULL;      // Audio device enumerator.
    IMMDeviceCollection* pDevices = NULL;   // Audio device collection.
    IMMDevice* pDevice = NULL;              // An audio device.
    IMFAttributes* pAttributes = NULL;      // Attribute store.
    LPWSTR wstrID = NULL;                   // Device ID.
    UINT deviceCount = 0;

    // Create the device enumerator.
    hr = CoCreateInstance(
        __uuidof(MMDeviceEnumerator),
        NULL,
        CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator),
        (void**)&pEnum
    );

    // Enumerate the rendering devices.
    hr = pEnum->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &pDevices);
    CHECK_HR(hr, "Failed to enumerate audio end points.");

    hr = pDevices->GetCount(&deviceCount);
    CHECK_HR(hr, "Failed to get audio end points count.");

    if (deviceIndex >= deviceCount) {
        printf("The audio output device index was invalid.\n");
        hr = E_INVALIDARG;
    }
    else {
        hr = pDevices->Item(deviceIndex, &pDevice);
        CHECK_HR(hr, "Failed to get device for ID.");

        hr = pDevice->GetId(&wstrID);
        CHECK_HR(hr, "Failed to get name for device.");

        std::wcout << "Audio output device for index " << deviceIndex << ": " << wstrID << "." << std::endl;

        // Create an attribute store and set the device ID attribute.
        hr = MFCreateAttributes(&pAttributes, 1);
        CHECK_HR(hr, "Failed to create attribute store.");

        hr = pAttributes->SetString(MF_AUDIO_RENDERER_ATTRIBUTE_ENDPOINT_ID, wstrID);
        CHECK_HR(hr, "Failed to set endpoint ID attribute.");

        // Create the audio renderer.
        hr = MFCreateAudioRenderer(pAttributes, ppAudioSink);
        CHECK_HR(hr, "Failed to create the audio output sink.");
    }

done:

    CoTaskMemFree(wstrID);
    SAFE_RELEASE(pEnum);
    SAFE_RELEASE(pDevices);
    SAFE_RELEASE(pDevice);
    SAFE_RELEASE(pAttributes);

    return hr;
}


int main()
{
    IMFSourceReader* pSourceReader = NULL;
    IMFMediaType* pFileAudioMediaType = NULL;
    IMFMediaSink* pAudioSink = NULL;
    IMFStreamSink* pStreamSink = NULL;
    IMFMediaTypeHandler* pSinkMediaTypeHandler = NULL;
    IMFMediaType* pSinkSupportedType = NULL;
    IMFMediaType* pSinkMediaType = NULL;
    IMFSinkWriter* pSinkWriter = NULL;
    DWORD sinkMediaTypeCount = 0;

    CHECK_HR(CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE),
        "COM initialisation failed.");

    CHECK_HR(MFStartup(MF_VERSION),
        "Media Foundation initialisation failed.");

    ListAudioOutputDevices();

    // Source.
    CHECK_HR(MFCreateSourceReaderFromURL(
        MEDIA_FILE_PATH,
        NULL,
        &pSourceReader),
        "Failed to create source reader from file.");

    CHECK_HR(pSourceReader->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, &pFileAudioMediaType),
        "Error retrieving current media type from first audio stream.");

    CHECK_HR(pSourceReader->SetStreamSelection((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, TRUE),
        "Failed to set the first audio stream on the source reader.");

    std::cout << GetMediaTypeDescription(pFileAudioMediaType) << std::endl;

    // Sink.
    CHECK_HR(GetAudioOutputDevice(AUDIO_DEVICE_INDEX, &pAudioSink),
        "Failed to get audio renderer device.");

    CHECK_HR(pAudioSink->GetStreamSinkByIndex(0, &pStreamSink),
        "Failed to get audio renderer stream by index.");

    CHECK_HR(pStreamSink->GetMediaTypeHandler(&pSinkMediaTypeHandler),
        "Failed to get media type handler.");

    CHECK_HR(pSinkMediaTypeHandler->GetMediaTypeCount(&sinkMediaTypeCount),
        "Error getting sink media type count.");

    // Find a media type that the stream sink supports.
    for (UINT i = 0; i < sinkMediaTypeCount; i++)
    {
        CHECK_HR(pSinkMediaTypeHandler->GetMediaTypeByIndex(i, &pSinkSupportedType),
            "Error getting media type from sink media type handler.");

        std::cout << GetMediaTypeDescription(pSinkSupportedType) << std::endl;

        if (pSinkMediaTypeHandler->IsMediaTypeSupported(pSinkSupportedType, NULL) == S_OK) {
            std::cout << "Matching media type found." << std::endl;
            break;
        }
        else {
            std::cout << "Sink media type does not match." << std::endl;
            SAFE_RELEASE(pSinkSupportedType);
        }
    }

    if (pSinkSupportedType != NULL) {
        // Set the supported type on the reader.
        CHECK_HR(pSourceReader->SetCurrentMediaType(0, NULL, pSinkSupportedType),
            "Failed to set media type on reader.");

        CHECK_HR(MFCreateSinkWriterFromMediaSink(pAudioSink, NULL, &pSinkWriter),
            "Failed to create sink writer for default speaker.");

        CHECK_HR(pSinkWriter->SetInputMediaType(0, pSinkSupportedType, NULL),
            "Error setting sink media type.");

        // Start the read-write loop.
        std::cout << "Read audio samples from file and write to speaker." << std::endl;

        CHECK_HR(pSinkWriter->BeginWriting(),
            "Sink writer begin writing call failed.");

        while (true)
        {
            IMFSample* audioSample = NULL;
            DWORD streamIndex, flags;
            LONGLONG llAudioTimeStamp;

            CHECK_HR(pSourceReader->ReadSample(
                MF_SOURCE_READER_FIRST_AUDIO_STREAM,
                0,                              // Flags.
                &streamIndex,                   // Receives the actual stream index. 
                &flags,                         // Receives status flags.
                &llAudioTimeStamp,              // Receives the time stamp.
                &audioSample                    // Receives the sample or NULL.
            ), "Error reading audio sample.");

            if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
            {
                printf("\tEnd of stream");
                break;
            }
            if (flags & MF_SOURCE_READERF_NEWSTREAM)
            {
                printf("\tNew stream\n");
                break;
            }
            if (flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED)
            {
                printf("\tNative type changed\n");
                break;
            }
            if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)
            {
                printf("\tCurrent type changed\n");
                break;
            }
            if (flags & MF_SOURCE_READERF_STREAMTICK)
            {
                printf("Stream tick.\n");
                CHECK_HR(pSinkWriter->SendStreamTick(0, llAudioTimeStamp),
                    "Error sending stream tick.");
            }

            if (!audioSample)
            {
                printf("Null audio sample.\n");
            }
            else
            {
                CHECK_HR(pSinkWriter->WriteSample(0, audioSample),
                    "The stream sink writer was not happy with the sample.");
            }
        }
    }
    else {
        printf("No matching media type could be found.\n");
    }

done:

    printf("finished.\n");
    int c = getchar();

    SAFE_RELEASE(pSourceReader);
    SAFE_RELEASE(pFileAudioMediaType);
    SAFE_RELEASE(pAudioSink);
    SAFE_RELEASE(pStreamSink);
    SAFE_RELEASE(pSinkMediaTypeHandler);
    SAFE_RELEASE(pSinkSupportedType);
    SAFE_RELEASE(pSinkMediaType);
    SAFE_RELEASE(pSinkWriter);

    return 0;
}
