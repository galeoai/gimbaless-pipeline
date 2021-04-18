#include "kaya_interface.h"
#include "KYFGLib_defines.h"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <cuda_runtime.h>

void Stream_callback_func(STREAM_BUFFER_HANDLE streamBufferHandle,
			  void *userContext) {
    if (0 == userContext) { return; }
    auto config = (kaya_config *)userContext;

    unsigned char* pFrameMemory = 0;
    int64_t totalFrames = -1;
    uint32_t frameId = 0;
    size_t bufferSize = 0;
    void* pUserContext;
    uint64_t timeStamp;
    double instantFps;

    if (!streamBufferHandle)
    {
        // this callback indicates that acquisition has stopped
        return;
    }
    // get data pointer (pFramememory)
    KYFG_BufferGetInfo(streamBufferHandle, 
                        KY_STREAM_BUFFER_INFO_BASE, 
                        &pFrameMemory, 
                        NULL, 
                        NULL);
    //Get Data
    KYFG_BufferGetInfo(streamBufferHandle, 
                        KY_STREAM_BUFFER_INFO_ID, 
                        &frameId, 
                        NULL, 
                        NULL);

    KYFG_BufferGetInfo(streamBufferHandle, 
                        KY_STREAM_BUFFER_INFO_SIZE, 
                        &bufferSize, 
                        NULL, 
                        NULL);

    KYFG_BufferGetInfo(streamBufferHandle, 
                        KY_STREAM_BUFFER_INFO_USER_PTR, 
                        &pUserContext, 
                        NULL, 
                        NULL);

    KYFG_BufferGetInfo(streamBufferHandle, 
                        KY_STREAM_BUFFER_INFO_TIMESTAMP, 
                        &timeStamp, 
                        NULL, 
                        NULL);

    KYFG_BufferGetInfo(streamBufferHandle, 
                        KY_STREAM_BUFFER_INFO_INSTANTFPS, 
                        &instantFps, 
                        NULL, 
                        NULL);

    //printf(//"\n" // Uncomment to print on new line each time
    //        "\rGood callback stream's buffer handle:%" PRISTREAM_BUFFER_HANDLE ", ID:%d", streamBufferHandle, frameId);

    if (nullptr == config->process) { return; }

    cv::Mat image(config->height, config->width, CV_8UC1, pFrameMemory);  // TODO: make CV_8UC1 depend on the pixelformat
    
    try {
	config->process(image);
    } catch (...) {
	std::cout << "Fatal Error!"
		  << "\n";
	return;
	stop(*config);
    }
    if (config->totalFrames % 10 == 0) {
	image.copyTo(config->image);
    }
    
    KYFG_BufferToQueue(streamBufferHandle, KY_ACQ_QUEUE_INPUT);
}

bool setup(kaya_config &config) {
    KYFGLib_InitParameters kyInit;
    kyInit.version = 2;
    kyInit.concurrency_mode = 0;
    kyInit.logging_mode = 0;
    kyInit.noVideoStreamProcess = KYFALSE;
    if (FGSTATUS_OK != KYFGLib_Initialize(&kyInit)) {
        printf("Library initialization failed \n ");
        return false;
    }
    int infosize = 0;
    KY_DeviceScan(&infosize);  // Retrieve the number of virtual and hardware
    if ((config.handle = KYFG_Open(config.grabberIndex)) != -1) {
        printf("FG %d was connected successfully\n", config.grabberIndex);
    } else {
        printf("Couldn't connect to FG\n");
        return false;
    }

    int detectionCount = 4;
    KYFG_UpdateCameraList(config.handle, config.camHandleArray, &detectionCount);
    printf("detection count: %d\n", detectionCount);
    if (FGSTATUS_OK == KYFG_CameraOpen2(config.camHandleArray[config.cameraIndex], 0)) {
        printf("Camera was connected successfully\n");
    } else {
        printf("Camera isn't connected\n");
        return false;
    }
    KYFG_SetCameraValueInt(config.camHandleArray[config.cameraIndex],
                           "Width",
                           config.width);
    KYFG_SetCameraValueInt(config.camHandleArray[config.cameraIndex],
                           "Height",
                           config.height);

    KYFG_SetCameraValueInt(config.camHandleArray[config.cameraIndex],
                           "OffsetX",
                           config.offsetx);

    KYFG_SetCameraValueInt(config.camHandleArray[config.cameraIndex],
                           "OffsetY",
                           config.offsety);

    KYFG_SetCameraValueFloat(config.camHandleArray[config.cameraIndex],
                             "AcquisitionFrameRate",
                             config.fps);

    config.fps = KYFG_GetCameraValueFloat(config.camHandleArray[config.cameraIndex], "AcquisitionFrameRate");

    KYFG_SetCameraValueFloat(config.camHandleArray[config.cameraIndex],
                             "ExposureTime",
                             config.exposure);

    config.exposure = KYFG_GetCameraValueFloat(config.camHandleArray[config.cameraIndex], "ExposureTime");

    KYFG_SetCameraValueEnum_ByValueName(config.camHandleArray[config.cameraIndex],
                                        "PixelFormat",
                                        config.pixelFormat.c_str());
    
    KYFG_StreamCreate(config.camHandleArray[config.cameraIndex], &config.cameraStreamHandle, 0);
    KYFG_StreamBufferCallbackRegister(config.cameraStreamHandle, Stream_callback_func, (void *)&config);

    return true;
}

bool start(kaya_config &config) {
    for (auto iFrame = 0; iFrame < 16; iFrame++) {
	size_t frameDataSize;
	KYFG_StreamGetInfo(config.cameraStreamHandle, 
			   KY_STREAM_INFO_PAYLOAD_SIZE, 
			   &frameDataSize, 
			   NULL, NULL);
	void *pBuffer;
	cudaHostAlloc(&pBuffer, frameDataSize, 
		      cudaHostAllocDefault);
	
	auto ret = KYFG_BufferAnnounce(config.cameraStreamHandle,
				       pBuffer,
				       frameDataSize,
				       NULL,
				       &config.streamBufferHandle[iFrame]);

        if (FGSTATUS_OK != ret) {
            printf("Failed to allocate buffer.\n");
            return false;
        }
    }
    KYFG_BufferQueueAll(config.cameraStreamHandle,
			KY_ACQ_QUEUE_UNQUEUED,
			KY_ACQ_QUEUE_INPUT);  
    KYFG_CameraStart(config.camHandleArray[config.cameraIndex], 
		     config.cameraStreamHandle, 0);

    return true;
}

bool stop(kaya_config &config) {
    KYFG_CameraStop(config.camHandleArray[config.cameraIndex]);  // close camera
    if (FGSTATUS_OK != KYFG_Close(config.handle)) {
        printf("wasn't able to close grabber #%d\n", 1);
        return false;
    }  // Close the selected device and unregisters all associated routines

    return true;
}
