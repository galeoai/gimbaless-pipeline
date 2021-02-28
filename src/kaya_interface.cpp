#include "kaya_interface.h"
#include "KYFGLib_defines.h"
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include <string>

void Stream_callback_func(void *userContext, STREAM_HANDLE streamHandle) {
    auto config = (kaya_config *)userContext;
    static KYBOOL copyingDataFlag = KYFALSE;
    void *buffData;
    if (0 == streamHandle)  // callback with indicator for acquisition stop
    {
        copyingDataFlag = KYFALSE;
        return;
    }
    config->totalFrames = KYFG_GetGrabberValueInt(streamHandle, "RXFrameCounter");
    auto buffSize = KYFG_StreamGetSize(streamHandle);  // get buffer size
    auto buffIndex = KYFG_StreamGetFrameIndex(streamHandle);
    buffData =
        KYFG_StreamGetPtr(streamHandle, buffIndex);  // get pointer of buffer data

    if (KYFALSE == copyingDataFlag) {
        copyingDataFlag = KYTRUE;

        if (0 == userContext) {
            return;
        }
        printf("\rGood callback buffer handle:%X, current index:%" PRISTREAM_HANDLE
               ", total frames:%lld        ",
               streamHandle, buffIndex, config->totalFrames);
	config->process(config->args,buffData);

        if (config->totalFrames % 10 == 0) {
	    config->args.IIR.copyTo(config->image);
	}

        copyingDataFlag = KYFALSE;
    }
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

    KYFG_SetCameraValueFloat(config.camHandleArray[config.cameraIndex],
                             "AcquisitionFrameRate",
                             config.fps);

    config.fps = KYFG_GetCameraValueFloat(config.camHandleArray[config.cameraIndex], "AcquisitionFrameRate");

    KYFG_SetCameraValueEnum_ByValueName(config.camHandleArray[config.cameraIndex],
                                        "PixelFormat",
                                        config.pixelFormat.c_str());

    KYFG_CameraCallbackRegister(config.camHandleArray[config.cameraIndex],
                                Stream_callback_func,
                                (void *)&config);
    return true;
}

bool start(kaya_config &config) {
    if (FGSTATUS_OK !=
        KYFG_StreamCreateAndAlloc(config.camHandleArray[config.cameraIndex],
                                  &config.streamHandle, 32, 0)) {
        printf("Failed to allocate buffer.\n");
        return false;
    }
    KYFG_CameraStart(config.camHandleArray[config.cameraIndex], config.streamHandle, 0);

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
