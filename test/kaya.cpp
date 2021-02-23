#include "../src/kaya_interface.h"

#include <cstdlib>  //for uint8_t
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // for memcpy


int main(int argc, char *argv[]) {
    kaya_config config;
    config.width = 1024;
    config.height = 1024;
    config.grabberIndex = 1;
    config.cameraIndex = 0;
    config.image = 255*cv::Mat::ones(1024,1024,CV_8UC1);
    setup(config);
    start(config);
    while(true){
	cv::imshow( "image", config.image );
	char c=(char)cv::waitKey(25);
	if(c==27) break;
    };
    stop(config);
    /*
    FGHANDLE handle;
    int grabberIndex = 0, i;
    STREAM_HANDLE streamHandle = 0;
    CAMHANDLE camHandleArray[KY_MAX_CAMERAS];  // there are maximum KY_MAX_CAMERAS
                                               // cameras
    int detectedCameras;
    char c = 0;
    int bLoopInProgress = 0;
    KYFGLib_InitParameters kyInit;
    int infosize = 0;
    auto image = cv::Mat(720, 1920, CV_8UC1);

    kyInit.version = 2;
    kyInit.concurrency_mode = 0;
    kyInit.logging_mode = 0;
    kyInit.noVideoStreamProcess = KYFALSE;
    if (FGSTATUS_OK != KYFGLib_Initialize(&kyInit)) {
        printf("Library initialization failed \n ");
        return 1;
    }

    KY_DeviceScan(&infosize);  // Retrieve the number of virtual and hardware
                               // devices connected to PC
    if ((handle = KYFG_Open(1)) != -1) {
        printf("FG 1 was connected successfully\n");
    } else {
        printf("Couldn't connect to FG\n");
        return 1;
    }
    int detectionCount = 4;
    KYFG_UpdateCameraList(handle, camHandleArray, &detectionCount);
    printf("detection count: %d\n", detectionCount);
    if (FGSTATUS_OK == KYFG_CameraOpen2(camHandleArray[0], 0)) {
        printf("Camera 0 was connected successfully\n");
    } else {
        printf("Camera isn't connected\n");
        return 1;
    }
    KYFG_CameraCallbackRegister(camHandleArray[0], Stream_callback_func,
                                (void *)image.ptr<uint8_t>());
    KYFG_SetCameraValueInt(camHandleArray[0], "Width", 1920);
    KYFG_SetCameraValueInt(camHandleArray[0], "Height", 720);
    KYFG_SetCameraValueEnum_ByValueName(camHandleArray[0], "PixelFormat",
                                        "Mono8");
    if (FGSTATUS_OK !=
        KYFG_StreamCreateAndAlloc(camHandleArray[0], &streamHandle, 16, 0)) {
        printf("Failed to allocate buffer.\n");
        return 1;
    }
    KYFG_CameraStart(camHandleArray[0], streamHandle, 0);

    while (true) {
	// Press  ESC on keyboard to exit
	//cv::imshow( "image", image );
	//char c=(char)cv::waitKey(25);
	//if(c==27) break;
    }

    KYFG_CameraStop(camHandleArray[0]);  // close camera
    if (FGSTATUS_OK != KYFG_Close(handle)) {
        printf("wasn't able to close grabber #%d\n", 1);
        return 1;
    }  // Close the selected device and unregisters all associated routines
    */
    printf("\nSuccess!\n");
    return 0;
}
