#include "KYFGLib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memcpy

void Stream_callback_func(void* userContext, STREAM_HANDLE streamHandle)
{
    static void* data = 0;
    static KYBOOL copyingDataFlag = KYFALSE;
    long long totalFrames = 0, buffSize = 0;
    int buffIndex;
    void* buffData;

    if(0 == streamHandle)		// callback with indicator for acquisition stop
    {
        copyingDataFlag = KYFALSE;
        return;
    }

    totalFrames = KYFG_GetGrabberValueInt(streamHandle, "RXFrameCounter");
    buffSize = KYFG_StreamGetSize(streamHandle);			// get buffer size
    buffIndex = KYFG_StreamGetFrameIndex(streamHandle);
    buffData = KYFG_StreamGetPtr(streamHandle, buffIndex);		// get pointer of buffer data

    if(KYFALSE == copyingDataFlag)
    {
        copyingDataFlag = KYTRUE;
        data = (void*)realloc(data, (size_t)buffSize); 		// allocate size for local buffer
        if(0 == data)
        {
            return;
        }
        printf("\rGood callback buffer handle:%X, current index:%" PRISTREAM_HANDLE ", total frames:%lld        ", streamHandle, buffIndex, totalFrames);
        memcpy(data, buffData, (size_t)buffSize);			// copy data to local buffer
        //... Show Image with data ...
	//for (auto i=0; i < 640; ++i) {
	//    for (auto j = 0; j < 480; ++j) {
	// 	printf("%d|",((char*)data)[i*480+j]);
	//    }
	//    printf("\n");
	//}
        copyingDataFlag = KYFALSE;
    }
}



int main(int argc, char *argv[])
{
    FGHANDLE handle;
    int grabberIndex = 0, i;
    STREAM_HANDLE streamHandle = 0;
    CAMHANDLE camHandleArray[KY_MAX_CAMERAS];        // there are maximum KY_MAX_CAMERAS cameras
    int detectedCameras;
    char c = 0;
    int bLoopInProgress = 0;
    KYFGLib_InitParameters kyInit;
    int infosize = 0;

    kyInit.version = 2;
    kyInit.concurrency_mode = 0;
    kyInit.logging_mode = 0;
    kyInit.noVideoStreamProcess = KYFALSE;
    if (FGSTATUS_OK != KYFGLib_Initialize(&kyInit)) {
	printf("Library initialization failed \n ");
	return 1;
    }


    KY_DeviceScan(&infosize);	// Retrieve the number of virtual and hardware devices connected to PC
    if ((handle = KYFG_Open(1)) != -1){
	printf("FG 1 was connected successfully\n");
    }
    else {
	printf("Couldn't connect to FG\n");
	return 1;
    }
    int detectionCount =4;
    KYFG_UpdateCameraList(handle, camHandleArray, &detectionCount);
    printf("detection count: %d\n",detectionCount);
    if(FGSTATUS_OK == KYFG_CameraOpen2(camHandleArray[0], 0)){
	printf("Camera 0 was connected successfully\n");
    }
    else {
	printf("Camera isn't connected\n");
	return 1;
    }
    KYFG_CameraCallbackRegister(camHandleArray[0], Stream_callback_func, 0);
    KYFG_SetCameraValueInt(camHandleArray[0], "Width", 640);
    KYFG_SetCameraValueInt(camHandleArray[0], "Height", 480);
    KYFG_SetCameraValueEnum_ByValueName(camHandleArray[0], "PixelFormat", "Mono8");
    if(FGSTATUS_OK != KYFG_StreamCreateAndAlloc(camHandleArray[0], &streamHandle , 16, 0)){
	printf("Failed to allocate buffer.\n");
	return 1;
    }
    KYFG_CameraStart(camHandleArray[0], streamHandle, 0);
    

    KYFG_CameraStop(camHandleArray[0]); //close camera
    if (FGSTATUS_OK != KYFG_Close(handle)){
	printf("wasn't able to close grabber #%d\n", 1);
	return 1; 
    }// Close the selected device and unregisters all associated routines
    for (int i = 0; i < 100; ++i){
	
    }
    printf("Success!\n");
    return 0;
}
