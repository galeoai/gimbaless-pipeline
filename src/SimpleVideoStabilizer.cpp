#include "SimpleVideoStabilizer.h"

using namespace cv;
using namespace std;

Tracker::Tracker() {
    freshStart = true;
    rigidTransform = Mat::eye(3,3,CV_32FC1); //affine 2x3 in a 3x3 matrix
}

void Tracker::processImage(Mat& img) {
    vector<Point2f> corners;
    if(trackedFeatures.size() < 200) {
	goodFeaturesToTrack(img,corners,300,0.01,10);
	std::cout << "found " << corners.size() << " features\n";
	for (int i = 0; i < corners.size(); ++i) {
	    trackedFeatures.push_back(corners[i]);
	}
    }
    
    if(!prev.empty()) {
	vector<uchar> status; vector<float> errors;
	calcOpticalFlowPyrLK(prev,img,trackedFeatures,corners,status,errors,Size(10,10));
	
	if(countNonZero(status) < status.size() * 0.8) {
	    cout << "cataclysmic error \n";
	    rigidTransform = Mat::eye(3,3,CV_32FC1);
	    trackedFeatures.clear();
	    prev.release();
	    freshStart = true;
	    return;
	} else {
	    freshStart = false;
	}
	Mat_<float> newRigidTransform = estimateRigidTransform(trackedFeatures,corners,false);
	Mat_<float> nrt33 = Mat_<float>::eye(3,3);
	newRigidTransform.copyTo(nrt33.rowRange(0,2));
	rigidTransform *= nrt33;
	
	trackedFeatures.clear();
	for (int i = 0; i < status.size(); ++i) {
	    if(status[i]) {
		trackedFeatures.push_back(corners[i]);
	    }
	}
    }
    
    img.copyTo(prev);
}

