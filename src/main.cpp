#include <iostream>
#include <string>
#include <string>
#include <cassert>
#include <bits/stdint-uintn.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>
#include <cstdlib>
#include <unistd.h>

// halide pipeline
#include "process.h"
#include "mem_helper.h"

using std::cout; using std::endl; using std::vector;
using cv::Mat; using cv::VideoWriter; using cv::imshow; using cv::namedWindow;


void verifyArgs(int argc) {
    if (argc != 2) {
	    cout << "ERROR: usage ./app <dir> " << endl;
	    exit(0);
    }
}

vector<cv::String> * getTifFiles(std::string directory) {
    vector<cv::String> * tifFiles = new vector<cv::String>();
    vector<cv::String> singleFImages; // *.tif
    std::string singleFPath =  directory[directory.size()-1] == '/' ? directory + "*.tif": directory + "/*.tif";
    cv::glob(singleFPath, singleFImages);
    vector<cv::String> doubleFImages; // *.tiff
    std::string doubleFPath = directory[directory.size()-1] == '/' ? directory + "*.tiff": directory + "/*.tiff";
    cv::glob(doubleFPath, doubleFImages);
    tifFiles->insert(tifFiles->end(), singleFImages.begin(), singleFImages.end());
    tifFiles->insert(tifFiles->end(), doubleFImages.begin(), doubleFImages.end());
    return tifFiles;
}

vector<Mat> * getImages(std::string directory) {
    vector<Mat> * images = new vector<Mat>();
    vector<cv::String> * filenames = getTifFiles(directory);
    for (auto &file : *filenames) {
        std::cout << "ADDING FILE: " << file << "\n";
        images->push_back(cv::imread(file, cv::IMREAD_GRAYSCALE));
    }
    delete filenames;
    return images;
}

void verifyImages(vector<Mat> * images) {
    if (images->size() == 0) {
        cout << "ERROR: NO TIF FILES FOUND AT GIVEN LOCATION" << endl;
        exit(0);
    }
}

int main(int argc, char *argv[]) {
    verifyArgs(argc);
    vector<Mat> images = *getImages(std::string(argv[1]));
    verifyImages(&images);
    Mat image_out = Mat::zeros(images[0].rows, images[0].cols, images[0].type());
    VideoWriter out;
    out.open("appsrc ! videoconvert ! x264enc bitrate=5000 ! rtph264pay ! udpsink host=10.4.20.16 port=5000 ",
             0,
             30,
             image_out.size(),
             false);
    auto gpu = [](Mat img) {return mem::to_halide<uint8_t>(mem::gpu<uint8_t>(img));};
    int i = 0;
    auto output = gpu(image_out);
    Mat tmp_image = Mat(images[0].rows, images[0].cols, images[0].type());
    auto h_image_in = gpu(tmp_image);
    Mat offset = Mat::zeros(images[0].rows, images[0].cols, images[0].type());
    auto h_offset = gpu(offset);

    Mat noise = Mat(images[0].rows, images[0].cols, images[0].type());
    namedWindow("gimbaless");
    while (true) {
        if (noise.rows != images[i].rows || noise.cols == images[i].cols || noise.type() != images[i].type()) {
            noise = Mat(images[i].rows, images[i].cols, images[i].type());
        }
        cv::randu(noise, -30, 30);
        tmp_image = images[i] + noise;
        imshow("gimbaless", tmp_image);
        cv::waitKey(1000);
        ++i;
        i %= images.size();
    }
    std::cout << "Done!" << "\n";
    return 0;
}
