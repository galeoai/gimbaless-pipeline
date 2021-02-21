#include "opencv2/opencv.hpp"
#include "halide_benchmark.h"
#include "../src/zerocopy.h" //FIXME

#include "process.h"

int main(int argc, char *argv[])
{
    //load images
    cv::Mat image1,image2,image_out;
    image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE); // 8bit
    image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    image_out = cv::Mat::zeros(image1.rows,image1.cols,image1.type());
    if ( !image1.data && !image2.data )
    {
        printf("No images data \n");
        return -1;
    }
    auto h_image1 = Zerocopy::gpu<uint8_t>(image1);
    auto h_image2 = Zerocopy::gpu<uint8_t>(image2);
    auto output	= Zerocopy::gpu<uint8_t>(image_out);
    // benchmark
    double auto_schedule_off = Halide::Tools::benchmark(5, 10, [&]() {
	process(h_image1,h_image2, output);
     	output.device_sync();
    });
    printf("Manual schedule: %gms\n", auto_schedule_off * 1e3);

    if (auto_schedule_off*1e3 > 5) {
	std::cout << "pipline is slower that 5ms" << "\n";
	return 1;
    }
    std::cout << "Success!" << "\n";
    return 0;
}
