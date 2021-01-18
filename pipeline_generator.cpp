#include "Halide.h"
using namespace Halide;
using namespace Halide::ConciseCasts; // for i32
namespace {

class GpuOnly : public Halide::Generator<GpuOnly> {
public:
    Input<Buffer<int>> image1{"image1", 2};
    Input<Buffer<int>> image2{"image2", 2};

    Output<Buffer<int>> output{"output", 2};

    void generate() {
        Var x("x"), y("y"),x1("x1"), y1("y1");
	Func diff("diff"),diff_x("diff_x"), out_arg("out_arg");
	Func img1 = BoundaryConditions::repeat_edge(image1);
	Func img2 = BoundaryConditions::repeat_edge(image2);
        // Create a simple pipeline that scales pixel values by 2.
	RDom patch(0, 8), search(-5, 2*5+1, -5, 2*5+1);
	diff_x(x, y, x1, y1) = sum(abs(i32(img1(x+patch , y)) -
				       i32(img2(x1+patch, y1))));
	diff(x, y, x1, y1) = sum(diff_x(x,y+patch,x1,y1+patch));
	out_arg(x, y) = argmin(search, diff(x, y, x+search.x, y+search.y)); 
	output(x,y) = out_arg(x*8,y*8)[0];
	//output(x,y) = cast<int>(diff(x, y, x, y));
	
	//output(x, y) = input(x, y) * 3;
	

        Target target = get_target();
	if (target.has_gpu_feature()) {
	    Var xo, yo, xi, yi;
	    output.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
	}
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(GpuOnly, gpu_only)
