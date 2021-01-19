#include "Halide.h"
#define PATCH_SIZE 32
#define RADIUS 5

using namespace Halide;
using namespace Halide::ConciseCasts; // for i32
namespace {

class GpuOnly : public Halide::Generator<GpuOnly> {
public:
    Input<Buffer<int16_t>> image1{"image1", 2};
    Input<Buffer<int16_t>> image2{"image2", 2};

    Output<Buffer<int16_t>> output{"output", 2};

    void generate() {
        Var x("x"), y("y"),dx("dx"), dy("dy"),dyo,dyi;
	Func diff("diff"),diff_x("diff_x"), out_arg("out_arg");
	Func img1 = BoundaryConditions::repeat_edge(image1);
	Func img2 = BoundaryConditions::repeat_edge(image2);

	
	RDom patch(0, PATCH_SIZE, 0, PATCH_SIZE), search(-RADIUS, 2*RADIUS+1, -RADIUS, 2*RADIUS+1);
	diff(x, y, dx, dy) = sum(abs(i32(img1(x+patch.x , y+patch.y)) -
	 			     i32(img2(x+dx+patch.x, y+dy+patch.y))));

	out_arg(x, y) = argmin(search, diff(PATCH_SIZE*x, PATCH_SIZE*y, search.x, search.y)); 
	output(x,y) = cast<int16_t>(out_arg(x,y)[1]);
	//output(x,y) = cast<int>(diff(x, y, x, y));
	
        Target target = get_target();
	if (target.has_gpu_feature()) {
	    Var xo, yo, xi, yi,xy,sx,sy,s;
	    //output.gpu_tile(x, y, xo, yo, xi, yi, 16, 16).vectorize(xi,2);
	    output.fuse(x,y,xy).gpu_blocks(xy);
	    diff.compute_at(output,xy).gpu_threads(dx,dy).vectorize(dy,4); // 7ms with 64, 8ms with 32, 12ms with 16
	}
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(GpuOnly, gpu_only)
