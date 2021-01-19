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

    Output<Buffer<int>> output{"output", 2};

    void generate() {
        Var x("x"), y("y"),dx("dx"), dy("dy");
	Func diff("diff"),shift("shift");
	Func img1 = BoundaryConditions::repeat_edge(image1);
	Func img2 = BoundaryConditions::repeat_edge(image2);

	
	RDom patch(0, PATCH_SIZE, 0, PATCH_SIZE), search(-RADIUS, 2*RADIUS+1, -RADIUS, 2*RADIUS+1);
	diff(x, y, dx, dy) = sum(abs(i32(img1(x+patch.x , y+patch.y)) -
	 			     i32(img2(x+dx+patch.x, y+dy+patch.y))));

	shift(x, y) = argmin(search, diff(PATCH_SIZE*x, PATCH_SIZE*y, search.x, search.y));
	//out_arg(x, y) = minimum(diff(PATCH_SIZE*x, PATCH_SIZE*y, search.x, search.y));
	output(x,y) = cast<int>(shift(x,y)[1]);
	//output(x,y) = cast<int16_t>(diff(x, y, -2, -3));
	//output(x,y) = cast<int>(diff(x,y,0,0));
	
        Target target = get_target();
	if (target.has_gpu_feature()) {
	    Var xo, yo, xi, yi,xy;
	    //output.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
	    output.fuse(x,y,xy).gpu_blocks(xy);
	    diff.compute_at(output,xy).gpu_threads(dx,dy).vectorize(dy,4); // 7ms with 64, 8ms with 32, 12ms with 16
	}
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(GpuOnly, gpu_only)
