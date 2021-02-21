#include "Halide.h"
#define PATCH_SIZE 32
#define RADIUS 5

using namespace Halide;
using namespace Halide::ConciseCasts; // for i32
namespace {

class Gimbaless : public Halide::Generator<Gimbaless> {
public:
    Input<Buffer<uint8_t>> image1{"image1", 2};
    Input<Buffer<uint8_t>> image2{"image2", 2};

    Output<Buffer<uint8_t>> output{"output", 2};

    void generate() {
        Var x("x"), y("y"),dx("dx"), dy("dy");
	Func diff("diff"),shift("shift");
	Func img1 = BoundaryConditions::repeat_edge(image1);
	Func img2 = BoundaryConditions::repeat_edge(image2);

	
	RDom patch(0, PATCH_SIZE, 0, PATCH_SIZE), search(-RADIUS, 2*RADIUS+1, -RADIUS, 2*RADIUS+1);
	diff(x, y, dx, dy) = sum(abs(i32(img1(x+patch.x , y+patch.y)) -
	 			     i32(img2(x+dx+patch.x, y+dy+patch.y))));

	shift(x, y) = argmin(search, diff(PATCH_SIZE*x, PATCH_SIZE*y, search.x, search.y));
	
	output(x,y) = cast<uint8_t>(0.7f*img1(x,y) + 
				    0.3f*img2(x+shift(x/PATCH_SIZE,y/PATCH_SIZE)[0],
					      y+shift(x/PATCH_SIZE,y/PATCH_SIZE)[1]));

	// STMT output
	//Func output_stmt;
	//output_stmt(x,y) = output(x,y);
	//output_stmt.compile_to_lowered_stmt("lowered_stmt.html", output_stmt.infer_arguments(), Halide::HTML);
	
		
        Target target = get_target();
	if (target.has_gpu_feature()) {
	    Var xo, yo, xi, yi,xy;
	    shift.compute_root().fuse(x,y,xy).gpu_blocks(xy);
	    diff.compute_at(shift,xy).gpu_threads(dx,dy).vectorize(dy,8); 
	    output.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
	}
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Gimbaless, process)