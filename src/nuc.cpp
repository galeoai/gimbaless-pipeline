#include "nuc.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/hdf/hdf5.hpp>
#include <ostream>

using namespace cv;

namespace nuc {

void load(std::string filename, struct nuc_t nuc) {
    Ptr<hdf::HDF5> h5io = hdf::open(filename);
    h5io->dsread(nuc.gain, "gain");
    h5io->dsread(nuc.offset, "offset");
    // TODO: test size and content

    h5io->close();
}
};  // namespace nuc
