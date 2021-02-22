#ifndef NUC_H
#define NUC_H
#include <opencv2/core.hpp>

namespace nuc {
struct nuc_t {
    cv::Mat gain;
    cv::Mat offset;
};

void load(std::string filename, struct nuc_t nuc);
}  // namespace nuc
#endif
