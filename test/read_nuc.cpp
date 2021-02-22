#include "../src/nuc.h"
#include <iostream>

int main(int argc, char *argv[]) {
    struct nuc::nuc_t nuc;
    
    nuc::load(argv[1],nuc);
    std::cout << nuc.gain << "\n";

    return 0;
}
