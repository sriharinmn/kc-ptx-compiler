// kc-link: links PTX files (from separately compiled .kc files) into one cubin with nvJitLink.
//   kc-link a.ptx b.ptx ... -o program.cubin [--arch sm_89]
// Then: cuobjdump -sass program.cubin, or cubin-info program.cubin.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "linker.h"

int main(int argc, char** argv) {
    std::vector<PtxModule> modules;
    std::string output = "a.cubin", arch = "sm_89";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) {
            output = argv[++i];
        } else if (a == "--arch" && i + 1 < argc) {
            arch = argv[++i];
        } else {
            std::ifstream in(a);
            if (!in) {
                std::fprintf(stderr, "kc-link: cannot open %s\n", a.c_str());
                return 1;
            }
            std::stringstream ss;
            ss << in.rdbuf();
            modules.push_back({a, ss.str()});
        }
    }
    if (modules.empty()) {
        std::fprintf(stderr, "usage: kc-link a.ptx b.ptx ... -o program.cubin [--arch sm_89]\n");
        return 2;
    }
    try {
        std::vector<char> cubin = linkPtx(modules, arch);
        std::ofstream(output, std::ios::binary).write(cubin.data(), static_cast<std::streamsize>(cubin.size()));
        std::printf("linked %zu files into %s (%zu bytes)\n", modules.size(), output.c_str(), cubin.size());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kc-link: %s\n", e.what());
        return 1;
    }
}
