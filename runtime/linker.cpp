#include "linker.h"

#include <nvJitLink.h>

#include <stdexcept>

namespace {

std::string errorLog(nvJitLinkHandle h) {
    size_t size = 0;
    nvJitLinkGetErrorLogSize(h, &size);
    std::string log(size, '\0');
    if (size) nvJitLinkGetErrorLog(h, log.data());
    return log;
}

}  // namespace

std::vector<char> linkPtx(const std::vector<PtxModule>& modules, const std::string& arch) {
    nvJitLinkHandle h = nullptr;
    std::string archOption = "-arch=" + arch;
    const char* options[] = {archOption.c_str()};
    if (nvJitLinkCreate(&h, 1, options) != NVJITLINK_SUCCESS) throw std::runtime_error("nvJitLinkCreate failed");

    auto fail = [&](const std::string& what) {
        std::string log = errorLog(h);
        nvJitLinkDestroy(&h);
        throw std::runtime_error("link failed (" + what + "):\n" + log);
    };
    for (const PtxModule& m : modules)
        if (nvJitLinkAddData(h, NVJITLINK_INPUT_PTX, m.ptx.data(), m.ptx.size(), m.name.c_str()) != NVJITLINK_SUCCESS)
            fail("adding " + m.name);
    if (nvJitLinkComplete(h) != NVJITLINK_SUCCESS) fail("linking");

    size_t size = 0;
    if (nvJitLinkGetLinkedCubinSize(h, &size) != NVJITLINK_SUCCESS) fail("getting the cubin size");
    std::vector<char> cubin(size);
    if (nvJitLinkGetLinkedCubin(h, cubin.data()) != NVJITLINK_SUCCESS) fail("getting the cubin");
    nvJitLinkDestroy(&h);
    return cubin;
}
