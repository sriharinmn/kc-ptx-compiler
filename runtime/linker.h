#pragma once

// Linking several PTX modules into one program with nvJitLink, NVIDIA's
// linker library (the in-process version of nvlink).
//
// This is separate compilation, like compiling several .cu files with -rdc:
//   - each .kc file is compiled to its own PTX
//   - a module that calls a function from another file declares it
//     ".extern .func"; the file that defines it marks it ".visible .func"
//   - nvJitLink compiles each PTX into a relocatable object (machine code with
//     "fill in the address of f here" notes, called relocations), then joins them
//     and fills in every call: the result is one cubin that the driver can load.

#include <string>
#include <vector>

struct PtxModule {
    std::string name;  // shown in error messages, e.g. "main.kc"
    std::string ptx;
};

// Throws std::runtime_error with the linker's log (e.g. an undefined function).
std::vector<char> linkPtx(const std::vector<PtxModule>& modules, const std::string& arch);
