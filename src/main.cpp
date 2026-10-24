// kc: command-line compiler.  kc input.kc [-o output.ptx] [--arch sm_89] [--emit-ir]

#include <fstream>
#include <iostream>
#include <sstream>

#include "compiler.h"
#include "diag.h"

namespace {

void usage() {
    std::cerr << "usage: kc <input.kc> [-o output.ptx] [--arch sm_XX] [--emit-ir]\n"
                 "  -o FILE     write PTX to FILE (default: input name with .ptx; '-' = stdout)\n"
                 "  --arch A    target GPU architecture (default: sm_89)\n"
                 "  --emit-ir   also print the SSA IR to stdout\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string input, output;
    CompileOptions opts;
    bool emitIr = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) {
            output = argv[++i];
        } else if (a == "--arch" && i + 1 < argc) {
            opts.arch = argv[++i];
        } else if (a == "--emit-ir") {
            emitIr = true;
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (input.empty() && a[0] != '-') {
            input = a;
        } else {
            usage();
            return 2;
        }
    }
    if (input.empty()) {
        usage();
        return 2;
    }

    std::ifstream in(input);
    if (!in) {
        std::cerr << "kc: cannot open " << input << "\n";
        return 1;
    }
    std::stringstream source;
    source << in.rdbuf();

    CompileResult result;
    try {
        result = compile(source.str(), opts);
    } catch (const CompileError& e) {
        std::cerr << input << ":" << e.what() << "\n";
        return 1;
    }

    if (emitIr) std::cout << result.ir;

    if (output.empty()) {
        size_t dot = input.rfind('.');
        output = (dot == std::string::npos ? input : input.substr(0, dot)) + ".ptx";
    }
    if (output == "-") {
        std::cout << result.ptx;
    } else {
        std::ofstream out(output);
        if (!out) {
            std::cerr << "kc: cannot write " << output << "\n";
            return 1;
        }
        out << result.ptx;
    }
    return 0;
}
