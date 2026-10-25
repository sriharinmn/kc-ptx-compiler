// kc: command-line compiler.
//   kc input.kc [-o output.ptx] [--arch sm_89] [-O0|-O1] [--emit-tokens] [--emit-ast] [--emit-ir] [--passes]

#include <fstream>
#include <iostream>
#include <sstream>

#include "compiler.h"
#include "diag.h"
#include "verify.h"

namespace {

void usage() {
    std::cerr << "usage: kc <input.kc> [-o output.ptx] [--arch sm_XX] [-O0|-O1] [--emit-tokens] [--emit-ast] [--emit-ir] [--passes]\n"
                 "  -o FILE         write PTX to FILE (default: input name with .ptx; '-' = stdout)\n"
                 "  --arch A        target GPU architecture (default: sm_89)\n"
                 "  -O0, -O1        optimization level (default: -O1)\n"
                 "  --emit-tokens   also print the tokens to stdout\n"
                 "  --emit-ast      also print the syntax tree to stdout\n"
                 "  --emit-ir       also print the SSA IR (after optimization) to stdout\n"
                 "  --passes        also print what each optimizer pass changed\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string input, output;
    CompileOptions opts;
    bool emitTokens = false, emitAst = false, emitIr = false, showPasses = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) {
            output = argv[++i];
        } else if (a == "--arch" && i + 1 < argc) {
            opts.arch = argv[++i];
        } else if (a == "-O0" || a == "-O1") {
            opts.optLevel = a[2] - '0';
        } else if (a == "--emit-tokens") {
            emitTokens = true;
        } else if (a == "--emit-ast") {
            emitAst = true;
        } else if (a == "--emit-ir") {
            emitIr = true;
        } else if (a == "--passes") {
            showPasses = true;
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
    } catch (const InternalError& e) {
        std::cerr << input << ": " << e.what() << "\n";
        return 3;
    }
    std::istringstream warnings(result.warnings);
    for (std::string w; std::getline(warnings, w);) std::cerr << input << ":" << w << "\n";

    if (emitTokens) std::cout << result.tokens << "\n";
    if (emitAst) std::cout << result.ast;
    if (emitIr) std::cout << (result.optIr.empty() ? result.ir : result.optIr);
    if (showPasses) std::cout << (result.passes.empty() ? "optimizer off (-O0)\n" : result.passes) << "\n";

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
