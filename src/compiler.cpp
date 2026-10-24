#include "compiler.h"

#include "irgen.h"
#include "lexer.h"
#include "parser.h"
#include "ptx_emit.h"
#include "sema.h"

namespace {

// Runs the stages in order, writing each stage's output into r as soon as it is ready.
void runPipeline(const std::string& source, const CompileOptions& opts, CompileResult& r) {
    std::vector<Token> tokens = lex(source);
    r.tokens = printTokens(tokens);

    Program program = parse(tokens);
    r.ast = printAst(program);  // without types, in case sema fails

    checkProgram(program);
    r.ast = printAst(program);  // now with types

    IrModule ir = generateIr(program);
    r.ir = printIr(ir);
    r.ptx = emitPtx(ir, opts.arch);
}

}  // namespace

CompileResult compile(const std::string& source, const CompileOptions& opts) {
    CompileResult r;
    runPipeline(source, opts, r);
    return r;
}

ExploreResult explore(const std::string& source, const CompileOptions& opts) {
    ExploreResult result;
    try {
        runPipeline(source, opts, result.stages);
    } catch (const CompileError& e) {
        result.error = e.what();
        result.errorLine = e.loc.line;
        result.errorCol = e.loc.col;
    }
    return result;
}
