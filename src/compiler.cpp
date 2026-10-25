#include "compiler.h"

#include "divergence.h"
#include "irgen.h"
#include "lexer.h"
#include "opt/opt.h"
#include "parser.h"
#include "ptx_emit.h"
#include "sema.h"
#include "verify.h"

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
    r.ir = printIr(ir);  // before verifying, so a broken IR can still be looked at
    verifyModule(ir, "irgen");
    // Checked before optimizing, so the warnings are the same at -O0 and -O1.
    for (const std::string& w : divergenceWarnings(ir)) r.warnings += w + "\n";
    if (opts.optLevel > 0) {
        r.passes = optimize(ir, opts.optLevel);
        r.optIr = printIr(ir);
    }
    r.ptx = emitPtx(ir, opts.arch, opts.lineInfo ? opts.fileName : "");
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
    } catch (const InternalError& e) {
        result.error = e.what();  // a compiler bug: no source position to point at
    }
    return result;
}
