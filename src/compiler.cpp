#include "compiler.h"

#include "irgen.h"
#include "lexer.h"
#include "parser.h"
#include "ptx_emit.h"
#include "sema.h"

CompileResult compile(const std::string& source, const CompileOptions& opts) {
    std::vector<Token> tokens = lex(source);
    Program program = parse(tokens);
    checkProgram(program);
    IrModule ir = generateIr(program);

    CompileResult result;
    result.ir = printIr(ir);
    result.ptx = emitPtx(ir, opts.arch);
    return result;
}
