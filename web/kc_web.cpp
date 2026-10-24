// The bridge between the compiler and the web playground.
// Built with Emscripten (C++ -> WebAssembly) by web/build.sh, so the real kc
// compiler runs inside the browser: no server needed.

#include <emscripten/emscripten.h>

#include <cstdio>
#include <string>

#include "compiler.h"

namespace {

std::string jsonString(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);  // UTF-8 bytes pass through unchanged
                }
        }
    }
    return out + "\"";
}

}  // namespace

// Compiles 'source' and returns every stage as JSON:
// {"tokens": ..., "ast": ..., "ir": ..., "ptx": ..., "error": ..., "line": N, "col": N}
extern "C" EMSCRIPTEN_KEEPALIVE const char* kc_explore(const char* source, const char* arch) {
    static std::string json;  // stays alive until the next call, so JavaScript can read it
    CompileOptions opts;
    opts.arch = arch;
    ExploreResult r = explore(source, opts);
    json = "{\"tokens\":" + jsonString(r.stages.tokens) + ",\"ast\":" + jsonString(r.stages.ast) +
           ",\"ir\":" + jsonString(r.stages.ir) + ",\"ptx\":" + jsonString(r.stages.ptx) +
           ",\"error\":" + jsonString(r.error) + ",\"line\":" + std::to_string(r.errorLine) +
           ",\"col\":" + std::to_string(r.errorCol) + "}";
    return json.c_str();
}
