#include "lexer.h"

#include <cctype>
#include <unordered_map>

namespace {

const std::unordered_map<std::string, Tok> kKeywords = {
    {"kernel", Tok::KwKernel}, {"let", Tok::KwLet},       {"if", Tok::KwIf},
    {"else", Tok::KwElse},     {"while", Tok::KwWhile},   {"for", Tok::KwFor},
    {"return", Tok::KwReturn}, {"true", Tok::KwTrue},     {"false", Tok::KwFalse},
    {"i32", Tok::KwI32},       {"f32", Tok::KwF32},       {"bool", Tok::KwBool},
};

bool isDigit(char c) { return c >= '0' && c <= '9'; }
bool isIdentStart(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
bool isIdentChar(char c) { return isIdentStart(c) || isDigit(c); }

class Lexer {
public:
    explicit Lexer(const std::string& src) : src_(src) {}

    std::vector<Token> run() {
        std::vector<Token> out;
        while (true) {
            skipSpaceAndComments();
            Loc loc{line_, col_};
            if (atEnd()) {
                out.push_back({Tok::End, "", loc});
                return out;
            }
            char c = peek();
            if (isIdentStart(c))
                out.push_back(identOrKeyword(loc));
            else if (isDigit(c))
                out.push_back(number(loc));
            else
                out.push_back(punct(loc));
        }
    }

private:
    const std::string& src_;
    size_t pos_ = 0;
    int line_ = 1;
    int col_ = 1;

    bool atEnd() const { return pos_ >= src_.size(); }
    char peek(size_t ahead = 0) const { return pos_ + ahead < src_.size() ? src_[pos_ + ahead] : '\0'; }

    char advance() {
        char c = src_[pos_++];
        if (c == '\n') {
            line_++;
            col_ = 1;
        } else {
            col_++;
        }
        return c;
    }

    void skipSpaceAndComments() {
        while (!atEnd()) {
            char c = peek();
            if (std::isspace(static_cast<unsigned char>(c))) {
                advance();
            } else if (c == '/' && peek(1) == '/') {
                while (!atEnd() && peek() != '\n') advance();
            } else if (c == '/' && peek(1) == '*') {
                Loc start{line_, col_};
                advance();
                advance();
                while (!(peek() == '*' && peek(1) == '/')) {
                    if (atEnd()) throw CompileError(start, "comment is never closed (missing */)");
                    advance();
                }
                advance();
                advance();
            } else {
                break;
            }
        }
    }

    Token identOrKeyword(Loc loc) {
        size_t start = pos_;
        while (isIdentChar(peek())) advance();
        std::string text = src_.substr(start, pos_ - start);
        auto it = kKeywords.find(text);
        return {it != kKeywords.end() ? it->second : Tok::Ident, text, loc};
    }

    // 42 is an integer; 4.2 and 4e2 are floats.
    Token number(Loc loc) {
        size_t start = pos_;
        bool isFloat = false;
        while (isDigit(peek())) advance();
        if (peek() == '.' && isDigit(peek(1))) {
            isFloat = true;
            advance();
            while (isDigit(peek())) advance();
        }
        if (peek() == 'e' || peek() == 'E') {
            isFloat = true;
            advance();
            if (peek() == '+' || peek() == '-') advance();
            if (!isDigit(peek())) throw CompileError(loc, "malformed number (missing exponent digits)");
            while (isDigit(peek())) advance();
        }
        if (isIdentChar(peek())) throw CompileError(loc, "malformed number");
        return {isFloat ? Tok::FloatLit : Tok::IntLit, src_.substr(start, pos_ - start), loc};
    }

    Token punct(Loc loc) {
        // Two-character operators are checked first so "<=" is not read as "<" then "=".
        static const struct {
            const char* text;
            Tok kind;
        } kTwoChar[] = {
            {"&&", Tok::AmpAmp},     {"||", Tok::PipePipe},    {"<=", Tok::Le},
            {">=", Tok::Ge},         {"==", Tok::EqEq},        {"!=", Tok::NotEq},
            {"<<", Tok::Shl},        {">>", Tok::Shr},         {"+=", Tok::PlusAssign},
            {"-=", Tok::MinusAssign}, {"*=", Tok::StarAssign}, {"/=", Tok::SlashAssign},
        };
        for (const auto& t : kTwoChar) {
            if (peek() == t.text[0] && peek(1) == t.text[1]) {
                advance();
                advance();
                return {t.kind, t.text, loc};
            }
        }

        char c = advance();
        Tok kind;
        switch (c) {
            case '(': kind = Tok::LParen; break;
            case ')': kind = Tok::RParen; break;
            case '{': kind = Tok::LBrace; break;
            case '}': kind = Tok::RBrace; break;
            case '[': kind = Tok::LBracket; break;
            case ']': kind = Tok::RBracket; break;
            case ',': kind = Tok::Comma; break;
            case ';': kind = Tok::Semi; break;
            case ':': kind = Tok::Colon; break;
            case '.': kind = Tok::Dot; break;
            case '+': kind = Tok::Plus; break;
            case '-': kind = Tok::Minus; break;
            case '*': kind = Tok::Star; break;
            case '/': kind = Tok::Slash; break;
            case '%': kind = Tok::Percent; break;
            case '&': kind = Tok::Amp; break;
            case '|': kind = Tok::Pipe; break;
            case '^': kind = Tok::Caret; break;
            case '!': kind = Tok::Bang; break;
            case '<': kind = Tok::Lt; break;
            case '>': kind = Tok::Gt; break;
            case '=': kind = Tok::Assign; break;
            default: throw CompileError(loc, std::string("unexpected character '") + c + "'");
        }
        return {kind, std::string(1, c), loc};
    }
};

}  // namespace

std::vector<Token> lex(const std::string& source) { return Lexer(source).run(); }

const char* tokName(Tok t) {
    switch (t) {
        case Tok::End: return "end of file";
        case Tok::Ident: return "a name";
        case Tok::IntLit: return "an integer";
        case Tok::FloatLit: return "a float";
        case Tok::KwKernel: return "'kernel'";
        case Tok::KwLet: return "'let'";
        case Tok::KwIf: return "'if'";
        case Tok::KwElse: return "'else'";
        case Tok::KwWhile: return "'while'";
        case Tok::KwFor: return "'for'";
        case Tok::KwReturn: return "'return'";
        case Tok::KwTrue: return "'true'";
        case Tok::KwFalse: return "'false'";
        case Tok::KwI32: return "'i32'";
        case Tok::KwF32: return "'f32'";
        case Tok::KwBool: return "'bool'";
        case Tok::LParen: return "'('";
        case Tok::RParen: return "')'";
        case Tok::LBrace: return "'{'";
        case Tok::RBrace: return "'}'";
        case Tok::LBracket: return "'['";
        case Tok::RBracket: return "']'";
        case Tok::Comma: return "','";
        case Tok::Semi: return "';'";
        case Tok::Colon: return "':'";
        case Tok::Dot: return "'.'";
        case Tok::Plus: return "'+'";
        case Tok::Minus: return "'-'";
        case Tok::Star: return "'*'";
        case Tok::Slash: return "'/'";
        case Tok::Percent: return "'%'";
        case Tok::Amp: return "'&'";
        case Tok::Pipe: return "'|'";
        case Tok::Caret: return "'^'";
        case Tok::Shl: return "'<<'";
        case Tok::Shr: return "'>>'";
        case Tok::Bang: return "'!'";
        case Tok::AmpAmp: return "'&&'";
        case Tok::PipePipe: return "'||'";
        case Tok::Lt: return "'<'";
        case Tok::Le: return "'<='";
        case Tok::Gt: return "'>'";
        case Tok::Ge: return "'>='";
        case Tok::EqEq: return "'=='";
        case Tok::NotEq: return "'!='";
        case Tok::Assign: return "'='";
        case Tok::PlusAssign: return "'+='";
        case Tok::MinusAssign: return "'-='";
        case Tok::StarAssign: return "'*='";
        case Tok::SlashAssign: return "'/='";
    }
    return "?";
}
