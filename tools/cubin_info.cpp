// cubin-info: reads a cubin (GPU machine code file) and explains what is inside.
//   cubin-info file.cubin
//
// A cubin is an ELF file, the same container format Linux uses for programs and
// object files: a header, then "sections" (named chunks of bytes), a symbol table
// (names of functions and where they are), and, in an object file that still
// needs linking, "relocations" (places where an address must be filled in).
// The standard ELF structs come from <elf.h>; the CUDA-specific meanings below
// were checked against readelf, cuobjdump and ptxas -v.

#include <elf.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr int kInstructionBytes = 16;     // every GPU instruction is 128 bits (sm_70 and newer)
constexpr int kParamOffset = 0x160;        // kernel parameters start here in constant bank 0 (c[0x0][0x160])
constexpr unsigned char kCudaEntry = 0x10;  // st_other flag of a symbol that is a kernel

struct Cubin {
    std::vector<char> bytes;

    template <typename T>
    const T& at(size_t offset) const { return *reinterpret_cast<const T*>(bytes.data() + offset); }

    const Elf64_Ehdr& header() const { return at<Elf64_Ehdr>(0); }
    const Elf64_Shdr& section(size_t i) const { return at<Elf64_Shdr>(header().e_shoff + i * sizeof(Elf64_Shdr)); }
    size_t sectionCount() const { return header().e_shnum; }
    std::string sectionName(size_t i) const {
        return bytes.data() + section(header().e_shstrndx).sh_offset + section(i).sh_name;
    }
    std::string symbolName(const Elf64_Shdr& symtab, const Elf64_Sym& sym) const {
        return bytes.data() + section(symtab.sh_link).sh_offset + sym.st_name;
    }
};

bool startsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

// A one-line explanation of each kind of section.
std::string describe(const std::string& name) {
    if (startsWith(name, ".text.")) return "machine code (SASS) of " + name.substr(6);
    if (startsWith(name, ".nv.constant0.")) return "constant bank 0 of " + name.substr(14) + ": its parameters";
    if (startsWith(name, ".nv.shared.")) return "shared memory of " + name.substr(11);
    if (startsWith(name, ".nv.info.")) return "attributes of " + name.substr(9) + " for the driver";
    if (startsWith(name, ".rela.") || startsWith(name, ".rel."))
        return "relocations: addresses to fill in inside " + name.substr(name.find('.', 1));
    static const std::map<std::string, std::string> known = {
        {".shstrtab", "section names"},
        {".strtab", "symbol names"},
        {".symtab", "symbol table: functions and where they are"},
        {".nv.info", "attributes for the whole file"},
        {".nv.callgraph", "which function calls which (for the call stack size)"},
        {".debug_line", "line table: machine code -> source lines (from .loc)"},
        {".nv_debug_line_sass", "line table for the SASS"},
        {".nv_debug_ptx_txt", "the PTX text, kept for debuggers"},
        {".debug_frame", "call frame info for debuggers"},
        {".nv.rel.action", "how the linker applies relocations"},
    };
    auto it = known.find(name);
    if (it != known.end()) return it->second;
    if (startsWith(name, ".note.nv.")) return "note: toolkit information";
    return "";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: cubin-info file.cubin\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    Cubin c{std::vector<char>(std::istreambuf_iterator<char>(in), {})};
    if (c.bytes.size() < sizeof(Elf64_Ehdr) || std::memcmp(c.bytes.data(), ELFMAG, SELFMAG) != 0 ||
        c.header().e_machine != EM_CUDA) {
        std::fprintf(stderr, "cubin-info: %s is not a CUDA ELF file\n", argv[1]);
        return 1;
    }

    const Elf64_Ehdr& h = c.header();
    bool relocatable = h.e_type == ET_REL;
    std::printf("%s: CUDA ELF for sm_%u, %s\n\n", argv[1], (h.e_flags >> 8) & 0xff,
                relocatable ? "relocatable (an object file: must be linked before loading)"
                            : "executable (ready for the driver to load)");

    std::printf("sections:\n");
    std::map<std::string, size_t> sizeOf;
    for (size_t i = 1; i < c.sectionCount(); ++i) {
        std::string name = c.sectionName(i);
        sizeOf[name] = c.section(i).sh_size;
        std::printf("  %-26s %6zu bytes  %s\n", name.c_str(), static_cast<size_t>(c.section(i).sh_size),
                    describe(name).c_str());
    }

    // Symbols: kernels, device functions, and the undefined ones a linker must provide.
    std::printf("\nfunctions:\n");
    const Elf64_Shdr* symtab = nullptr;
    for (size_t i = 1; i < c.sectionCount(); ++i)
        if (c.section(i).sh_type == SHT_SYMTAB) symtab = &c.section(i);
    std::vector<std::string> symbolNames;
    for (size_t off = 0; symtab && off < symtab->sh_size; off += sizeof(Elf64_Sym)) {
        const Elf64_Sym& sym = c.at<Elf64_Sym>(symtab->sh_offset + off);
        std::string name = c.symbolName(*symtab, sym);
        symbolNames.push_back(name);
        if (ELF64_ST_TYPE(sym.st_info) != STT_FUNC) continue;
        if (sym.st_shndx == SHN_UNDEF) {
            std::printf("  %-20s undefined: defined in another file, the linker fills in its address\n", name.c_str());
            continue;
        }
        const Elf64_Shdr& text = c.section(sym.st_shndx);
        bool kernel = sym.st_other & kCudaEntry;
        std::printf("  %-20s %s, %zu bytes = %zu instruction slots, %u registers", name.c_str(),
                    kernel ? "kernel         " : "device function", static_cast<size_t>(sym.st_size),
                    static_cast<size_t>(sym.st_size) / kInstructionBytes, text.sh_info >> 24);
        if (kernel) {
            size_t constBank = sizeOf[".nv.constant0." + name];
            std::printf(", %zu bytes of parameters, %zu bytes of shared memory",
                        constBank > kParamOffset ? constBank - kParamOffset : 0, sizeOf[".nv.shared." + name]);
        }
        std::printf("\n");
    }

    // Relocations in machine code: the places the linker still has to patch.
    bool any = false;
    for (size_t i = 1; i < c.sectionCount(); ++i) {
        const Elf64_Shdr& s = c.section(i);
        if ((s.sh_type != SHT_REL && s.sh_type != SHT_RELA) || !startsWith(c.sectionName(s.sh_info), ".text.")) continue;
        size_t entry = s.sh_type == SHT_REL ? sizeof(Elf64_Rel) : sizeof(Elf64_Rela);
        for (size_t off = 0; off < s.sh_size; off += entry) {
            const Elf64_Rel& r = c.at<Elf64_Rel>(s.sh_offset + off);  // Rela starts the same way
            if (!any) std::printf("\nrelocations (addresses the linker fills in):\n");
            any = true;
            std::printf("  %s + 0x%llx: address of %s (CUDA relocation type %llu)\n", c.sectionName(s.sh_info).c_str(),
                        static_cast<unsigned long long>(r.r_offset), symbolNames[ELF64_R_SYM(r.r_info)].c_str(),
                        static_cast<unsigned long long>(ELF64_R_TYPE(r.r_info)));
        }
    }
    return 0;
}
