#pragma once

// Facts about a function's control flow, used by the optimizer, the verifier
// and the divergence analysis.
//
// Dominators: block A "dominates" block B if every path from the entry to B
// goes through A. So a value computed in A is always ready when B runs.
// Computed with the iterative algorithm from Cooper, Harvey and Kennedy,
// "A Simple, Fast Dominance Algorithm" (2001).
//
// Post-dominators: block A "post-dominates" B if every path from B to the end
// of the function goes through A. The closest one is where the paths leaving
// a branch meet again. Same algorithm, run on the graph with its edges reversed.
//
// Loops: an edge B -> H where H dominates B jumps backwards (a "back edge"),
// and H is the header of a loop. The loop is H plus every block that can
// reach B without going through H.

#include <map>
#include <set>
#include <vector>

#include "ir.h"

// Blocks reachable from the entry, in reverse postorder: each block comes
// before the blocks it jumps to (back edges aside). The entry is first.
std::vector<Block*> reversePostOrder(const IrFunction& f);

class DomTree {
public:
    explicit DomTree(const IrFunction& f);

    bool reachable(const Block* b) const { return order_.count(b) > 0; }
    Block* idom(const Block* b) const;                   // closest dominator; nullptr for the entry
    bool dominates(const Block* a, const Block* b) const;
    const std::vector<Block*>& children(const Block* b) const;  // blocks whose idom is b
    const std::vector<Block*>& rpo() const { return rpo_; }

private:
    std::vector<Block*> rpo_;
    std::map<const Block*, int> order_;  // position in rpo_
    std::map<const Block*, Block*> idom_;
    std::map<const Block*, std::vector<Block*>> children_;
};

class PostDomTree {
public:
    explicit PostDomTree(const IrFunction& f);

    // The first block that every path from b passes through, or nullptr if the
    // paths only meet at the end of the function (e.g. one of them returns).
    Block* ipdom(const Block* b) const;

private:
    std::map<const Block*, Block*> ipdom_;
};

struct Loop {
    Block* header;
    std::set<Block*> blocks;  // includes the header
};

std::vector<Loop> findLoops(const IrFunction& f, const DomTree& dt);
