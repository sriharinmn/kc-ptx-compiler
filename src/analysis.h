#pragma once

// Facts about a function's control flow, used by the optimizer and the verifier.
//
// Dominators: block A "dominates" block B if every path from the entry to B
// goes through A. So a value computed in A is always ready when B runs.
// Computed with the iterative algorithm from Cooper, Harvey and Kennedy,
// "A Simple, Fast Dominance Algorithm" (2001).
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

struct Loop {
    Block* header;
    std::set<Block*> blocks;  // includes the header
};

std::vector<Loop> findLoops(const IrFunction& f, const DomTree& dt);
