#include "analysis.h"

#include <algorithm>
#include <functional>

std::vector<Block*> reversePostOrder(const IrFunction& f) {
    std::vector<Block*> post;
    std::set<const Block*> seen;
    std::function<void(Block*)> visit = [&](Block* b) {
        seen.insert(b);
        for (Block* s : successors(b))
            if (!seen.count(s)) visit(s);
        post.push_back(b);  // after everything reachable from b
    };
    visit(f.blocks[0].get());
    std::reverse(post.begin(), post.end());
    return post;
}

DomTree::DomTree(const IrFunction& f) : rpo_(reversePostOrder(f)) {
    for (size_t i = 0; i < rpo_.size(); ++i) order_[rpo_[i]] = static_cast<int>(i);
    Block* entry = rpo_[0];
    idom_[entry] = entry;  // makes the walk below stop at the entry

    // Walk up from a and b until they meet: that block dominates both.
    auto intersect = [&](Block* a, Block* b) {
        while (a != b) {
            while (order_.at(a) > order_.at(b)) a = idom_.at(a);
            while (order_.at(b) > order_.at(a)) b = idom_.at(b);
        }
        return a;
    };

    // Start with "nothing known" and repeat until the answer stops changing.
    // A block's idom is the meeting point of all its (already processed) predecessors.
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 1; i < rpo_.size(); ++i) {
            Block* b = rpo_[i];
            Block* newIdom = nullptr;
            for (Block* p : b->preds) {
                if (!idom_.count(p)) continue;  // not processed yet, or unreachable
                newIdom = newIdom ? intersect(p, newIdom) : p;
            }
            if (idom_[b] != newIdom) {
                idom_[b] = newIdom;
                changed = true;
            }
        }
    }

    for (Block* b : rpo_)
        if (b != entry) children_[idom_.at(b)].push_back(b);
}

Block* DomTree::idom(const Block* b) const {
    Block* d = idom_.at(b);
    return d == b ? nullptr : d;
}

bool DomTree::dominates(const Block* a, const Block* b) const {
    for (const Block* x = b; x; x = idom(x))
        if (x == a) return true;
    return false;
}

const std::vector<Block*>& DomTree::children(const Block* b) const {
    static const std::vector<Block*> none;
    auto it = children_.find(b);
    return it == children_.end() ? none : it->second;
}

// Headers are visited in reverse postorder, so outer loops come before the
// loops inside them, and the result is the same every run.
std::vector<Loop> findLoops(const IrFunction&, const DomTree& dt) {
    std::vector<Loop> loops;
    for (Block* h : dt.rpo()) {
        Loop loop{h, {h}};
        bool isHeader = false;
        for (Block* b : h->preds) {
            if (!dt.reachable(b) || !dt.dominates(h, b)) continue;  // not a back edge
            isHeader = true;
            // Walk backwards from the end of the loop body until we reach the header.
            std::vector<Block*> work = {b};
            while (!work.empty()) {
                Block* x = work.back();
                work.pop_back();
                if (!loop.blocks.insert(x).second) continue;  // already in the loop
                for (Block* p : x->preds) work.push_back(p);
            }
        }
        if (isHeader) loops.push_back(loop);  // several back edges to one header = one loop
    }
    return loops;
}
