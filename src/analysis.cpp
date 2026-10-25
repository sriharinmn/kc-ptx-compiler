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

namespace {

// The dominator algorithm works on plain numbered nodes, so the same code can
// compute dominators (on the control flow graph) and post-dominators (on the
// graph with every edge reversed).
// Returns each node's immediate dominator (the start node: itself; unreachable: -1),
// and the reachable nodes in reverse postorder.
std::vector<int> immediateDominators(const std::vector<std::vector<int>>& succs, int start, std::vector<int>& rpo) {
    const int n = static_cast<int>(succs.size());
    std::vector<int> post;
    std::vector<bool> seen(n);
    std::function<void(int)> visit = [&](int x) {
        seen[x] = true;
        for (int s : succs[x])
            if (!seen[s]) visit(s);
        post.push_back(x);
    };
    visit(start);
    rpo.assign(post.rbegin(), post.rend());

    std::vector<int> order(n, -1);  // position in rpo
    for (size_t i = 0; i < rpo.size(); ++i) order[rpo[i]] = static_cast<int>(i);
    std::vector<std::vector<int>> preds(n);
    for (int x = 0; x < n; ++x)
        for (int s : succs[x]) preds[s].push_back(x);

    std::vector<int> idom(n, -1);
    idom[start] = start;  // makes the walk below stop at the start
    // Walk up from a and b until they meet: that node dominates both.
    auto intersect = [&](int a, int b) {
        while (a != b) {
            while (order[a] > order[b]) a = idom[a];
            while (order[b] > order[a]) b = idom[b];
        }
        return a;
    };
    // Start with "nothing known" and repeat until the answer stops changing.
    // A node's idom is the meeting point of all its (already processed) predecessors.
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t i = 1; i < rpo.size(); ++i) {
            int b = rpo[i], newIdom = -1;
            for (int p : preds[b]) {
                if (idom[p] == -1) continue;  // not processed yet, or unreachable
                newIdom = newIdom == -1 ? p : intersect(p, newIdom);
            }
            if (idom[b] != newIdom) {
                idom[b] = newIdom;
                changed = true;
            }
        }
    }
    return idom;
}

std::map<const Block*, int> blockNumbers(const IrFunction& f) {
    std::map<const Block*, int> number;
    for (size_t i = 0; i < f.blocks.size(); ++i) number[f.blocks[i].get()] = static_cast<int>(i);
    return number;
}

}  // namespace

DomTree::DomTree(const IrFunction& f) {
    std::map<const Block*, int> number = blockNumbers(f);
    std::vector<std::vector<int>> succs(f.blocks.size());
    for (size_t i = 0; i < f.blocks.size(); ++i)
        for (Block* s : successors(f.blocks[i].get())) succs[i].push_back(number.at(s));

    std::vector<int> rpo;
    std::vector<int> idom = immediateDominators(succs, 0, rpo);
    for (size_t i = 0; i < rpo.size(); ++i) {
        Block* b = f.blocks[rpo[i]].get();
        rpo_.push_back(b);
        order_[b] = static_cast<int>(i);
        idom_[b] = f.blocks[idom[rpo[i]]].get();
        if (i > 0) children_[idom_[b]].push_back(b);
    }
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

PostDomTree::PostDomTree(const IrFunction& f) {
    // Reverse every edge, and add one "exit" node (number n) that every return leads to.
    const int n = static_cast<int>(f.blocks.size());
    std::map<const Block*, int> number = blockNumbers(f);
    std::vector<std::vector<int>> reversed(n + 1);
    for (int i = 0; i < n; ++i) {
        const Block* b = f.blocks[i].get();
        for (Block* s : successors(b)) reversed[number.at(s)].push_back(i);
        if (b->terminator()->op == Opc::Ret) reversed[n].push_back(i);
    }
    std::vector<int> rpo;
    std::vector<int> ipdom = immediateDominators(reversed, n, rpo);
    for (int i = 0; i < n; ++i)
        ipdom_[f.blocks[i].get()] = ipdom[i] < 0 || ipdom[i] == n ? nullptr : f.blocks[ipdom[i]].get();
}

Block* PostDomTree::ipdom(const Block* b) const { return ipdom_.at(b); }

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
