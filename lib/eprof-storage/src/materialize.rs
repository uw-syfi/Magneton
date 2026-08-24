//! Turning the flat event array into a tree.

use std::collections::HashMap;

/// libkineto flow type (GenericTraceActivity.h: kLinkAsyncCpuGpu = 2).
const K_LINK_ASYNC_CPU_GPU: u32 = 2;
/// EventType::Kineto. Repeated here so this module needs nothing from event.
const KINETO_TAG: i32 = 6;

/// Just enough of an event to parent it: the merge reads nothing else.
#[derive(Clone, Copy)]
pub struct MergeNode {
    pub id: i64,
    pub orig_parent_id: i64,
    pub tag: i32,
    pub flow_id: u32,
    pub flow_type: u32,
    pub flow_start: u32,
    pub linked_id: i64,
}

/// Recompute the parent id of every Kineto node from flow + linked signals.
pub fn reassign_kineto_parents(nodes: &[MergeNode]) -> Vec<i64> {
    // Pass 1: collect flow starts.
    let mut flow_map: HashMap<u32, i64> = HashMap::new();
    for n in nodes {
        if n.tag == KINETO_TAG
            && n.flow_type == K_LINK_ASYNC_CPU_GPU
            && n.flow_start == 1
        {
            flow_map.insert(n.flow_id, n.id);
        }
    }

    // Pass 2: kineto parent = flow-start (if a non-start flow member) else linked.
    nodes
        .iter()
        .map(|n| {
            if n.tag != KINETO_TAG {
                return n.orig_parent_id;
            }
            let mut parent = n.linked_id; // pass-1 default
            if n.flow_type == K_LINK_ASYNC_CPU_GPU && n.flow_start == 0 {
                if let Some(&p) = flow_map.get(&n.flow_id) {
                    parent = p; // flow takes priority
                }
            }
            parent
        })
        .collect()
}

/// Full node inputs for end-to-end materialization (flow/linked + containment).
#[derive(Clone, Copy)]
pub struct FullNode {
    pub id: i64,
    pub tag: i32,
    pub start_tid: u64,
    pub forward_tid: u64,
    pub start_ns: i64,
    pub end_ns: i64,
    pub flow_id: u32,
    pub flow_type: u32,
    pub flow_start: u32,
    pub linked_id: i64,
}

/// Result of materialization: parent id and (possibly recomputed) tid per node.
pub struct Materialized {
    pub parents: Vec<i64>,
    /// Effective tids; kineto nodes get their parent's tid (setKinetoTID).
    pub tids: Vec<u64>,
}

pub fn materialize(nodes: &[FullNode], current_tid: u64) -> Materialized {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = nodes.len();
    let mut parent = vec![-1i64; n];
    let mut finished = vec![false; n];
    let id_to_idx: HashMap<i64, usize> =
        nodes.iter().enumerate().map(|(i, nd)| (nd.id, i)).collect();

    // --- Stage 1: flow/linked merge (setParents). ---
    let mut flow_map: HashMap<u32, i64> = HashMap::new();
    for nd in nodes {
        if nd.tag == KINETO_TAG
            && nd.flow_type == K_LINK_ASYNC_CPU_GPU
            && nd.flow_start == 1
        {
            flow_map.insert(nd.flow_id, nd.id);
        }
    }
    for (i, nd) in nodes.iter().enumerate() {
        if nd.tag != KINETO_TAG {
            continue;
        }
        let mut p = nd.linked_id;
        if nd.flow_type == K_LINK_ASYNC_CPU_GPU && nd.flow_start == 0 {
            if let Some(&fp) = flow_map.get(&nd.flow_id) {
                p = fp;
            }
        }
        parent[i] = p;
        if p >= 0 {
            // setParents mark_finished: this node now belongs to a kineto subtree
            // and build_tree skips it.
            finished[i] = true;
        }
    }

    // --- setKinetoTID: kineto nodes take their parent's tid (current_tid if an
    // orphan root). Non-kineto nodes keep their collected tid. ---
    let mut tid: Vec<u64> = nodes.iter().map(|nd| nd.start_tid).collect();
    {
        let mut ch1: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut roots1: Vec<usize> = Vec::new();
        for i in 0..n {
            let p = parent[i];
            if p >= 0 {
                ch1[id_to_idx[&p]].push(i);
            } else {
                roots1.push(i);
            }
        }
        // Top-down DFS so a parent's tid is set before its children read it.
        let mut stack: Vec<(usize, Option<u64>)> =
            roots1.iter().map(|&r| (r, None)).collect();
        while let Some((i, parent_tid)) = stack.pop() {
            if nodes[i].tag == KINETO_TAG {
                tid[i] = parent_tid.unwrap_or(current_tid);
            }
            for &c in &ch1[i] {
                stack.push((c, Some(tid[i])));
            }
        }
    }

    // --- Stage 2: build_tree containment (stack replay sorted by start). ---
    let mut order: Vec<usize> = (0..n).collect();
    // stable: tie-break by input index, matching C++ stable_sort on start_ns.
    order.sort_by_key(|&i| (nodes[i].start_ns, i));

    let mut stacks: HashMap<u64, i64> = HashMap::new();
    let mut heap: BinaryHeap<Reverse<(i64, i64)>> = BinaryHeap::new();

    // pop_event: close `top_id`, unwinding any still-open frames above it on its
    // tid, then restore its parent as the tid's open frame.
    let pop_event = |top_id: i64,
                     parent: &mut Vec<i64>,
                     finished: &mut Vec<bool>,
                     stacks: &mut HashMap<u64, i64>| {
        let ti = id_to_idx[&top_id];
        if finished[ti] {
            return;
        }
        let etid = tid[ti];
        let mut frame = *stacks.get(&etid).expect("stacks.at(tid)");
        while frame != top_id {
            let fi = id_to_idx[&frame];
            finished[fi] = true;
            frame = parent[fi];
        }
        finished[ti] = true;
        stacks.remove(&etid);
        let np = parent[ti];
        if np >= 0 {
            stacks.insert(etid, np);
        }
    };

    for &i in &order {
        let start = nodes[i].start_ns;
        while let Some(&Reverse((e_end, e_id))) = heap.peek() {
            if e_end < start {
                heap.pop();
                pop_event(e_id, &mut parent, &mut finished, &mut stacks);
            } else {
                break;
            }
        }
        // push_event
        let nd = &nodes[i];
        if nd.tag == KINETO_TAG && finished[i] {
            continue; // already in a flow/linked subtree
        }
        let mut pid: i64 = -1;
        if let Some(&p) = stacks.get(&tid[i]) {
            pid = p;
        } else if nd.forward_tid != 0 {
            if let Some(&p) = stacks.get(&nd.forward_tid) {
                pid = p;
            }
        }
        parent[i] = pid;
        if nd.end_ns > nd.start_ns {
            stacks.insert(tid[i], nd.id);
            heap.push(Reverse((nd.end_ns, nd.id)));
        } else {
            finished[i] = true; // instant event (e.g. Power): not pushed
        }
    }
    while let Some(Reverse((_e_end, e_id))) = heap.pop() {
        pop_event(e_id, &mut parent, &mut finished, &mut stacks);
    }

    Materialized { parents: parent, tids: tid }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(id: i64, flow_id: u32, flow_start: u32, linked: i64) -> MergeNode {
        MergeNode {
            id,
            orig_parent_id: -999,
            tag: KINETO_TAG,
            flow_id,
            flow_type: K_LINK_ASYNC_CPU_GPU,
            flow_start,
            linked_id: linked,
        }
    }

    #[test]
    fn kernel_parents_to_flow_start_runtime() {
        // aten op id=0 (TorchOp), runtime call id=1 (flow start), kernel id=2.
        let op = MergeNode {
            id: 0, orig_parent_id: -1, tag: 0, flow_id: 0, flow_type: 0,
            flow_start: 0, linked_id: -1,
        };
        let runtime = k(1, 42, 1, 0); // flow start, linked to op 0
        let kernel = k(2, 42, 0, 0);  // shares flow 42, linked (stale) to op 0
        let parents = reassign_kineto_parents(&[op, runtime, kernel]);
        assert_eq!(parents[0], -1); // TorchOp keeps original
        assert_eq!(parents[1], 0);  // runtime -> linked op
        assert_eq!(parents[2], 1);  // kernel -> flow-start runtime (not linked op)
    }

    #[test]
    fn unmatched_flow_falls_back_to_linked() {
        let lone = k(5, 99, 0, -1); // non-start, no matching start, no link
        let parents = reassign_kineto_parents(&[lone]);
        assert_eq!(parents[0], -1); // stays root
    }
}
