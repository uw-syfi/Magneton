//! Pairing of backward ops with the forward ops that launched them.

use std::collections::HashMap;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FwdBwdInput {
    /// Non-zero marks this as a backward op, naming the forward op's thread.
    pub forward_tid: u64,
    pub sequence_number: i64,
    pub start_tid: u64,
    pub start_time: i64,
}

/// A matched pair.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FwdBwdLink {
    pub backward_idx: i64,
    pub forward_idx: i64,
    pub flow_id: u32,
}

fn key(tid: u64, seq: i64) -> u64 {
    (tid << 48) | ((seq as u64) & ((1u64 << 48) - 1))
}

pub fn link(inputs: &[FwdBwdInput], first_flow_id: u32) -> Vec<FwdBwdLink> {
    let mut pending: HashMap<u64, usize> = HashMap::new();
    let mut links = Vec::new();
    let mut flow_id = first_flow_id;

    for (i, ev) in inputs.iter().enumerate() {
        if ev.forward_tid > 0 {
            // Backward op: claim the forward op recorded for this (tid, seq).
            let k = key(ev.forward_tid, ev.sequence_number);
            if let Some(fwd) = pending.remove(&k) {
                // Removed, not just read: several backward ops can share a
                // sequence/tid pair and we must emit only one end for it.
                links.push(FwdBwdLink {
                    backward_idx: i as i64,
                    forward_idx: fwd as i64,
                    flow_id,
                });
                flow_id += 1;
            }
        } else if ev.start_tid != 0 {
            // Forward op. Among ops sharing a sequence number, the one with the
            // greatest start time is the one that launched the backward pass.
            let k = key(ev.start_tid, ev.sequence_number);
            match pending.get(&k) {
                Some(&prev) if inputs[prev].start_time > ev.start_time => {}
                _ => {
                    pending.insert(k, i);
                }
            }
        }
    }
    links
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fwd(tid: u64, seq: i64, t: i64) -> FwdBwdInput {
        FwdBwdInput { forward_tid: 0, sequence_number: seq, start_tid: tid, start_time: t }
    }
    fn bwd(fwd_tid: u64, seq: i64) -> FwdBwdInput {
        FwdBwdInput { forward_tid: fwd_tid, sequence_number: seq, start_tid: 0, start_time: 0 }
    }

    #[test]
    fn matches_backward_to_its_forward() {
        let links = link(&[fwd(7, 1, 100), bwd(7, 1)], 0);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].forward_idx, 0);
        assert_eq!(links[0].backward_idx, 1);
    }

    #[test]
    fn latest_forward_wins_for_a_sequence() {
        // Two forwards share (tid=7, seq=1); the later-starting one should win.
        let links = link(&[fwd(7, 1, 100), fwd(7, 1, 500), bwd(7, 1)], 0);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].forward_idx, 1);
    }

    #[test]
    fn a_forward_is_claimed_only_once() {
        // Two backwards, one forward: only the first backward links.
        let links = link(&[fwd(7, 1, 100), bwd(7, 1), bwd(7, 1)], 0);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].backward_idx, 1);
    }

    #[test]
    fn flow_ids_increment() {
        let links = link(&[fwd(7, 1, 1), bwd(7, 1), fwd(7, 2, 2), bwd(7, 2)], 10);
        assert_eq!(links.iter().map(|l| l.flow_id).collect::<Vec<_>>(), vec![10, 11]);
    }
}
