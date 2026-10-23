//! PageRank algorithm for graph-structural importance.
//!
//! Pure iterative computation (SBIO) — takes an adjacency list, returns
//! per-node scores. Wired into `reflect()` to periodically update
//! `importance_state.pagerank` on each memory.
//!
//! Uses the standard PageRank formula:
//! `PR(i) = (1-d)/N + d × Σ(PR(j)/L(j))` for all j linking to i
//!
//! Where:
//! - `d` = damping factor (typically 0.85)
//! - `N` = total number of nodes
//! - `L(j)` = outgoing edge count of node j

use std::collections::HashMap;

/// Compute PageRank scores for all nodes in the graph.
///
/// `adjacency`: slice of `(source_id, vec_of_target_ids)` pairs.
/// Returns a map of `node_id → score` normalized to [0.0, 1.0].
/// Disconnected nodes receive the base score `(1-d)/N`.
pub fn compute_pagerank(
    adjacency: &[(u64, Vec<u64>)],
    damping: f32,
    iterations: usize,
) -> HashMap<u64, f32> {
    if adjacency.is_empty() {
        return HashMap::new();
    }

    // Collect all unique node IDs (sources + targets)
    let mut all_ids: Vec<u64> = Vec::new();
    for (src, targets) in adjacency {
        all_ids.push(*src);
        all_ids.extend(targets);
    }
    all_ids.sort_unstable();
    all_ids.dedup();

    let n = all_ids.len() as f32;
    let base = (1.0 - damping) / n;

    // Build outgoing edges map
    let outgoing: HashMap<u64, &Vec<u64>> = adjacency.iter().map(|(s, t)| (*s, t)).collect();

    // Initialize uniform scores
    let mut scores: HashMap<u64, f32> = all_ids.iter().map(|&id| (id, 1.0 / n)).collect();

    // Iterative convergence
    for _ in 0..iterations {
        let mut new_scores: HashMap<u64, f32> = all_ids.iter().map(|&id| (id, base)).collect();

        for (&src, targets) in &outgoing {
            if targets.is_empty() {
                continue;
            }
            let share = scores[&src] / targets.len() as f32;
            for &target in targets.iter() {
                if let Some(score) = new_scores.get_mut(&target) {
                    *score += damping * share;
                }
            }
        }

        scores = new_scores;
    }

    // Normalize to [0.0, 1.0] by dividing by max score
    let max_score = scores.values().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_score > 0.0 {
        for score in scores.values_mut() {
            *score /= max_score;
        }
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph() {
        let scores = compute_pagerank(&[], 0.85, 20);
        assert!(scores.is_empty());
    }

    #[test]
    fn triangle_converges_to_equal() {
        // A→B, B→C, C→A — symmetric cycle
        let adj: Vec<(u64, Vec<u64>)> = vec![(1, vec![2]), (2, vec![3]), (3, vec![1])];
        let scores = compute_pagerank(&adj, 0.85, 20);
        assert_eq!(scores.len(), 3);

        let s1 = scores[&1];
        let s2 = scores[&2];
        let s3 = scores[&3];
        // All should be roughly equal in a symmetric cycle
        assert!((s1 - s2).abs() < 0.05, "s1={s1} s2={s2}");
        assert!((s2 - s3).abs() < 0.05, "s2={s2} s3={s3}");
    }

    #[test]
    fn star_graph_hub_highest() {
        // Hub (1) points to 2,3,4. No back-edges.
        let adj: Vec<(u64, Vec<u64>)> =
            vec![(1, vec![2, 3, 4]), (2, vec![]), (3, vec![]), (4, vec![])];
        let scores = compute_pagerank(&adj, 0.85, 20);

        // Nodes 2,3,4 receive rank from hub; hub only gets base rank.
        // With normalization, 2/3/4 should score higher than 1.
        let hub = scores[&1];
        let leaf = scores[&2];
        assert!(
            leaf > hub,
            "leaf should score higher than hub (receives rank): leaf={leaf} hub={hub}"
        );
    }

    #[test]
    fn disconnected_nodes_get_base_rank() {
        // Node 1→2, node 3 disconnected
        let adj: Vec<(u64, Vec<u64>)> = vec![(1, vec![2]), (2, vec![]), (3, vec![])];
        let scores = compute_pagerank(&adj, 0.85, 20);
        assert_eq!(scores.len(), 3);

        // Node 3 should have the lowest score (only gets base (1-d)/N)
        let s3 = scores[&3];
        let s2 = scores[&2]; // receives rank from 1
        assert!(
            s2 > s3,
            "connected node should rank higher: s2={s2} s3={s3}"
        );
    }

    #[test]
    fn scores_normalized_to_unit() {
        let adj: Vec<(u64, Vec<u64>)> = vec![(1, vec![2]), (2, vec![3]), (3, vec![1])];
        let scores = compute_pagerank(&adj, 0.85, 20);
        let max = scores.values().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max - 1.0).abs() < 0.01,
            "max score should be ~1.0, got {max}"
        );
    }

    #[test]
    fn single_node_no_edges() {
        let adj: Vec<(u64, Vec<u64>)> = vec![(42, vec![])];
        let scores = compute_pagerank(&adj, 0.85, 20);
        assert_eq!(scores.len(), 1);
        assert!((scores[&42] - 1.0).abs() < 0.01);
    }

    #[test]
    fn two_node_mutual() {
        // A↔B: both point to each other
        let adj: Vec<(u64, Vec<u64>)> = vec![(1, vec![2]), (2, vec![1])];
        let scores = compute_pagerank(&adj, 0.85, 20);
        let s1 = scores[&1];
        let s2 = scores[&2];
        assert!(
            (s1 - s2).abs() < 0.01,
            "mutual links should be equal: s1={s1} s2={s2}"
        );
    }
}
