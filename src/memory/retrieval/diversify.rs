//! Maximal Marginal Relevance result diversification (pure, no I/O — SBIO).
//!
//! `rank_and_dedup` only removes exact-id duplicates; near-duplicate
//! *content* can still fill the top-k. MMR greedily selects results that are
//! relevant yet dissimilar to those already chosen:
//!
//! `next = argmax_d [ λ·rel(d) − (1−λ)·max_{s∈S} cos(d, s) ]`
//!
//! λ=1 reduces to pure relevance (identity); λ=0 is pure novelty.

/// Cosine similarity of two equal-length vectors. Returns 0.0 for a
/// zero-norm vector (e.g. pinned memories store a zero embedding).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Greedy MMR selection. `vectors[i]` is the embedding of candidate `i` and
/// `relevance[i]` its (already-ranked) score, candidates pre-sorted by
/// relevance descending. Returns up to `k` selected indices in pick order.
///
/// Degrades to relevance order (identity) when `vectors`/`relevance` lengths
/// disagree or are empty — callers use this as the "diversity unavailable"
/// fallback (e.g. remote store can't supply vectors).
pub fn mmr(vectors: &[Vec<f32>], relevance: &[f32], lambda: f32, k: usize) -> Vec<usize> {
    let n = vectors.len();
    if n == 0 || n != relevance.len() {
        return (0..relevance.len().min(k)).collect();
    }
    let k = k.min(n);
    let lambda = lambda.clamp(0.0, 1.0);

    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut remaining: Vec<usize> = (0..n).collect();

    while selected.len() < k {
        let mut best = remaining[0];
        let mut best_score = f32::MIN;
        for &cand in &remaining {
            let max_sim = selected
                .iter()
                .map(|&s| cosine(&vectors[cand], &vectors[s]))
                .fold(0.0_f32, f32::max);
            let score = lambda * relevance[cand] - (1.0 - lambda) * max_sim;
            if score > best_score {
                best_score = score;
                best = cand;
            }
        }
        selected.push(best);
        remaining.retain(|&x| x != best);
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_on_length_mismatch() {
        let v = vec![vec![1.0, 0.0]];
        assert_eq!(mmr(&v, &[0.9, 0.8], 0.5, 5), vec![0, 1]);
    }

    #[test]
    fn lambda_one_is_pure_relevance_order() {
        // Three candidates pre-sorted by relevance; λ=1 keeps that order.
        let vectors = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let rel = [0.9, 0.8, 0.7];
        assert_eq!(mmr(&vectors, &rel, 1.0, 3), vec![0, 1, 2]);
    }

    #[test]
    fn diversity_demotes_near_duplicate() {
        // 0 and 1 are identical; 2 is orthogonal but slightly less relevant.
        // With strong diversity weight, the novel doc 2 should be picked
        // before the redundant duplicate 1.
        let vectors = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let rel = [0.9, 0.85, 0.6];
        let picked = mmr(&vectors, &rel, 0.3, 3);
        assert_eq!(picked[0], 0, "most relevant picked first");
        assert_eq!(picked[1], 2, "novel doc beats redundant duplicate");
        assert_eq!(picked[2], 1);
    }

    #[test]
    fn k_truncates() {
        let vectors = vec![vec![1.0], vec![0.5], vec![0.2]];
        let rel = [0.9, 0.8, 0.7];
        assert_eq!(mmr(&vectors, &rel, 0.7, 2).len(), 2);
    }
}
