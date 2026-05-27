use crate::genome::MusicalGenome;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Gaussian mutation: add N(0, σ) noise to each gene with probability `rate`.
pub fn gaussian_mutation(
    genome: &MusicalGenome,
    rate: f64,
    sigma: f64,
    rng: &mut impl Rng,
) -> MusicalGenome {
    let normal = Normal::new(0.0, sigma).unwrap();
    let g = genome.genes();
    let mut new_genes = *g;
    for gene in &mut new_genes {
        if rng.gen_bool(rate) {
            *gene += normal.sample(rng);
        }
    }
    MusicalGenome::new(&new_genes)
}

/// Uniform mutation: replace selected genes with random values in [0, 5] with probability `rate`.
pub fn uniform_mutation(genome: &MusicalGenome, rate: f64, rng: &mut impl Rng) -> MusicalGenome {
    let g = genome.genes();
    let mut new_genes = *g;
    for gene in &mut new_genes {
        if rng.gen_bool(rate) {
            *gene = rng.gen_range(0.0..5.0);
        }
    }
    MusicalGenome::new(&new_genes)
}

/// Inversion mutation: reverse a random contiguous subsequence of genes.
pub fn inversion_mutation(genome: &MusicalGenome, rng: &mut impl Rng) -> MusicalGenome {
    let g = genome.genes();
    let mut new_genes = *g;
    let n = new_genes.len();
    let i = rng.gen_range(0..n);
    let j = rng.gen_range(0..n);
    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
    new_genes[lo..=hi].reverse();
    MusicalGenome::new(&new_genes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(99)
    }

    #[test]
    fn test_gaussian_mutation_zero_rate() {
        let g = MusicalGenome::new(&[2.5; 25]);
        let mutated = gaussian_mutation(&g, 0.0, 0.5, &mut test_rng());
        assert_eq!(g, mutated);
    }

    #[test]
    fn test_gaussian_mutation_full_rate() {
        let g = MusicalGenome::new(&[2.5; 25]);
        let mutated = gaussian_mutation(&g, 1.0, 0.5, &mut test_rng());
        // Very unlikely all genes stay at exactly 2.5
        assert_ne!(g, mutated);
    }

    #[test]
    fn test_uniform_mutation_zero_rate() {
        let g = MusicalGenome::new(&[2.5; 25]);
        let mutated = uniform_mutation(&g, 0.0, &mut test_rng());
        assert_eq!(g, mutated);
    }

    #[test]
    fn test_inversion_mutation_preserves_genes() {
        let g = MusicalGenome::new(&[2.5; 25]);
        let mutated = inversion_mutation(&g, &mut test_rng());
        // All-constant genes are invariant under inversion
        assert_eq!(g, mutated);
    }
}
