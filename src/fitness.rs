use crate::genome::MusicalGenome;

/// Euclidean distance in 3D dial space to a target.
pub fn dial_distance(genome: &MusicalGenome, target: (f64, f64, f64)) -> f64 {
    genome.fitness(target)
}

/// Novelty score: mean distance to k nearest neighbours in the population.
pub fn novelty_score(genome: &MusicalGenome, population: &[MusicalGenome], k: usize) -> f64 {
    if population.is_empty() {
        return 0.0;
    }
    let mut dists: Vec<f64> = population
        .iter()
        .map(|other| euclidean_genes(genome.genes(), other.genes()))
        .collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = k.min(dists.len());
    dists[..k].iter().sum::<f64>() / k as f64
}

/// Conservation score: negative Euclidean distance to an ancestor.
pub fn conservation_score(genome: &MusicalGenome, ancestor: &MusicalGenome) -> f64 {
    -euclidean_genes(genome.genes(), ancestor.genes())
}

fn euclidean_genes(a: &[f64; 25], b: &[f64; 25]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dial_distance_zero() {
        let mut genes = vec![0.0; 25];
        genes[0..8].fill(2.5);
        genes[8..16].fill(2.5);
        genes[16..24].fill(2.5);
        let g = MusicalGenome::new(&genes);
        let d = dial_distance(&g, (2.5, 2.5, 2.5));
        assert!(d < 1e-10);
    }

    #[test]
    fn test_dial_distance_nonzero() {
        let mut genes = vec![0.0; 25];
        genes[0..8].fill(3.0);
        genes[8..16].fill(3.0);
        genes[16..24].fill(3.0);
        let g = MusicalGenome::new(&genes);
        let d = dial_distance(&g, (0.0, 0.0, 0.0));
        assert!(d > 0.0);
    }

    #[test]
    fn test_novelty_empty() {
        let g = MusicalGenome::zeros();
        assert_eq!(novelty_score(&g, &[], 5), 0.0);
    }

    #[test]
    fn test_conservation_identical() {
        let g = MusicalGenome::zeros();
        let score = conservation_score(&g, &g);
        assert!((score - 0.0).abs() < 1e-10);
    }
}
