use rand::Rng;
use std::fmt;

/// Number of genes in a MusicalGenome.
pub const N_GENES: usize = 25;

/// Approximate dial centres for the 10 traditions (harmonic, rhythmic, spectral).
pub(crate) const TRADITION_CENTRES: &[(&str, (f64, f64, f64))] = &[
    ("Jazz", (3.2, 2.8, 2.5)),
    ("Classical", (1.8, 1.2, 1.5)),
    ("Rock", (3.5, 3.8, 3.0)),
    ("Blues", (3.0, 2.5, 2.0)),
    ("Electronic", (3.8, 4.0, 4.5)),
    ("Hindustani", (2.5, 3.2, 1.8)),
    ("Gamelan", (2.0, 3.5, 2.2)),
    ("Gagaku", (1.5, 1.8, 1.0)),
    ("WestAfrican", (2.8, 4.2, 2.8)),
    ("FreeImprovisation", (4.0, 3.5, 3.8)),
];

fn clip(val: f64, lo: f64, hi: f64) -> f64 {
    if val < lo {
        lo
    } else if val > hi {
        hi
    } else {
        val
    }
}

/// A genome encoding a musical tradition's position in dial space.
///
/// A vector of 25 genes (f64 in [0, 5]) organized as:
/// - `harmonic_genes` (indices 0–7): harmonic tension
/// - `rhythmic_genes` (indices 8–15): rhythmic complexity
/// - `spectral_genes` (indices 16–23): spectral density
/// - `metadata_gene` (index 24): generation/parent info
///
/// The phenotype (dial position) is `(h, r, s)` where each is the mean
/// of its 8-gene block.
#[derive(Clone, Debug)]
pub struct MusicalGenome {
    genes: [f64; N_GENES],
}

impl MusicalGenome {
    /// Create a new genome from a slice of 25 values, clamped to [0, 5].
    ///
    /// # Panics
    /// Panics if `genes` does not have exactly 25 elements.
    pub fn new(genes: &[f64]) -> Self {
        assert_eq!(genes.len(), N_GENES, "genes must have {} elements", N_GENES);
        let mut arr = [0.0; N_GENES];
        for (i, &g) in genes.iter().enumerate() {
            arr[i] = clip(g, 0.0, 5.0);
        }
        Self { genes: arr }
    }

    /// Create a genome with all zeros.
    pub fn zeros() -> Self {
        Self {
            genes: [0.0; N_GENES],
        }
    }

    /// Create a random genome with values uniformly in [0, 5].
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut arr = [0.0; N_GENES];
        for g in &mut arr {
            *g = rng.gen_range(0.0..5.0);
        }
        Self { genes: arr }
    }

    /// Create a genome from a known tradition's dial position.
    ///
    /// Fills each 8-gene block around the tradition's centre with small
    /// Gaussian variation (σ = 0.3).
    pub fn from_tradition(name: &str, rng: &mut impl Rng) -> Result<Self, String> {
        let centre = TRADITION_CENTRES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, c)| *c)
            .ok_or_else(|| format!("Unknown tradition '{}'", name))?;

        let mut arr = [0.0; N_GENES];
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, 0.3).unwrap();
        for (block_start, c) in [(0, centre.0), (8, centre.1), (16, centre.2)] {
            for i in 0..8 {
                let val = c + normal.sample(rng);
                arr[block_start + i] = clip(val, 0.0, 5.0);
            }
        }
        Ok(Self { genes: arr })
    }

    /// Get the full gene array.
    pub fn genes(&self) -> &[f64; N_GENES] {
        &self.genes
    }

    /// Harmonic genes (indices 0–7).
    pub fn harmonic_genes(&self) -> &[f64] {
        &self.genes[0..8]
    }

    /// Rhythmic genes (indices 8–15).
    pub fn rhythmic_genes(&self) -> &[f64] {
        &self.genes[8..16]
    }

    /// Spectral genes (indices 16–23).
    pub fn spectral_genes(&self) -> &[f64] {
        &self.genes[16..24]
    }

    /// Metadata gene (index 24).
    pub fn metadata_gene(&self) -> f64 {
        self.genes[24]
    }

    /// Set the metadata gene, clamped to [0, 5].
    pub fn set_metadata_gene(&mut self, value: f64) {
        self.genes[24] = clip(value, 0.0, 5.0);
    }

    /// Express genes as a 3D dial position (harmonic, rhythmic, spectral).
    ///
    /// Each component is the mean of its 8-gene block.
    pub fn dial_position(&self) -> (f64, f64, f64) {
        let h = self.genes[0..8].iter().sum::<f64>() / 8.0;
        let r = self.genes[8..16].iter().sum::<f64>() / 8.0;
        let s = self.genes[16..24].iter().sum::<f64>() / 8.0;
        (h, r, s)
    }

    /// Euclidean distance in dial space to a target dial position.
    ///
    /// Lower is better. `f = √((h−hₜ)² + (r−rₜ)² + (s−sₜ)²)`
    pub fn fitness(&self, target_dial: (f64, f64, f64)) -> f64 {
        let pos = self.dial_position();
        let dh = pos.0 - target_dial.0;
        let dr = pos.1 - target_dial.1;
        let ds = pos.2 - target_dial.2;
        (dh * dh + dr * dr + ds * ds).sqrt()
    }
}

impl PartialEq for MusicalGenome {
    fn eq(&self, other: &Self) -> bool {
        self.genes
            .iter()
            .zip(other.genes.iter())
            .all(|(a, b)| (a - b).abs() < 1e-10)
    }
}

impl fmt::Display for MusicalGenome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (h, r, s) = self.dial_position();
        write!(f, "MusicalGenome(dial=({:.2}, {:.2}, {:.2}))", h, r, s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_new_valid() {
        let genes: Vec<f64> = (0..25).map(|i| (i as f64) * 0.2).collect();
        let g = MusicalGenome::new(&genes);
        assert_eq!(g.genes().len(), 25);
    }

    #[test]
    #[should_panic]
    fn test_new_wrong_length() {
        MusicalGenome::new(&[1.0; 10]);
    }

    #[test]
    fn test_clamping() {
        let genes: Vec<f64> = (0..25).map(|_| 10.0).collect();
        let g = MusicalGenome::new(&genes);
        assert!(g.genes().iter().all(|&v| v <= 5.0));
    }

    #[test]
    fn test_dial_position() {
        let g = MusicalGenome::zeros();
        let (h, _r, _s) = g.dial_position();
        let _s = _s; // use
        assert!((h - 0.0).abs() < 1e-10);
        assert!((_r - 0.0).abs() < 1e-10);
        assert!((_s - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gene_blocks() {
        let mut genes = vec![0.0; 25];
        genes[0..8].fill(1.0);
        genes[8..16].fill(2.0);
        genes[16..24].fill(3.0);
        genes[24] = 4.0;
        let g = MusicalGenome::new(&genes);
        assert!(g.harmonic_genes().iter().all(|&v| (v - 1.0).abs() < 1e-10));
        assert!(g.rhythmic_genes().iter().all(|&v| (v - 2.0).abs() < 1e-10));
        assert!(g.spectral_genes().iter().all(|&v| (v - 3.0).abs() < 1e-10));
        assert!((g.metadata_gene() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_fitness() {
        let mut genes = vec![0.0; 25];
        genes[0..8].fill(2.5);
        genes[8..16].fill(2.5);
        genes[16..24].fill(2.5);
        let g = MusicalGenome::new(&genes);
        let f = g.fitness((2.5, 2.5, 2.5));
        assert!(f < 1e-10, "fitness should be ~0, got {}", f);
    }

    #[test]
    fn test_from_tradition() {
        let mut rng = test_rng();
        let g = MusicalGenome::from_tradition("Jazz", &mut rng).unwrap();
        let (h, r, s) = g.dial_position();
        // Should be roughly around (3.2, 2.8, 2.5)
        assert!(h > 2.0 && h < 4.5, "h={}", h);
        assert!(r > 1.5 && r < 4.0, "r={}", r);
    }

    #[test]
    fn test_from_unknown_tradition() {
        let mut rng = test_rng();
        assert!(MusicalGenome::from_tradition("NonExistent", &mut rng).is_err());
    }

    #[test]
    fn test_random() {
        let mut rng = test_rng();
        let g = MusicalGenome::random(&mut rng);
        assert!(g.genes().iter().all(|&v| v >= 0.0 && v <= 5.0));
    }

    #[test]
    fn test_copy() {
        let g = MusicalGenome::zeros();
        let g2 = g.clone();
        assert_eq!(g, g2);
    }

    #[test]
    fn test_display() {
        let g = MusicalGenome::zeros();
        let s = format!("{}", g);
        assert!(s.contains("MusicalGenome"));
    }
}
