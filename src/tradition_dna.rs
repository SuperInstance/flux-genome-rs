use crate::genome::{MusicalGenome, N_GENES, TRADITION_CENTRES};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Names of the 10 recognised musical traditions.
pub const TRADITION_NAMES: &[&str] = &[
    "Jazz",
    "Classical",
    "Rock",
    "Blues",
    "Electronic",
    "Hindustani",
    "Gamelan",
    "Gagaku",
    "WestAfrican",
    "FreeImprovisation",
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

/// Encode a tradition as a MusicalGenome with controlled variation.
///
/// Each 8-gene block is sampled from N(dial_center_component, spread), clamped to [0, 5].
pub fn encode_tradition(
    name: &str,
    dial_center: (f64, f64, f64),
    spread: f64,
    rng: &mut impl Rng,
) -> MusicalGenome {
    let normal = Normal::new(0.0, spread).unwrap();
    let mut arr = [0.0; N_GENES];
    for (block_start, c) in [(0, dial_center.0), (8, dial_center.1), (16, dial_center.2)] {
        for i in 0..8 {
            let val = c + normal.sample(rng);
            arr[block_start + i] = clip(val, 0.0, 5.0);
        }
    }
    // Use hash of name for metadata gene (deterministic marker)
    let hash_val = name
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    arr[24] = (hash_val % 500) as f64 / 100.0;
    MusicalGenome::new(&arr)
}

/// Decode a genome into a tradition-like summary.
pub fn decode_tradition(genome: &MusicalGenome) -> TraditionInfo {
    TraditionInfo {
        dial_position: genome.dial_position(),
        harmonic_genes: genome.harmonic_genes().to_vec(),
        rhythmic_genes: genome.rhythmic_genes().to_vec(),
        spectral_genes: genome.spectral_genes().to_vec(),
        metadata: genome.metadata_gene(),
    }
}

/// Summary of a decoded tradition genome.
#[derive(Debug, Clone)]
pub struct TraditionInfo {
    pub dial_position: (f64, f64, f64),
    pub harmonic_genes: Vec<f64>,
    pub rhythmic_genes: Vec<f64>,
    pub spectral_genes: Vec<f64>,
    pub metadata: f64,
}

/// Pre-built genomes for the 10 traditions.
///
/// Lazily initialised on first access via `TRADITION_GENOMES::get()`.
pub struct TraditionGenomes {
    genomes: HashMap<String, MusicalGenome>,
}

impl TraditionGenomes {
    /// Initialise all tradition genomes with deterministic seeds.
    pub fn new() -> Self {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut genomes = HashMap::new();
        for (i, &(name, centre)) in TRADITION_CENTRES.iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(i as u64);
            genomes.insert(
                name.to_string(),
                encode_tradition(name, centre, 0.3, &mut rng),
            );
        }
        Self { genomes }
    }

    /// Get a tradition genome by name.
    pub fn get(&self, name: &str) -> Option<&MusicalGenome> {
        self.genomes.get(name)
    }

    /// Get all tradition names.
    pub fn names(&self) -> Vec<&str> {
        self.genomes.keys().map(|s| s.as_str()).collect()
    }

    /// Iterate over all (name, genome) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MusicalGenome)> {
        self.genomes.iter()
    }
}

impl Default for TraditionGenomes {
    fn default() -> Self {
        Self::new()
    }
}

/// Lazy static for tradition genomes.
pub static TRADITION_GENOMES: once_cell::sync::Lazy<TraditionGenomes> =
    once_cell::sync::Lazy::new(TraditionGenomes::new);

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_encode_tradition() {
        let g = encode_tradition("Test", (2.5, 2.5, 2.5), 0.3, &mut test_rng());
        let (h, r, s) = g.dial_position();
        // Should be roughly near (2.5, 2.5, 2.5)
        assert!(h > 1.0 && h < 4.0);
        let (_r, _s) = (r, s);
    }

    #[test]
    fn test_decode_tradition() {
        let g = MusicalGenome::zeros();
        let info = decode_tradition(&g);
        assert_eq!(info.harmonic_genes.len(), 8);
        assert_eq!(info.rhythmic_genes.len(), 8);
        assert_eq!(info.spectral_genes.len(), 8);
    }

    #[test]
    fn test_tradition_genomes() {
        let tg = TraditionGenomes::new();
        assert!(tg.get("Jazz").is_some());
        assert!(tg.get("NonExistent").is_none());
        assert_eq!(tg.names().len(), 10);
    }

    #[test]
    fn test_tradition_names() {
        assert_eq!(TRADITION_NAMES.len(), 10);
        assert!(TRADITION_NAMES.contains(&"Jazz"));
    }
}
