use crate::genome::MusicalGenome;
use rand::Rng;

/// Uniform crossover: each gene comes from either parent with 50% chance.
pub fn uniform_crossover(
    parent_a: &MusicalGenome,
    parent_b: &MusicalGenome,
    rng: &mut impl Rng,
) -> MusicalGenome {
    let ga = parent_a.genes();
    let gb = parent_b.genes();
    let mut child = [0.0; 25];
    for i in 0..25 {
        child[i] = if rng.gen_bool(0.5) { ga[i] } else { gb[i] };
    }
    MusicalGenome::new(&child)
}

/// Arithmetic crossover: weighted average of parents.
///
/// `child[i] = weight * a[i] + (1 - weight) * b[i]`
pub fn arithmetic_crossover(
    parent_a: &MusicalGenome,
    parent_b: &MusicalGenome,
    weight: f64,
) -> MusicalGenome {
    let ga = parent_a.genes();
    let gb = parent_b.genes();
    let mut child = [0.0; 25];
    for i in 0..25 {
        child[i] = weight * ga[i] + (1.0 - weight) * gb[i];
    }
    MusicalGenome::new(&child)
}

/// BLX-α crossover: sample each child gene from an extended interval.
///
/// For each gene: `lo = min(a,b) - α*|a-b|`, `hi = max(a,b) + α*|a-b|`
pub fn blend_crossover(
    parent_a: &MusicalGenome,
    parent_b: &MusicalGenome,
    alpha: f64,
    rng: &mut impl Rng,
) -> MusicalGenome {
    let ga = parent_a.genes();
    let gb = parent_b.genes();
    let mut child = [0.0; 25];
    for i in 0..25 {
        let (a, b) = (ga[i], gb[i]);
        let lo = a.min(b) - alpha * (a - b).abs();
        let hi = a.max(b) + alpha * (a - b).abs();
        if lo >= hi {
            child[i] = a; // fallback when range is empty
        } else {
            child[i] = rng.gen_range(lo..hi);
        }
    }
    MusicalGenome::new(&child)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(123)
    }

    fn parent_a() -> MusicalGenome {
        MusicalGenome::new(&[1.0; 25])
    }
    fn parent_b() -> MusicalGenome {
        MusicalGenome::new(&[3.0; 25])
    }

    #[test]
    fn test_uniform_crossover() {
        let mut rng = test_rng();
        let child = uniform_crossover(&parent_a(), &parent_b(), &mut rng);
        for &g in child.genes() {
            assert!((g - 1.0).abs() < 1e-10 || (g - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_arithmetic_crossover_equal() {
        let child = arithmetic_crossover(&parent_a(), &parent_b(), 0.5);
        for &g in child.genes() {
            assert!((g - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_arithmetic_crossover_weight_1() {
        let child = arithmetic_crossover(&parent_a(), &parent_b(), 1.0);
        for &g in child.genes() {
            assert!((g - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_blend_crossover() {
        let mut rng = test_rng();
        let child = blend_crossover(&parent_a(), &parent_b(), 0.5, &mut rng);
        // Should be between 0 and 5 (clamped)
        for &g in child.genes() {
            assert!(g >= 0.0 && g <= 5.0);
        }
    }
}
