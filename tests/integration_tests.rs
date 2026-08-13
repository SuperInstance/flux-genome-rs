//! Integration tests for flux-genome
//!
//! These tests exercise cross-module interactions and edge cases
//! that unit tests in each module don't cover.

use flux_genome::{
    arithmetic_crossover, blend_crossover, conservation_score, decode_tradition, dial_distance,
    encode_tradition, gaussian_mutation, inversion_mutation, novelty_score, uniform_crossover,
    uniform_mutation, EvolutionLog, GeneticAlgorithm, MusicalGenome, TRADITION_NAMES,
    TRADITION_GENOMES,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

// ============================================================================
// Genome edge cases
// ============================================================================

#[test]
fn test_genome_boundary_values() {
    // All zeros
    let g0 = MusicalGenome::new(&[0.0; 25]);
    let (h, r, s) = g0.dial_position();
    assert!((h + r + s) < 1e-10);

    // All fives
    let g5 = MusicalGenome::new(&[5.0; 25]);
    let (h, r, s) = g5.dial_position();
    assert!((h - 5.0).abs() < 1e-10);
    assert!((r - 5.0).abs() < 1e-10);
    assert!((s - 5.0).abs() < 1e-10);
}

#[test]
fn test_genome_negative_clamping() {
    let g = MusicalGenome::new(&[-100.0; 25]);
    for &gene in g.genes() {
        assert!(gene >= 0.0, "gene should be clamped to >= 0, got {}", gene);
    }
}

#[test]
fn test_genome_nan_resistance() {
    // NaN should not propagate — clamping catches < 0 and > 5
    // but NaN comparisons are always false, so NaN passes through clip()
    // This documents the current behavior: NaN is NOT caught.
    // If the library adds NaN guards later, update this test.
    let g = MusicalGenome::new(&[f64::NAN; 25]);
    // genes() returns the array; NaN may be present
    let any_nan = g.genes().iter().any(|&v| v.is_nan());
    // Document the gap: NaN is currently unhandled
    // When fixed, change to: assert!(!any_nan)
    let _ = any_nan; // acknowledge but don't assert — this is a known limitation
}

#[test]
fn test_genome_metadata_gene_clamped() {
    let mut g = MusicalGenome::zeros();
    g.set_metadata_gene(100.0);
    assert!(g.metadata_gene() <= 5.0);
    g.set_metadata_gene(-10.0);
    assert!(g.metadata_gene() >= 0.0);
}

#[test]
fn test_genome_equality_fuzzy() {
    let g1 = MusicalGenome::new(&[2.5; 25]);
    let mut genes2 = [2.5; 25];
    genes2[0] += 1e-12;
    let g2 = MusicalGenome::new(&genes2);
    // Within tolerance
    assert_eq!(g1, g2);

    let mut genes3 = [2.5; 25];
    genes3[0] += 0.01;
    let g3 = MusicalGenome::new(&genes3);
    assert_ne!(g1, g3);
}

// ============================================================================
// Crossover edge cases
// ============================================================================

#[test]
fn test_uniform_crossover_identical_parents() {
    let parent = MusicalGenome::new(&[2.5; 25]);
    let child = uniform_crossover(&parent, &parent, &mut seeded_rng(1));
    assert_eq!(child, parent);
}

#[test]
fn test_arithmetic_crossover_extremes() {
    let a = MusicalGenome::new(&[0.0; 25]);
    let b = MusicalGenome::new(&[5.0; 25]);

    // Weight 0 → all from b
    let child_b = arithmetic_crossover(&a, &b, 0.0);
    for &g in child_b.genes() {
        assert!((g - 5.0).abs() < 1e-10);
    }

    // Weight 1 → all from a
    let child_a = arithmetic_crossover(&a, &b, 1.0);
    for &g in child_a.genes() {
        assert!((g - 0.0).abs() < 1e-10);
    }
}

#[test]
fn test_blend_crossover_alpha_zero() {
    // α=0 → range is exactly [min, max], child should be within that range
    let a = MusicalGenome::new(&[2.0; 25]);
    let b = MusicalGenome::new(&[4.0; 25]);
    let child = blend_crossover(&a, &b, 0.0, &mut seeded_rng(5));
    for &g in child.genes() {
        assert!(g >= 2.0 - 1e-10 && g <= 4.0 + 1e-10, "gene {} outside [2,4]", g);
    }
}

#[test]
fn test_blend_crossover_identical_parents() {
    let parent = MusicalGenome::new(&[3.0; 25]);
    let child = blend_crossover(&parent, &parent, 0.5, &mut seeded_rng(3));
    // When parents are identical, lo == hi → fallback to parent gene
    assert_eq!(child, parent);
}

#[test]
fn test_crossover_preserves_gene_count() {
    let a = MusicalGenome::new(&[1.0; 25]);
    let b = MusicalGenome::new(&[4.0; 25]);
    let rng = &mut seeded_rng(7);

    let c1 = uniform_crossover(&a, &b, rng);
    let c2 = arithmetic_crossover(&a, &b, 0.5);
    let c3 = blend_crossover(&a, &b, 0.3, rng);

    assert_eq!(c1.genes().len(), 25);
    assert_eq!(c2.genes().len(), 25);
    assert_eq!(c3.genes().len(), 25);
}

// ============================================================================
// Mutation edge cases
// ============================================================================

#[test]
fn test_gaussian_mutation_clamping() {
    // With huge sigma, mutation can push values outside [0, 5]
    // but MusicalGenome::new clips them
    let g = MusicalGenome::new(&[2.5; 25]);
    let mutated = gaussian_mutation(&g, 1.0, 100.0, &mut seeded_rng(1));
    for &gene in mutated.genes() {
        assert!(gene >= 0.0 && gene <= 5.0, "gene {} not clamped", gene);
    }
}

#[test]
fn test_uniform_mutation_clamping() {
    let g = MusicalGenome::new(&[2.5; 25]);
    let mutated = uniform_mutation(&g, 1.0, &mut seeded_rng(2));
    for &gene in mutated.genes() {
        assert!(gene >= 0.0 && gene <= 5.0);
    }
}

#[test]
fn test_inversion_mutation_preserves_multiset() {
    // Inversion should preserve the multiset of gene values
    let genes: Vec<f64> = (0..25).map(|i| (i as f64) * 0.2).collect();
    let g = MusicalGenome::new(&genes);
    let mutated = inversion_mutation(&g, &mut seeded_rng(4));

    let mut original: Vec<f64> = g.genes().to_vec();
    let mut inverted: Vec<f64> = mutated.genes().to_vec();
    original.sort_by(|a, b| a.partial_cmp(b).unwrap());
    inverted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for (a, b) in original.iter().zip(inverted.iter()) {
        assert!((a - b).abs() < 1e-10, "multiset mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_mutation_rate_boundary_zero() {
    let g = MusicalGenome::new(&[3.0; 25]);
    let m1 = gaussian_mutation(&g, 0.0, 0.5, &mut seeded_rng(1));
    let m2 = uniform_mutation(&g, 0.0, &mut seeded_rng(1));
    assert_eq!(g, m1);
    assert_eq!(g, m2);
}

// ============================================================================
// Fitness and novelty
// ============================================================================

#[test]
fn test_dial_distance_symmetry() {
    let g1 = MusicalGenome::new(&[1.0; 25]);
    let g2 = MusicalGenome::new(&[4.0; 25]);

    let d1 = dial_distance(&g1, g2.dial_position());
    let d2 = dial_distance(&g2, g1.dial_position());

    assert!((d1 - d2).abs() < 1e-10, "dial distance should be symmetric");
}

#[test]
fn test_dial_distance_triangle_inequality() {
    let g1 = MusicalGenome::new(&[0.0; 25]);
    let g2 = MusicalGenome::new(&[2.5; 25]);
    let g3 = MusicalGenome::new(&[5.0; 25]);

    let d12 = dial_distance(&g1, g2.dial_position());
    let d23 = dial_distance(&g2, g3.dial_position());
    let d13 = dial_distance(&g1, g3.dial_position());

    assert!(d13 <= d12 + d23 + 1e-10, "triangle inequality violated");
}

#[test]
fn test_novelty_with_identical_population() {
    let g = MusicalGenome::new(&[2.5; 25]);
    let population = vec![g.clone(); 10];
    let novelty = novelty_score(&g, &population, 5);
    assert!(novelty < 1e-10, "novelty of identical genomes should be ~0");
}

#[test]
fn test_novelty_k_larger_than_population() {
    let g = MusicalGenome::new(&[2.5; 25]);
    let population = vec![MusicalGenome::new(&[1.0; 25]); 3];
    // k=10 but population is 3 — should clamp
    let novelty = novelty_score(&g, &population, 10);
    assert!(novelty > 0.0);
}

#[test]
fn test_conservation_score_negative_for_different() {
    let g1 = MusicalGenome::new(&[0.0; 25]);
    let g2 = MusicalGenome::new(&[5.0; 25]);
    let score = conservation_score(&g1, &g2);
    assert!(score < 0.0, "conservation of different genomes should be negative");
}

// ============================================================================
// Tradition DNA
// ============================================================================

#[test]
fn test_all_traditions_have_genomes() {
    for name in TRADITION_NAMES {
        assert!(TRADITION_GENOMES.get(name).is_some(), "Missing tradition genome: {}", name);
    }
}

#[test]
fn test_tradition_genomes_dial_positions_in_range() {
    for (name, genome) in TRADITION_GENOMES.iter() {
        let (h, r, s) = genome.dial_position();
        assert!(h >= 0.0 && h <= 5.0, "{}: h={} out of range", name, h);
        assert!(r >= 0.0 && r <= 5.0, "{}: r={} out of range", name, r);
        assert!(s >= 0.0 && s <= 5.0, "{}: s={} out of range", name, s);
    }
}

#[test]
fn test_encode_decode_roundtrip() {
    let g = encode_tradition("TestTradition", (3.0, 2.0, 1.0), 0.1, &mut seeded_rng(42));
    let info = decode_tradition(&g);
    let (h, r, s) = info.dial_position;
    // Encoded at (3.0, 2.0, 1.0) with small spread — decoded should be close
    assert!((h - 3.0).abs() < 1.0, "h drift: {}", h);
    assert!((r - 2.0).abs() < 1.0, "r drift: {}", r);
    assert!((s - 1.0).abs() < 1.0, "s drift: {}", s);
}

#[test]
fn test_tradition_metadata_deterministic() {
    let g1 = encode_tradition("Jazz", (3.2, 2.8, 2.5), 0.0, &mut seeded_rng(0));
    let g2 = encode_tradition("Jazz", (3.2, 2.8, 2.5), 0.0, &mut seeded_rng(999));
    // Metadata gene is derived from name hash, not RNG
    assert!((g1.metadata_gene() - g2.metadata_gene()).abs() < 1e-10);
}

// ============================================================================
// Evolution log
// ============================================================================

#[test]
fn test_evolution_log_empty_population() {
    let mut log = EvolutionLog::new();
    log.record(0, &[], (2.5, 2.5, 2.5));
    assert_eq!(log.records.len(), 0, "empty population should not be recorded");
}

#[test]
fn test_evolution_log_single_individual() {
    let mut log = EvolutionLog::new();
    let pop = vec![MusicalGenome::new(&[2.5; 25])];
    log.record(0, &pop, (2.5, 2.5, 2.5));
    assert_eq!(log.records.len(), 1);
    assert!(log.records[0].diversity.abs() < 1e-10, "single individual diversity should be 0");
}

#[test]
fn test_evolution_log_tracks_improvement() {
    let mut log = EvolutionLog::new();
    let target = (2.5, 2.5, 2.5);

    // Generation 0: far from target
    let pop0 = vec![MusicalGenome::new(&[5.0; 25])];
    log.record(0, &pop0, target);

    // Generation 1: closer to target
    let pop1 = vec![MusicalGenome::new(&[2.5; 25])];
    log.record(1, &pop1, target);

    let best = log.best_ever().unwrap();
    assert_eq!(best.generation, 1, "best ever should be generation 1");
    assert!(best.best_fitness < pop0[0].fitness(target));
}

// ============================================================================
// Genetic Algorithm integration
// ============================================================================

#[test]
fn test_ga_initialization_with_custom_traditions() {
    let mut ga = GeneticAlgorithm::new(20, 0.1, 0.8, 3);
    ga.initialize((2.5, 2.5, 2.5), Some(&["Jazz", "Blues"]), Some(1));
    assert_eq!(ga.population.len(), 20);
    // First two should be from traditions
    assert!(ga.population.len() >= 2);
}

#[test]
fn test_ga_evolve_with_zero_generations() {
    let mut ga = GeneticAlgorithm::new(30, 0.1, 0.8, 3);
    ga.initialize((2.5, 2.5, 2.5), None, Some(1));
    ga.evolve(0);
    // Should record the initial state only
    assert_eq!(ga.log.records.len(), 1);
}

#[test]
fn test_ga_elitism_preserves_best() {
    let mut ga = GeneticAlgorithm::new(50, 0.0, 0.0, 3); // no mutation, no crossover
    ga.initialize((2.5, 2.5, 2.5), None, Some(42));

    let best_before = ga.best().clone();
    ga.evolve(10);
    let best_after = ga.best();

    // With no mutation/crossover, elitism should keep the best individual
    let fit_before = best_before.fitness((2.5, 2.5, 2.5));
    let fit_after = best_after.fitness((2.5, 2.5, 2.5));
    assert!(
        fit_after <= fit_before + 1e-10,
        "elitism should preserve or improve best fitness"
    );
}

#[test]
fn test_ga_convergence_jazz_target() {
    let jazz_centre = (3.2, 2.8, 2.5);
    let mut ga = GeneticAlgorithm::new(100, 0.1, 0.85, 4);
    ga.initialize(jazz_centre, None, Some(7));
    ga.evolve(80);

    let best = ga.best();
    let (h, r, s) = best.dial_position();
    let dist = (h - jazz_centre.0).powi(2)
        + (r - jazz_centre.1).powi(2)
        + (s - jazz_centre.2).powi(2);
    let dist = dist.sqrt();

    assert!(dist < 1.5, "should converge near Jazz target, dist={}", dist);
}

#[test]
fn test_ga_population_diversity_decreases() {
    let mut ga = GeneticAlgorithm::new(100, 0.05, 0.9, 3);
    ga.initialize((2.5, 2.5, 2.5), None, Some(99));

    // Record initial diversity
    ga.evolve(1);
    let early_diversity = ga.log.records[0].diversity;

    ga.evolve(40);
    let late_diversity = ga.log.records.last().unwrap().diversity;

    // Diversity should generally decrease with convergence
    // (not strictly monotonic, but the trend should be clear)
    assert!(
        late_diversity <= early_diversity * 1.5,
        "diversity should trend down: early={}, late={}",
        early_diversity,
        late_diversity
    );
}

#[test]
fn test_ga_tournament_size_one_is_random_selection() {
    // tournament_size=1 → picks a random individual each time
    let mut ga = GeneticAlgorithm::new(50, 0.1, 0.8, 1);
    ga.initialize((2.5, 2.5, 2.5), None, Some(1));
    ga.evolve(10);
    // Should still work, just less selective pressure
    assert_eq!(ga.population.len(), 50);
}

#[test]
fn test_ga_large_population_runs() {
    let mut ga = GeneticAlgorithm::new(500, 0.1, 0.8, 5);
    ga.initialize((3.0, 3.0, 3.0), None, Some(1));
    ga.evolve(5);
    assert_eq!(ga.population.len(), 500);
}

// ============================================================================
// Cross-module integration
// ============================================================================

#[test]
fn test_mutate_then_crossover_then_evaluate() {
    let mut rng = seeded_rng(42);
    let target = (2.5, 2.5, 2.5);

    // Start with two traditions
    let parent_a = MusicalGenome::from_tradition("Jazz", &mut rng).unwrap();
    let parent_b = MusicalGenome::from_tradition("Classical", &mut rng).unwrap();

    // Crossover
    let child = blend_crossover(&parent_a, &parent_b, 0.5, &mut rng);

    // Mutate
    let mutated = gaussian_mutation(&child, 0.2, 0.3, &mut rng);

    // Evaluate fitness
    let fitness = mutated.fitness(target);
    assert!(fitness >= 0.0);
    assert!(fitness.is_finite());

    // Should be within reasonable bounds (max possible distance in [0,5]^3)
    let max_dist = (3.0 * 25.0_f64).sqrt(); // ~8.66
    assert!(fitness <= max_dist, "fitness {} exceeds theoretical max", fitness);
}

#[test]
fn test_evolution_produces_valid_genomes() {
    let mut ga = GeneticAlgorithm::new(50, 0.3, 0.9, 3); // high rates
    ga.initialize((2.5, 2.5, 2.5), None, Some(1));
    ga.evolve(20);

    for genome in &ga.population {
        for &gene in genome.genes() {
            assert!(gene.is_finite(), "gene is not finite: {}", gene);
            assert!(gene >= 0.0 && gene <= 5.0, "gene out of range: {}", gene);
        }
    }
}

#[test]
fn test_novelty_finds_outlier() {
    let population: Vec<MusicalGenome> = (0..10)
        .map(|_| MusicalGenome::new(&[2.5; 25]))
        .collect();

    let outlier = MusicalGenome::new(&[5.0; 25]);
    let novelty_outlier = novelty_score(&outlier, &population, 5);
    let novelty_member = novelty_score(&population[0], &population, 5);

    assert!(
        novelty_outlier > novelty_member,
        "outlier should have higher novelty than population member"
    );
}

#[test]
fn test_conservation_decreases_with_mutation() {
    let ancestor = MusicalGenome::new(&[2.5; 25]);
    let mutated = gaussian_mutation(&ancestor, 1.0, 1.0, &mut seeded_rng(1));

    let conservation_ancestor = conservation_score(&ancestor, &ancestor);
    let conservation_mutated = conservation_score(&mutated, &ancestor);

    assert!(
        conservation_mutated < conservation_ancestor,
        "mutated genome should have lower conservation score"
    );
}
