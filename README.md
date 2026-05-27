# flux-genome-rs

Rust port of [flux-genome](https://github.com/SuperInstance/flux-genome) — a genetic algorithm framework for evolving musical traditions in dial space.

## Overview

A `MusicalGenome` is a vector of 25 genes (f64 in [0, 5]) encoding a musical tradition's position in "dial space". The phenotype is a 3-tuple `(harmonic, rhythmic, spectral)` computed by averaging each 8-gene block.

The `GeneticAlgorithm` evolves populations of genomes toward target dial positions using selection, crossover, and mutation.

## Usage

```rust
use flux_genome::{MusicalGenome, GeneticAlgorithm};

// Create a random genome
use rand::rngs::StdRng;
use rand::SeedableRng;
let mut rng = StdRng::seed_from_u64(42);
let genome = MusicalGenome::random(&mut rng);
println!("Dial position: {:?}", genome.dial_position());

// Run evolution
let mut ga = GeneticAlgorithm::new(100, 0.1, 0.8, 3);
ga.initialize((2.5, 2.5, 2.5), None, Some(42));
ga.evolve(50);
println!("Best: {}", ga.best());
```

## Operators

- **Crossover**: uniform, arithmetic, blend (BLX-α)
- **Mutation**: Gaussian, uniform, inversion
- **Fitness**: dial distance, novelty score, conservation score

## Relation to Python version

This is a pure Rust port with no Python FFI. All algorithms are reimplemented idiomatically in Rust with the same mathematical semantics.

## License

MIT
