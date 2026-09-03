# flux-genome-rs

Rust port of [flux-genome](https://github.com/SuperInstance/flux-genome) — a genetic algorithm framework for evolving musical traditions in dial space.

<p align="center">
  <img src="assets/images/hero.jpg" width="680" alt="A wall of brass dials in a dark workshop, each set to a different angle — a genome of musical tradition, one dial glowing as it is selected">
</p>

## Overview

A `MusicalGenome` is a vector of 25 genes (`f64` in `[0, 5]`) encoding a musical tradition's position in "dial space". The phenotype is a 3-tuple `(harmonic, rhythmic, spectral)` computed by averaging each 8-gene block.

The `GeneticAlgorithm` evolves populations of genomes toward target dial positions using selection, crossover, and mutation.

## Usage

### Create a genome from a tradition

```rust
use flux_genome::MusicalGenome;
use rand::rngs::StdRng;
use rand::SeedableRng;

let mut rng = StdRng::seed_from_u64(42);
let genome = MusicalGenome::from_tradition("Jazz", &mut rng).unwrap();
println!("Jazz dial position: {:?}", genome.dial_position());
```

### Run evolution toward a target

```rust
use flux_genome::{MusicalGenome, GeneticAlgorithm};

let mut ga = GeneticAlgorithm::new(100, 0.1, 0.8, 3);
ga.initialize((2.5, 2.5, 2.5), None, Some(42));
ga.evolve(50);
println!("Best: {}", ga.best());
```

### Use built-in tradition DNA

```rust
use flux_genome::tradition_dna::TRADITION_GENOMES;

let jazz = TRADITION_GENOMES.get("Jazz").unwrap();
println!("Jazz genes: {:?}", jazz.genes());
```

### Apply mutation operators

```rust
use flux_genome::{MusicalGenome, mutation};
use rand::rngs::StdRng;
use rand::SeedableRng;

let mut rng = StdRng::seed_from_u64(42);
let genome = MusicalGenome::random(&mut rng);
let mutated = mutation::gaussian_mutation(&genome, 0.1, 0.5, &mut rng);
```

### Track evolution with logs

```rust
use flux_genome::{GeneticAlgorithm, evolution_log::EvolutionLog};

let mut ga = GeneticAlgorithm::new(100, 0.1, 0.8, 3);
ga.initialize((3.0, 3.0, 3.0), None, Some(42));
let mut log = EvolutionLog::new();
let target = (3.0, 3.0, 3.0);
for gen in 0..100 {
    ga.evolve(1);
    log.record(gen, ga.population(), target);
}
if let Some(best) = log.best_ever() {
    println!("Best fitness: {}", best.best_fitness);
}
```

## Operators

- **Crossover**: uniform, arithmetic, blend (BLX-α)
- **Mutation**: Gaussian, uniform, inversion
- **Fitness**: dial distance, novelty score, conservation score

## Built-in Traditions

| Tradition | (harmonic, rhythmic, spectral) |
|-----------|-------------------------------|
| Jazz | (3.2, 2.8, 2.5) |
| Classical | (1.8, 1.2, 1.5) |
| Rock | (3.5, 3.8, 3.0) |
| Blues | (3.0, 2.5, 2.0) |
| Electronic | (3.8, 4.0, 4.5) |
| Hindustani | (2.5, 3.2, 1.8) |
| Gamelan | (2.0, 3.5, 2.2) |
| Gagaku | (1.5, 1.8, 1.0) |
| WestAfrican | (2.8, 4.2, 2.8) |
| FreeImprovisation | (4.0, 3.5, 3.8) |

## Relation to Python version

This is a pure Rust port with no Python FFI. All algorithms are reimplemented idiomatically in Rust with the same mathematical semantics.

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
