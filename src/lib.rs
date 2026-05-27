//! flux-genome: Genetic algorithm framework for evolving musical traditions in dial space.
//!
//! Rust port of the Python [flux-genome](https://github.com/SuperInstance/flux-genome) library.
//!
//! # Overview
//!
//! A `MusicalGenome` is a vector of 25 genes (f64 in [0, 5]) encoding a musical
//! tradition's position in "dial space". The phenotype is a 3-tuple
//! `(harmonic, rhythmic, spectral)` computed by averaging each 8-gene block.
//!
//! The `GeneticAlgorithm` evolves populations of genomes toward target dial
//! positions using selection, crossover, and mutation operators.

mod crossover;
mod evolution_log;
mod fitness;
mod genome;
mod mutation;
mod population;
mod tradition_dna;

pub use crossover::{arithmetic_crossover, blend_crossover, uniform_crossover};
pub use evolution_log::{EvolutionLog, GenerationRecord};
pub use fitness::{conservation_score, dial_distance, novelty_score};
pub use genome::MusicalGenome;
pub use mutation::{gaussian_mutation, inversion_mutation, uniform_mutation};
pub use population::GeneticAlgorithm;
pub use tradition_dna::{decode_tradition, encode_tradition, TRADITION_GENOMES, TRADITION_NAMES};
