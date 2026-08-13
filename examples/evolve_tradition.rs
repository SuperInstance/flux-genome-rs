//! Example: Evolve a population toward a target musical tradition.
//!
//! Run with: `cargo run --example evolve_tradition`

use flux_genome::{GeneticAlgorithm, MusicalGenome, TRADITION_GENOMES, TRADITION_NAMES};

fn main() {
    // Target: West African polyrhythms
    let west_african = TRADITION_GENOMES
        .get("WestAfrican")
        .expect("WestAfrican tradition must exist");
    let target = west_african.dial_position();
    println!("Target: WestAfrican dial = {:?}", target);
    println!("Starting dial: {:?}", target);
    println!();

    // Create GA with moderate parameters
    let mut ga = GeneticAlgorithm::new(
        100,  // population size
        0.12, // mutation rate
        0.85, // crossover rate
        4,    // tournament size
    );

    // Initialize with ONLY non-African traditions — the GA must evolve toward WestAfrican
    // from Jazz, Classical, Rock, Blues, Electronic, etc. without being seeded with the answer
    let starting_traditions = [
        "Jazz", "Classical", "Rock", "Blues", "Electronic",
        "Hindustani", "Gamelan", "Gagaku", "FreeImprovisation",
    ];
    ga.initialize(target, Some(&starting_traditions), Some(42));

    // Show initial best
    let initial = ga.best().clone();
    println!("Generation  0: best fitness = {:.4}, dial = {:?}", 
             initial.fitness(target), initial.dial_position());

    // Evolve in chunks, printing progress
    for chunk in 0..10 {
        ga.evolve(10);
        let gen = (chunk + 1) * 10;
        let best = ga.best();
        println!("Generation {:2}: best fitness = {:.4}, dial = {:?}",
                 gen, best.fitness(target), best.dial_position());
    }

    // Final report
    println!();
    let final_best = ga.best();
    println!("=== Final Result ===");
    println!("Fitness:  {:.4}", final_best.fitness(target));
    println!("Dial pos: {:?}", final_best.dial_position());
    println!("Target:   {:?}", target);
    println!();

    // Show which tradition is closest
    let mut closest = ("", f64::MAX);
    for name in TRADITION_NAMES {
        if let Some(genome) = TRADITION_GENOMES.get(name) {
            let dist = final_best.fitness(genome.dial_position());
            if dist < closest.1 {
                closest = (name, dist);
            }
        }
    }
    println!("Closest tradition: {} (distance {:.4})", closest.0, closest.1);

    // Show evolution log stats
    println!();
    println!("=== Evolution Log ===");
    if let Some(record) = ga.log.best_ever() {
        println!("Best generation: {}", record.generation);
        println!("Best fitness:    {:.4}", record.best_fitness);
        println!("Mean fitness:    {:.4}", record.mean_fitness);
        println!("Diversity:       {:.4}", record.diversity);
    }
}
