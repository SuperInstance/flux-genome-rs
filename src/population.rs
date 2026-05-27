use crate::crossover::blend_crossover;
use crate::evolution_log::EvolutionLog;
use crate::genome::MusicalGenome;
use crate::mutation::gaussian_mutation;
use crate::tradition_dna::TRADITION_GENOMES;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Genetic algorithm for evolving musical traditions.
pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub tournament_size: usize,
    pub population: Vec<MusicalGenome>,
    pub log: EvolutionLog,
    target: Option<(f64, f64, f64)>,
    rng: StdRng,
}

impl GeneticAlgorithm {
    pub fn new(
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        tournament_size: usize,
    ) -> Self {
        Self {
            population_size,
            mutation_rate,
            crossover_rate,
            tournament_size,
            population: Vec::new(),
            log: EvolutionLog::new(),
            target: None,
            rng: StdRng::seed_from_u64(0),
        }
    }

    /// Seed the population with known traditions plus random genomes.
    pub fn initialize(
        &mut self,
        target_dial: (f64, f64, f64),
        traditions: Option<&[&str]>,
        seed: Option<u64>,
    ) {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }
        self.target = Some(target_dial);
        self.population.clear();

        let trad_names: Vec<&str> = traditions
            .map(|t| t.to_vec())
            .unwrap_or_else(|| super::tradition_dna::TRADITION_NAMES.to_vec());

        for name in &trad_names {
            if let Some(genome) = TRADITION_GENOMES.get(name) {
                self.population.push(genome.clone());
            } else {
                self.population.push(MusicalGenome::random(&mut self.rng));
            }
        }

        while self.population.len() < self.population_size {
            self.population.push(MusicalGenome::random(&mut self.rng));
        }
        self.population.truncate(self.population_size);
        assert!(
            !self.population.is_empty(),
            "Population must not be empty after initialization"
        );
    }

    fn tournament_select(&mut self) -> usize {
        let target = self.target.unwrap();
        let mut best_idx = 0;
        let mut best_fit = f64::MAX;
        for _ in 0..self.tournament_size {
            let idx = self.rng.gen_range(0..self.population.len());
            let fit = self.population[idx].fitness(target);
            if fit < best_fit {
                best_fit = fit;
                best_idx = idx;
            }
        }
        best_idx
    }

    /// Run evolution for n_generations.
    pub fn evolve(&mut self, n_generations: usize) {
        let target = self.target.expect("Call initialize() before evolve()");
        self.log = EvolutionLog::new();

        for gen in 0..n_generations {
            self.log.record(gen, &self.population, target);

            let mut new_pop = Vec::new();
            // Elitism: carry forward the best individual
            let best = self
                .population
                .iter()
                .min_by(|a, b| a.fitness(target).partial_cmp(&b.fitness(target)).unwrap())
                .unwrap()
                .clone();
            new_pop.push(best);

            while new_pop.len() < self.population_size {
                let p1_idx = self.tournament_select();
                let p2_idx = self.tournament_select();
                let child = if self.rng.gen::<f64>() < self.crossover_rate {
                    blend_crossover(
                        &self.population[p1_idx],
                        &self.population[p2_idx],
                        0.5,
                        &mut self.rng,
                    )
                } else {
                    self.population[p1_idx].clone()
                };
                let child = gaussian_mutation(&child, self.mutation_rate, 0.3, &mut self.rng);
                new_pop.push(child);
            }

            self.population = new_pop;
            self.population.truncate(self.population_size);
        }
        self.log.record(n_generations, &self.population, target);
    }

    /// Return the current best genome (lowest fitness).
    pub fn best(&self) -> &MusicalGenome {
        let target = self.target.expect("Population not initialized");
        self.population
            .iter()
            .min_by(|a, b| a.fitness(target).partial_cmp(&b.fitness(target)).unwrap())
            .expect("Population is empty")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolution_improves() {
        let mut ga = GeneticAlgorithm::new(50, 0.1, 0.8, 3);
        ga.initialize((2.5, 2.5, 2.5), None, Some(42));

        let initial_best = ga.best().fitness((2.5, 2.5, 2.5));
        ga.evolve(50);
        let final_best = ga.best().fitness((2.5, 2.5, 2.5));

        assert!(
            final_best <= initial_best,
            "Evolution should not worsen fitness"
        );
        assert_eq!(ga.log.records.len(), 51); // 50 + 1 final
    }

    #[test]
    fn test_best_ever() {
        let mut ga = GeneticAlgorithm::new(30, 0.1, 0.8, 3);
        ga.initialize((3.0, 2.0, 1.0), None, Some(7));
        ga.evolve(20);
        assert!(ga.log.best_ever().is_some());
        assert!(ga.log.best_ever().unwrap().best_fitness < 5.0);
    }

    #[test]
    fn test_population_size() {
        let mut ga = GeneticAlgorithm::new(100, 0.1, 0.8, 3);
        ga.initialize((2.5, 2.5, 2.5), None, Some(1));
        assert_eq!(ga.population.len(), 100);
        ga.evolve(10);
        assert_eq!(ga.population.len(), 100);
    }

    #[test]
    fn test_convergence_to_target() {
        let mut ga = GeneticAlgorithm::new(200, 0.15, 0.9, 5);
        ga.initialize((2.5, 2.5, 2.5), None, Some(123));
        ga.evolve(100);
        let best = ga.best();
        let (h, r, s) = best.dial_position();
        // Should be reasonably close after 100 generations with 200 pop
        assert!((h - 2.5).abs() < 1.0, "h={}", h);
        let (_r, _s) = (r, s);
    }
}
