use crate::genome::MusicalGenome;

/// Snapshot of a single generation.
#[derive(Debug, Clone)]
pub struct GenerationRecord {
    pub generation: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub diversity: f64,
    pub best_dial: (f64, f64, f64),
}

/// Accumulates generation-by-generation evolution statistics.
#[derive(Debug, Clone, Default)]
pub struct EvolutionLog {
    pub records: Vec<GenerationRecord>,
}

impl EvolutionLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a generation snapshot.
    pub fn record(
        &mut self,
        generation: usize,
        population: &[MusicalGenome],
        target_dial: (f64, f64, f64),
    ) {
        let n = population.len();
        if n == 0 {
            return;
        }

        let fitnesses: Vec<f64> = population.iter().map(|g| g.fitness(target_dial)).collect();
        let best_idx = fitnesses
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let best_fitness = fitnesses[best_idx];
        let mean_fitness = fitnesses.iter().sum::<f64>() / n as f64;
        let best_dial = population[best_idx].dial_position();

        // Diversity: mean pairwise Euclidean distance
        let diversity = if n > 1 {
            let mut total_dist = 0.0;
            let mut count = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    total_dist += gene_distance(population[i].genes(), population[j].genes());
                    count += 1;
                }
            }
            total_dist / count as f64 * 2.0 // Mean pairwise
        } else {
            0.0
        };

        self.records.push(GenerationRecord {
            generation,
            best_fitness,
            mean_fitness,
            diversity,
            best_dial,
        });
    }

    /// Return the generation record with the lowest best_fitness.
    pub fn best_ever(&self) -> Option<&GenerationRecord> {
        self.records
            .iter()
            .min_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap())
    }
}

fn gene_distance(a: &[f64; 25], b: &[f64; 25]) -> f64 {
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
    fn test_empty_log() {
        let log = EvolutionLog::new();
        assert!(log.best_ever().is_none());
    }

    #[test]
    fn test_record() {
        let mut log = EvolutionLog::new();
        let pop = vec![MusicalGenome::zeros()];
        log.record(0, &pop, (2.5, 2.5, 2.5));
        assert_eq!(log.records.len(), 1);
        assert!(log.best_ever().is_some());
    }
}
