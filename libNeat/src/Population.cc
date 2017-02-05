#include "Population.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <set>
#include <map>

Population::Population(std::vector<Species> species,
                       std::shared_ptr<RNG> gen, std::shared_ptr<Probabilities> params)
  : species(std::move(species)) {

  required(params); set_generator(gen);
}

Population::Population(Genome& first,
                       std::shared_ptr<RNG> gen, std::shared_ptr<Probabilities> params) {

  required(params);
  set_generator(gen);
  first.required(params);
  first.set_generator(gen);

  std::vector<Genome> genomes;

  for (auto i=0u; i < params->population_size; i++) {
    genomes.push_back(first.RandomizeWeights());
  }

  Speciate(species, genomes);
}

void Population::Speciate(std::vector<Species>& species,
                          const std::vector<Genome>& genomes) {
  for(auto& genome : genomes) {
    bool need_new_species = true;
    for(auto& spec : species) {
      double dist = genome.GeneticDistance(spec.representative);
      if(dist < required()->genetic_distance_species_threshold) {
        spec.organisms.push_back(genome);
        need_new_species = false;
        break;
      }
    }

    if(need_new_species) {
      Species new_spec;
      new_spec.id = random()*(1<<24);
      new_spec.representative = genome;
      new_spec.age = 0;
      new_spec.best_fitness = 0;
      new_spec.organisms.push_back(genome);

      species.push_back(new_spec);
    }
  }
}

void Population::CalculateAdjustedFitness() {
  // // adj_fitness = fitness / number_in_species
  // for(auto& spec : species) {
  //   auto num_in_species = spec.organisms.size();
  //   for(auto& org : spec.organisms) {
  //     org.adj_fitness = org.fitness / num_in_species
  //   }
  // }


  // adj_fitness = fitness/ number_of_genetically_similar_in_species
  for(auto& spec : species) {
    for(auto& org : spec.organisms) {
      int nearby_in_species = 0;

      for(auto& other : spec.organisms) {
        double dist = org.genome.GeneticDistance(other.genome);
        if(dist < required()->genetic_distance_species_threshold) {
          nearby_in_species++;
        }
      }

      org.adj_fitness = org.fitness/nearby_in_species;
    }
  }
}

Population Population::Reproduce() {
  // Generate the list of species for the next generation.
  std::vector<Species> next_gen_species;
  for(auto& spec : species) {
    std::sort(spec.organisms.begin(), spec.organisms.end(),
              [](auto& a, auto& b) { return a.fitness > b.fitness; });

    Species new_spec;
    new_spec.id = spec.id;
    new_spec.age = spec.age + 1;
    new_spec.best_fitness = spec.best_fitness;

    bool species_has_members = spec.organisms.size() > 0;

    if(species_has_members) {
      auto& champion = spec.organisms.front();
      if(champion.fitness > spec.best_fitness) {
        new_spec.age = 0;
        new_spec.best_fitness = champion.fitness;
      }
    }

    if(species_has_members &&
       required()->species_representative_from_previous_gen) {
      unsigned int rep_id = random()*spec.organisms.size();
      new_spec.representative = spec.organisms[rep_id].genome;
    }

    if(species_has_members ||
       required()->keep_empty_species) {
      next_gen_species.push_back(new_spec);
    }
  }

  // Determine total adjusted fitness for each species, and for the
  // entire population.
  double total_adj_fitness = 0;
  std::vector<double> total_adj_fitness_by_species;
  for(auto& spec : species) {
    double species_total_adj_fitness = 0;
    for(auto& org : spec.organisms) {
      species_total_adj_fitness += org.adj_fitness;
    }
    total_adj_fitness += species_total_adj_fitness;
    total_adj_fitness_by_species.push_back(species_total_adj_fitness);
  }

  // Determine number of children for each species.
  double children_per_adj_fitness = required()->population_size / total_adj_fitness;
  std::vector<int> num_children_by_species;
  for(double spec_total_adj_fitness : total_adj_fitness_by_species) {
    // Rounding differences here can cause slightly more or slightly
    // fewer genomes to be created than population_size. This is
    // probably ok, but should be studied.
    int num_children = std::round(children_per_adj_fitness *
                                  spec_total_adj_fitness);
    num_children_by_species.push_back(num_children);
  }


  std::vector<Genome> progeny;
  //std::vector<Genome> progeny;
  for(unsigned int i=0; i<species.size(); i++) {
    auto& org_list = species[i].organisms;
    double num_children = num_children_by_species[i];

    for(int i=0; i<num_children; i++) {
      if(i==0 && org_list.size() > required()->min_size_for_champion) {
        // Preserve the champion of large species.
        progeny.emplace_back(org_list.front().genome);
      } else {
        // Everyone else can mate
        float culling_ratio = required()->culling_ratio;

        // If only one organisms would be allowed to reproduce, just
        // take that one organism.
        if (org_list.size()*culling_ratio <= 1) {
          auto mutant = org_list.front().genome;
          mutant.Mutate();
          progeny.emplace_back(mutant);
          continue;
        }

        int idx1 = random()*org_list.size()*culling_ratio;
        int idx2 = random()*org_list.size()*culling_ratio;
        // while (idx1 == idx2) {
        //   idx1 = random()*org_list.size()*culling_ratio;
        //   idx2 = random()*org_list.size()*culling_ratio;
        // }
        Organism& parent1 = org_list[idx1];
        Organism& parent2 = org_list[idx2];
        Genome child;

        // determine relative fitness for mating
        if (parent1.fitness > parent2.fitness) {
          child = parent1.genome.MateWith(parent2.genome);
        } else if (parent2.fitness > parent1.fitness) {
          child = parent2.genome.MateWith(parent1.genome);
        } else {
          // break a fitness tie with a check on size
          if (parent1.genome.Size() > parent2.genome.Size()) {
            child = parent1.genome.MateWith(parent2.genome);
          }
          else { // equal size or parent 2 is larger
            child = parent2.genome.MateWith(parent1.genome);
          }
        }
        child.Mutate();
        progeny.emplace_back(child);


      }
    }
  }

  Speciate(next_gen_species, progeny);

  return Population(next_gen_species, get_generator(), required());
}



NeuralNet* Population::BestNet() const {
  NeuralNet* output = nullptr;
  double best_fitness = -std::numeric_limits<double>::max();
  for(auto& spec : species) {
    for(auto& org : spec.organisms) {
      if(org.fitness > best_fitness) {
        output = const_cast<NeuralNet*>(&org.network);
        best_fitness = org.fitness;
      }
    }
  }
  return output;
}

unsigned int Population::NumViableSpecies() {
  size_t num_viable = 0;
  for(auto& spec : species) {
    if (spec.organisms.size() > 0) {
      num_viable++;
    }
  }
  return num_viable;
}
unsigned int Population::NumSpecies() {
  return species.size();
}

unsigned int Population::SpeciesSize(size_t i) const {
  return species.at(i).organisms.size();
}

std::pair<double, double> Population::MeanStdDev() const {
  double cumsum = 0;
  double cumsum2 = 0;
  int n = 0;
  for(auto& spec : species) {
    n += spec.organisms.size();
    for(auto& org : spec.organisms) {
      cumsum += org.fitness;
      cumsum2 += org.fitness*org.fitness;
    }
  }

  if(n == 0) {
    return {std::sqrt(-1), std::sqrt(-1)};
  }

  double mean = cumsum/n;
  double variance = cumsum2/n - mean*mean;
  return {mean, std::sqrt(variance)};
}
