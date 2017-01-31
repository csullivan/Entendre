#include "Population.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <set>
#include <map>

Population::Population(std::vector<Organism> organisms, std::vector<Species> species,
                       std::shared_ptr<RNG> gen, std::shared_ptr<Probabilities> params)
  : organisms(std::move(organisms)), population_species(std::move(species)) {

  required(params); set_generator(gen);

  AdvanceSpeciesAge();
  Speciate();
}

Population::Population(Genome& first,
                       std::shared_ptr<RNG> gen, std::shared_ptr<Probabilities> params) {

  required(params);
  set_generator(gen);
  first.required(params);
  first.set_generator(gen);

  for (auto i=0u; i < params->population_size; i++) {
    organisms.emplace_back(first.RandomizeWeights());
  }

  AdvanceSpeciesAge();
  Speciate();
}

void Population::AdvanceSpeciesAge() {
  for(auto& species : population_species) {
    species.age++;
  }
}

void Population::Speciate() {
  // NEAT uses a random member as the representative
  std::random_shuffle(organisms.begin(), organisms.end());

  for(auto& species : population_species) {
    species.size = 0;
  }

  for(auto& organism : organisms) {
    bool need_new_species = true;
    for(auto& species : population_species) {
      double dist = organism.genome.GeneticDistance(species.representative);
      if(dist < required()->genetic_distance_species_threshold) {
        organism.species = species.id;
        species.size++;
        need_new_species = false;
        break;
      }
    }

    if(need_new_species) {
      unsigned int id = population_species.size();
      organism.species = id;
      population_species.push_back({id, organism.genome, 0, -1, 1});
    }
  }


}

void Population::CalculateAdjustedFitness() {
  // // adj_fitness = fitness / number_in_species
  // for(auto& organism : organisms) {
  //   auto num_in_species = population_species.at(organism.species).size;
  //   organism.adj_fitness = organism.fitness/num_in_species;
  // }

  // adj_fitness = fitness/ number_of_genetically_similar_in_species
  for(auto& org : organisms) {
    int nearby_in_species = 0;

    for(auto& other : organisms) {
      if(org.species == other.species) {
        double dist = org.genome.GeneticDistance(other.genome);
        if(dist < required()->genetic_distance_species_threshold) {
          nearby_in_species++;
        }
      }
    }

    org.adj_fitness = org.fitness/nearby_in_species;
  }
}

Population Population::Reproduce() {
  // Little helper struct, local to function only
  struct species_info {
    double total_adj_fitness;
    int num_children;
    std::vector<Organism> orgs;

    int id() { return orgs.front().species; }
  };

  std::map<unsigned int, species_info> all_species;

  for(auto& organism : organisms) {
    bool species_is_stale =
      (population_species.at(organism.species).age >=
       required()->stale_species_num_generations);

    if(!species_is_stale) {
      all_species[organism.species].orgs.push_back(organism);
    }
  }


  for(auto& species_pair : all_species) {
    species_info& info = species_pair.second;

    // If using std::shuffle to randomize representative,
    // must sort species first, to find champion
    std::sort(info.orgs.begin(), info.orgs.end(),
              [](auto& a, auto& b) { return a.fitness > b.fitness; });

    // if a species has improved (ie it's champion has better fitness than)
    // the previous champion of that species, then revitalize the species
    auto& champion = info.orgs.front();
    auto& species = population_species.at(info.id());
    if(champion.fitness > species.best_fitness) {
      species.age = 0;
      species.best_fitness = champion.fitness;
    }
  }


  // Determine number of children for each species
  double total_adj_fitness = 0;
  std::vector<double> total_adj_fitness_by_species;
  for(auto& species_pair : all_species) {
    species_info& info = species_pair.second;

    info.total_adj_fitness = 0;
    for(auto& org : info.orgs) {
      info.total_adj_fitness += org.adj_fitness;
    }
    total_adj_fitness += info.total_adj_fitness;
  }

  double children_per_adj_fitness = required()->population_size / total_adj_fitness;
  for(auto& species_pair : all_species) {
    species_info& info = species_pair.second;
    info.num_children = std::round(children_per_adj_fitness *
                                   info.total_adj_fitness);
    // Rounding differences here can cause slightly more or slightly fewer genomes
    // to be created than population_size. This is probably ok, but should be studied
  }


  std::vector<Organism> progeny;
  //std::vector<Genome> progeny;
  for (auto& species_pair : all_species) {
    auto& species = species_pair.second.orgs;
    auto& num_children = species_pair.second.num_children;

    for(int i=0; i<num_children; i++) {
      if(i==0 && species.size() > required()->min_size_for_champion) {
        // Preserve the champion of large species.
        progeny.emplace_back(species.front().genome);
      } else {
        // Everyone else can mate

        float culling_ratio = required()->culling_ratio;

        // if only one organism in the species
        // or if two organisms in a species and culling ratio is at least half
        // then just take the more fit of the organisms
        if (species.size() == 1 ||
            (species.size() == 2 && culling_ratio <= 0.5 )) {
          auto mutant = species.front().genome;
          mutant.Mutate();
          progeny.emplace_back(mutant);
          continue;
        }

        int idx1 = random()*species.size()*culling_ratio;
        int idx2 = random()*species.size()*culling_ratio;
        // while (idx1 == idx2) {
        //   idx1 = random()*species.size()*culling_ratio;
        //   idx2 = random()*species.size()*culling_ratio;
        // }
        Organism& parent1 = species[idx1];
        Organism& parent2 = species[idx2];
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

  return Population(progeny, population_species, get_generator(), required());
}



NeuralNet* Population::BestNet() const {
  NeuralNet* output = nullptr;
  double best_fitness = -std::numeric_limits<double>::max();
  for(auto& org : organisms) {
    if(org.fitness > best_fitness) {
      output = const_cast<NeuralNet*>(&org.network);
      best_fitness = org.fitness;
    }
  }
  return output;
}

unsigned int Population::NumViableSpecies() {
  size_t num_viable = 0;
  for(auto& species : population_species) {
    if (species.size > 0) { num_viable++; }
  }
  return num_viable;
}
unsigned int Population::NumSpecies() {
  return population_species.size();
}

unsigned int Population::SpeciesSize(size_t i) const {
  return population_species.at(i).size;
}

std::pair<double, double> Population::MeanStdDev() const {
  if(organisms.size() == 0) {
    return {std::sqrt(-1), std::sqrt(-1)};
  }

  double cumsum = 0;
  double cumsum2 = 0;
  for(auto& org : organisms) {
    cumsum += org.fitness;
    cumsum2 += org.fitness*org.fitness;
  }

  double mean = cumsum/organisms.size();
  double variance = cumsum2/organisms.size() - mean*mean;
  return {mean, std::sqrt(variance)};
}
