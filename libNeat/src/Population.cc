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

}

Population::Population(const Population& other) {
  organisms = other.organisms;
  generator = other.generator;
  required_ = other.required_;
}

Population& Population::operator=(Population&& rhs) {
  organisms = std::move(rhs.organisms);
  generator = rhs.generator;
  required_ = rhs.required_;
  return *this;
}

Population& Population::operator=(const Population& rhs) {
  organisms = rhs.organisms;
  generator = rhs.generator;
  required_ = rhs.required_;
  return *this;
}

Population Population::Reproduce() {
  // Could consider sorting the most fit first
  // This would then use the champion of each species
  // as the representative for genetic distance.

  // NEAT uses a random member as the representative
  std::random_shuffle(organisms.begin(), organisms.end());


  std::map<unsigned int, std::vector<Organism> > all_species;
  for (auto& species : population_species) {
    species.size = 0;
    species.age++;
    all_species[species.id] = {};
  }
  // speciate
  for(auto& organism : organisms) {
    bool need_new_species = true;
    for(auto& species : all_species) {
      unsigned int id = species.first;
      double dist = organism.genome.GeneticDistance(population_species[id].representative);
      if(dist < required()->species_delta) {
        organism.species = id;
        // only add organism to species if the species is viable
        if (population_species.at(id).age >= required()->stale_species) { continue; }
        species.second.push_back(organism);
        need_new_species = false;
        population_species[id].size++;
        break;
      }
    }

    if(need_new_species) {
      unsigned int id = population_species.size();
      organism.species = id;
      population_species.push_back({id,organism.genome,0,organism.fitness,1});
      assert(all_species.count(id) == 0);
      all_species[id] = { organism };
      //all_species.push_back(std::vector<Organism>{organism});
    }
  }



  for(auto& species_pair : all_species) {
    // skip empty species
    if (species_pair.second.size() == 0) { continue; }

    // If using std::shuffle to randomize representative,
    // must sort species first, to find champion
    std::sort(species_pair.second.begin(), species_pair.second.end(),
              [](auto& a, auto& b) { return a.fitness > b.fitness; });

    // if a species has improved (ie it's champion has better fitness than)
    // the previous champion of that species, then revitalize the species
    auto& champion = species_pair.second.front();
    auto& species = population_species.at(champion.species);
    if (species.best_fitness <= champion.fitness) {
      species.age = 0;
      species.best_fitness = champion.fitness;
    }
  }


  // Find adjusted fitness
  // adj_fitness = fitness/ number_of_genetically_similar_in_species
  for(auto& species : all_species) {
    for(auto& org : species.second) {
      int nearby_in_species = 0;
      for(auto& other : species.second) {
        double dist = org.genome.GeneticDistance(other.genome);
        if(dist < required()->species_delta) {
          nearby_in_species++;
        }
      }
      org.adj_fitness = org.fitness/nearby_in_species;
    }
  }

  // Determine number of children for each species
  double total_adj_fitness = 0;
  std::vector<double> total_adj_fitness_by_species;
  for(auto& species : all_species) {
    double t = 0;
    for(auto& org : species.second) {
      t += org.adj_fitness;
    }
    total_adj_fitness_by_species.push_back(t);
    total_adj_fitness += t;
  }

  double children_per_adj_fitness = required()->population_size / total_adj_fitness;
  std::vector<int> num_children_by_species;
  for(auto& species_adj_fitness : total_adj_fitness_by_species) {
    double num_children = children_per_adj_fitness * species_adj_fitness;
    num_children_by_species.push_back(std::round(num_children));
    // Rounding differences here can cause slightly more or slightly fewer genomes
    // to be created than population_size. This is probably ok, but should be studied
  }


  std::vector<Organism> progeny;
  //std::vector<Genome> progeny;
  auto n = 0u;
  for (auto& species_pair : all_species) {
    auto& species = species_pair.second;
    auto& num_children = num_children_by_species[n++];

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
