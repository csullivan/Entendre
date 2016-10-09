#include "Population.hh"
#include <algorithm>
#include <cassert>

Population::Population(std::vector<Genome> population,
                       std::shared_ptr<RNG> gen, std::shared_ptr<Probabilities> params)
  : population(std::move(population)) {

  required(params); set_generator(gen);

  for(auto& genome : this->population) {
    genome.required(params);
    genome.set_generator(gen);
  }

  build_networks();
}

Population::Population(Genome& first,
                       std::shared_ptr<RNG> gen, std::shared_ptr<Probabilities> params) {

  required(params);
  set_generator(gen);
  first.required(params);
  first.set_generator(gen);

  for (auto i=0u; i < params->population_size; i++) {
    population.push_back(first.RandomizeWeights());
  }

  build_networks();
}

Population::Population(const Population& other) {
  population = other.population;
  networks = other.networks;
  organisms = other.organisms;
  generator = other.generator;
  required_ = other.required_;
  //std::cout << "copy construct" << required()->culling_ratio << std::endl;

}

Population& Population::operator=(Population&& rhs) {
  population = std::move(rhs.population);
  networks = std::move(rhs.networks);
  organisms = std::move(rhs.organisms);
  generator = rhs.generator;
  required_ = rhs.required_;
  //std::cout << "move assign " << required()->culling_ratio << std::endl;
  return *this;
}

Population Population::operator=(const Population& rhs) {
  population = rhs.population;
  networks = rhs.networks;
  organisms = rhs.organisms;
  generator = rhs.generator;
  required_ = rhs.required_;
  //std::cout << "copy assign" << required()->culling_ratio << std::endl;
  return *this;
}

Population Population::Reproduce() {
  // Could consider sorting the most fit first
  // This would then use the champion of each species
  // as the representative for genetic distance.

  // NEAT uses a random member as the representative
  std::random_shuffle(organisms.begin(), organisms.end());

  // speciate
  std::vector<std::vector<Organism>> all_species;
  for(auto& genome : organisms) {
    bool need_new_species = true;
    for(auto& species : all_species) {
      double dist = genome->GeneticDistance(*species.front());
      if(dist < required()->species_delta) {
        species.push_back(genome);
        need_new_species = false;
        break;
      }
    }

    if(need_new_species) {
      all_species.push_back(std::vector<Organism>{genome});
    }
  }


  // If using std::shuffle to randomize representative,
  // must sort species first, to find champion

  for(auto& species : all_species) {
    std::sort(species.begin(), species.end(),
              [](auto& a, auto& b) { return a.fitness > b.fitness; });
  }


  // Note: Destruction of species due to stagnation not currently implemented
  //       Species die off slowly, due to baby-stealing.

  // Find adjusted fitness
  // adj_fitness = fitness/ number_of_genetically_similar_in_species
  for(auto& species : all_species) {
    for(auto& genome : species) {
      int nearby_in_species = 0;
      for(auto& other : species) {
        double dist = genome->GeneticDistance(*other);
        if(dist < required()->species_delta) {
          nearby_in_species++;
        }
      }
      genome.adj_fitness = genome.fitness/nearby_in_species;
    }
  }

  // Determine number of children for each species
  double total_adj_fitness = 0;
  std::vector<double> total_adj_fitness_by_species;
  for(auto& species : all_species) {
    double t = 0;
    for(auto& genome : species) {
      t += genome.adj_fitness;
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


  std::vector<Genome> progeny;
  for (auto i=0u; i < all_species.size(); i++) {
    auto& species = all_species[i];
    auto& num_children = num_children_by_species[i];

    for(int i=0; i<num_children; i++) {
      if(i==0 && species.size() > required()->min_size_for_champion) {
        // Preserve the champion of large species.
        progeny.push_back(*species.front());
      } else {
        // Everyone else can mate

        // if only one organism in the species
        if (species.size() == 1) {
          species.front()->Mutate();
          progeny.push_back(*species.front());
          continue;
        }

        float culling_ratio = required()->culling_ratio;
        // if two organisms in a species and culling ratio is at least half
        // then just take the more fit of the organisms
        if (species.size() == 2 && culling_ratio <= 0.5) {
          species.front()->Mutate();
          progeny.push_back(*species.front());
          continue;
        }

        int idx1 = random()*species.size()*culling_ratio;
        int idx2 = random()*species.size()*culling_ratio;
        while (idx1 == idx2) {
          idx1 = random()*species.size()*culling_ratio;
          idx2 = random()*species.size()*culling_ratio;
        }
        Organism& parent1 = species[idx1];
        Organism& parent2 = species[idx2];
        Genome child;

        // determine relative fitness for mating
        if (parent1.fitness > parent2.fitness) {
          child = parent1->MateWith(*parent2);
        } else if (parent2.fitness > parent1.fitness) {
          child = parent2->MateWith(*parent1);
        } else {
          // break a fitness tie with a check on size
          if (parent1->Size() > parent2->Size()) {
            child = parent1->MateWith(*parent2);
          }
          else { // equal size or parent 2 is larger
            child = parent2->MateWith(*parent1);
          }
        }
        child.Mutate();
        progeny.push_back(child);


      }
    }
  }

  return Population(progeny, get_generator(), required());
}

void Population::build_networks() {
  networks.clear();

  for (auto& genome : population) {
    networks.push_back(NeuralNet(genome));
  }
}
