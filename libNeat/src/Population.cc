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
  SetNetType<ConsecutiveNeuralNet>();

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
        spec.organisms.emplace_back(genome,converter);
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
      new_spec.organisms.emplace_back(genome,converter);

      species.push_back(new_spec);
    }
  }
}

void Population::CalculateAdjustedFitness() {
  // adj_fitness = fitness / number_in_species
  for(auto& spec : species) {
    auto num_in_species = spec.organisms.size();
    for(auto& org : spec.organisms) {
      org.adj_fitness = org.fitness / num_in_species;
    }
  }


  // adj_fitness = fitness/ number_of_genetically_similar_in_species
  // for(auto& spec : species) {
  //   for(auto& org : spec.organisms) {
  //     int nearby_in_species = 0;

  //     for(auto& other : spec.organisms) {
  //       double dist = org.genome.GeneticDistance(other.genome);
  //       if(dist < required()->genetic_distance_species_threshold) {
  //         nearby_in_species++;
  //       }
  //     }

  //     org.adj_fitness = org.fitness/nearby_in_species;
  //   }
  // }
}

Population Population::Reproduce() {
  auto next_gen_species = MakeNextGenerationSpecies();
  auto next_gen_genomes = MakeNextGenerationGenomes();

  Speciate(next_gen_species, next_gen_genomes);

  Population pop(next_gen_species, get_generator(), required());
  pop.converter = converter;

  return pop;
}

std::vector<Species> Population::MakeNextGenerationSpecies() {
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
      if((champion.fitness - spec.best_fitness)/spec.best_fitness > required()->necessary_species_improvement) {
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

  return next_gen_species;
}





void Population::DistributeChildrenByRank(std::vector<unsigned int>& number_of_children) const {
  std::vector<std::pair<unsigned int, _float_>> organism_fitnesses;
  for (auto i=0u; i<species.size(); i++) {
    auto& spec = species[i];
    for (auto& org : spec.organisms) {
      organism_fitnesses.push_back({i,org.fitness});
    }
  }
  std::sort(organism_fitnesses.begin(),organism_fitnesses.end(),
            [](auto& a, auto& b){ return a.second > b.second; });

  // total number of organisms in top x percentile
  auto num_organisms_in_percentile =
    std::round(required()->species_survival_percentile*organism_fitnesses.size());

  assert(num_organisms_in_percentile <= organism_fitnesses.size());

  // count number of organisms in top x percentile for each species
  std::vector<float> num_organisms_in_percentile_by_species(species.size(),0);
  for (auto i=0u; i<num_organisms_in_percentile; i++) {
    auto i_species = organism_fitnesses[i].first;
    num_organisms_in_percentile_by_species[i_species]++;
  }


  // add children based on the relative performance of each species
  std::vector<float> ratio_of_org_in_percentile_by_species(species.size(),0);
  float total_org_in_percentile_ratio = 0.;
  for (auto i=0u; i<number_of_children.size(); i++) {
    auto species_size = species[i].organisms.size();
    float ratio = (species_size > 0) ? num_organisms_in_percentile_by_species[i]/species_size : 0;
    ratio_of_org_in_percentile_by_species[i] = ratio;
    total_org_in_percentile_ratio += ratio;
  }

  for (auto i=0u; i<number_of_children.size(); i++) {
    number_of_children[i] += ratio_of_org_in_percentile_by_species[i]/total_org_in_percentile_ratio*required()->population_size;
  }
}

void Population::DistributeNurseryChildren(std::vector<unsigned int>& number_of_children) const {

  // build list of nursery species
  std::vector<unsigned int> nursery;
  for (auto i=0u; i< species.size(); i++) {
    if (species[i].age < required()->nursery_age) {
      nursery.push_back(i);
    }
  }

  if (!required()->fixed_nursery_size) { // each nursery species gets a fixed number of children
    for (auto& i : nursery) {
      number_of_children[i] += required()->number_of_children_given_in_nursery;
    }

  } else { // each nursery species gets children based on the amount of absolute fitness it has

    // Determine total adjusted fitness for each species, and for the
    // entire population.

    double total_adj_fitness = 0;
    std::vector<double> total_adj_fitness_by_species;
    for(auto& i : nursery) {
      auto& spec = species[i];
      double species_total_adj_fitness = 0;

      for(auto& org : spec.organisms) {
        species_total_adj_fitness += org.adj_fitness;
      }

      // NOTICE: stale species are no longer implemented
      // bool is_stale = spec.age_since_last_improvement >= required()->stale_species_num_generations;
      // if(is_stale) {
      //   species_total_adj_fitness *= required()->stale_species_penalty;
      // }

      total_adj_fitness += species_total_adj_fitness;
      total_adj_fitness_by_species.push_back(species_total_adj_fitness);
    }

    // Determine number of children for each species.
    double children_per_adj_fitness = required()->number_of_children_given_in_nursery / total_adj_fitness;
    for(auto i=0u,j=0u; i<nursery.size(); i++) {
      double spec_total_adj_fitness = total_adj_fitness_by_species[j++];
      // Rounding differences here can cause slightly more or slightly
      // fewer genomes to be created than population_size. This is
      // probably ok, but should be studied.
      unsigned int num_children = std::round(children_per_adj_fitness *
                                             spec_total_adj_fitness);

      number_of_children[nursery[i]] += num_children;
    }
  }


}

std::vector<Genome> Population::MakeNextGenerationGenomes() {

  std::vector<unsigned int> num_children_by_species(species.size(),0);
  DistributeNurseryChildren(num_children_by_species);
  DistributeChildrenByRank(num_children_by_species);

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

  return progeny;
}



NeuralNet* Population::BestNet() {
  NeuralNet* output = nullptr;
  double best_fitness = -std::numeric_limits<double>::max();
  for(auto& spec : species) {
    for(auto& org : spec.organisms) {
      if(org.fitness > best_fitness) {
        output = org.network();
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
