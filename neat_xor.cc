#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>

#include "Population.hh"
#include "Timer.hh"

struct XOR_res {
  _float_ x;
  _float_ y;
  _float_ correct;
};

int main() {

  auto seed = Genome::ConnectedSeed(2,1);

  auto prob = std::make_shared<Probabilities>();
  prob->new_connection_is_recurrent = 0;
  prob->keep_non_matching_father_gene = 0;

  Population pop(seed,
                 std::make_shared<RNG_MersenneTwister>(),
                 prob);

  auto max_generations = 1000u;

  std::vector<XOR_res> inputs{
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
  };
  auto shuffled_inputs = inputs;

  std::unique_ptr<ConsecutiveNeuralNet> winner = nullptr;
  unsigned int generation;

  auto show = [&](){
    auto best = pop.BestNet();
    if(!best) { return; }
    std::cout << " ----------- Gen " << generation << " ----------------" << std::endl;
    auto num_species = pop.NumSpecies();
    std::cout << num_species << " species total" << std::endl;
    std::cout << pop.NumViableSpecies() << " viable species" << std::endl;
    //for (auto i=0u; i< std::min(num_species,5u); i++) {
    for (auto i=0u; i< num_species; i++) {
      size_t size = pop.SpeciesSize(i);
      if (size>0) {
        std::cout << "- species #" << i << " size: " << pop.SpeciesSize(i) << std::endl;
      }
    }
    std::cout << "Best (nodes, conn) = (" << best->num_nodes() << ", " << best->num_connections()
              << ")" << std::endl;
    _float_ error = 0;
    for(auto& input : inputs) {
      _float_ val = best->evaluate({input.x, input.y})[0];
      std::cout << input.x << " ^ " << input.y << " = " << val << std::endl;
      error += std::abs(val - input.correct);
    }
    std::cout << "Error: " << error << std::endl;
  };

  auto fitness = [&](ConsecutiveNeuralNet& net) {
    if(net.num_connections() == 0) {
      return 0.0;
    }

    _float_ error = 0;
    for(const auto& input : shuffled_inputs) {
      _float_ val = net.evaluate({input.x, input.y})[0];
      //error += std::abs(val - input.correct);
      error += std::pow(val - input.correct, 2);
    }

    return std::pow(4.0 - error, 2);
    //return 4 - error;
  };


  for (generation = 0u; generation < max_generations; generation++) {
    std::random_shuffle(shuffled_inputs.begin(), shuffled_inputs.end());

    auto next_gen = pop.Reproduce(fitness);

    auto best = pop.BestNet();
    bool have_winner = true;
    for(auto& input : inputs) {
      _float_ val = best->evaluate({input.x, input.y})[0];
      if(std::abs(val - input.correct) >= 0.5) {
        have_winner = false;
        break;
      }
    }

    if(have_winner) {
      winner = std::make_unique<ConsecutiveNeuralNet>(*best);
      break;
    }

    if(generation%10 == 0) {
      show();
    }

    pop = std::move(next_gen);
  }



  show();

  if(winner) {
    std::cout << "Winner found in generation " << generation  << ".\n"
              << *winner << std::endl;
  } else {
    std::cout << "No winner found after " << generation << " generations" << std::endl;
    pop.Evaluate(fitness);
    std::cout << "Best: " << *pop.BestNet() << std::endl;
  }

  return 0;

}
