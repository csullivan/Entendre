#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>

#include "Population.hh"
#include "Timer.hh"

struct XOR_res {
  double x;
  double y;
  double correct;
};

int main() {

  auto seed = Genome()
    .AddNode(NodeType::Bias)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Output)
    .AddConnection(0,3,true,1.)
    .AddConnection(1,3,true,1.)
    .AddConnection(2,3,true,1.);

  auto prob = std::make_shared<Probabilities>();
  prob->add_recurrent = 0;
  prob->single_lesser = 0;
  //prob->population_size = 150;

  Population pop(seed,
                 std::make_shared<Uniform>(0,1),
                 prob);


  auto max_generations = 1000u;

  std::vector<XOR_res> inputs{
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
  };
  auto shuffled_inputs = inputs;

  std::unique_ptr<NeuralNet> winner = nullptr;
  unsigned int generation;

  auto show = [&](){
    auto best = pop.BestNet();
    if(!best) { return; }
    std::cout << " ----------- Gen " << generation << " ----------------" << std::endl;
    std::cout << pop.NumSpecies() << " species total" << std::endl;
    std::cout << "Best (nodes, conn) = (" << best->num_nodes() << ", " << best->num_connections()
              << ")" << std::endl;
    double error = 0;
    for(auto& input : inputs) {
      double val = best->evaluate({input.x, input.y})[0];
      std::cout << input.x << " ^ " << input.y << " = " << val << std::endl;;
      error += std::abs(val - input.correct);
    }
    std::cout << "Error: " << error << std::endl;
  };

  auto fitness = [&](NeuralNet& net) {
    if(net.num_connections() == 0) {
      return 0.0;
    }

    double error = 0;
    for(const auto& input : shuffled_inputs) {
      double val = net.evaluate({input.x, input.y})[0];
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
      double val = best->evaluate({input.x, input.y})[0];
      if(std::abs(val - input.correct) >= 0.5) {
        have_winner = false;
        break;
      }
    }

    if(have_winner) {
      winner = std::make_unique<NeuralNet>(*best);
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
