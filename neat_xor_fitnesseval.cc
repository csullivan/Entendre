#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>

#include "Population.hh"
#include "Timer.hh"

struct EachAnswer {
  EachAnswer(_float_ a, _float_ b)
    : a(a), b(b), correct_result(int(a)^int(b)),
      nn_result(std::numeric_limits<_float_>::quiet_NaN()) { }


  _float_ a;
  _float_ b;
  _float_ correct_result;
  _float_ nn_result;
};


class XorFitness : public FitnessEvaluator {
public:
  XorFitness()
    : all_answers{
    EachAnswer(0,0),
      EachAnswer(0,1),
      EachAnswer(1,0),
      EachAnswer(1,1)} {
    //std::random_shuffle(all_answers.begin(), all_answers.end());
  }

  void step(NetProxy& proxy) {

    for(auto& ans : all_answers) {
      if(std::isnan(ans.nn_result)) {
        proxy.request_calc({ans.a, ans.b},
                           [&](const auto& nn_output) {
                             ans.nn_result = nn_output[0];
                           });
        return;
      }
    }

    if(proxy.num_connections() == 0) {
      proxy.set_fitness_value(0.0);
      return;
    }

    double error = 0;
    for(auto& ans : all_answers) {
      //error += std::abs(ans.nn_result - ans.correct_result);
      error += std::pow(ans.nn_result - ans.correct_result,2);
    }
    double fitness = std::pow(4.0 - error, 2); //16 - std::pow(cum_sum, 2);
    proxy.set_fitness_value(fitness);
  }

private:


  std::array<EachAnswer, 4> all_answers;
};


std::array<EachAnswer, 4> inputs = {EachAnswer(0,0),
                                    EachAnswer(0,1),
                                    EachAnswer(1,0),
                                    EachAnswer(1,1)};


int main() {

  auto seed = Genome::ConnectedSeed(2,1);

  auto prob = std::make_shared<Probabilities>();
  prob->new_connection_is_recurrent = 0;
  prob->keep_non_matching_father_gene = 0;

  Population pop(seed,
                 std::make_shared<RNG_MersenneTwister>(12),
                 prob);

  //pop.SetNetType<ConsecutiveNeuralNet>();
  pop.SetNetType<ConcurrentNeuralNet>();
  pop.EnableCompositeNet(/*hetero_inputs = */false);

  auto max_generations = 1000u;

  std::unique_ptr<NeuralNet> winner = nullptr;
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
      _float_ val = best->evaluate({input.a, input.b})[0];
      std::cout << input.a << " ^ " << input.b << " = " << val << std::endl;
      error += std::abs(val - input.correct_result);
    }
    std::cout << "Error: " << error << std::endl;
  };


  std::function<std::unique_ptr<FitnessEvaluator>(void)> fitness_factory = [](){return std::make_unique<XorFitness>();};

  for (generation = 0u; generation < max_generations; generation++) {

    auto next_gen = pop.Reproduce(fitness_factory);

    auto best = pop.BestNet();
    bool have_winner = true;
    for(auto& input : inputs) {
      _float_ val = best->evaluate({input.a, input.b})[0];
      if(std::abs(val - input.correct_result) >= 0.5) {
        have_winner = false;
        break;
      }
    }

    if(have_winner) {
      winner = best->clone();
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
    pop.Evaluate(fitness_factory);
    std::cout << "Best: " << *pop.BestNet() << std::endl;
  }

  return 0;

}
