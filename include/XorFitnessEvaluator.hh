#pragma once
#include <array>

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
