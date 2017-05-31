#include <cmath>

#include <gtest/gtest.h>

#include "Genome.hh"
#include "ConsecutiveNeuralNet.hh"


// Using a macro instead of a function because it results in better
// print messages from gtest.

#define test_single_func(func, input_val, output_val)   \
  {                                                     \
    auto genome = Genome()                              \
      .AddNode(NodeType::Input)                         \
      .AddNode(NodeType::Output, func)                  \
      .AddConnection(0, 1, true, 1.0);                  \
    auto net = genome.MakeNet<ConsecutiveNeuralNet>();  \
    auto res = net->evaluate({input_val}).at(0);        \
    EXPECT_NEAR(res, output_val, 1e-5);                 \
  }

TEST(CPPN, EvaluateAllActivationFunctions) {
  test_single_func(ActivationFunction::Sigmoid, 1, 1/(1 + std::exp(-1)));
  test_single_func(ActivationFunction::Identity, 1, 1);
  test_single_func(ActivationFunction::Identity, -1, -1);
  test_single_func(ActivationFunction::Tanh, 1, std::tanh(1));
  test_single_func(ActivationFunction::Relu, -1, 0);
  test_single_func(ActivationFunction::Relu, 1, 1);
  test_single_func(ActivationFunction::Gaussian, 0, 1);
  test_single_func(ActivationFunction::Gaussian, 1, std::exp(-1/2.0));
  test_single_func(ActivationFunction::Gaussian, -1, std::exp(-1/2.0));
  test_single_func(ActivationFunction::Sin, 0, 0);
  test_single_func(ActivationFunction::Sin, 1, std::sin(1));
  test_single_func(ActivationFunction::Cos, 0, 1);
  test_single_func(ActivationFunction::Cos, 1, std::cos(1));
  test_single_func(ActivationFunction::Abs, 1, 1);
  test_single_func(ActivationFunction::Abs, -1, 1);
  test_single_func(ActivationFunction::Square, 1, 1);
  test_single_func(ActivationFunction::Square, 2, 4);
  test_single_func(ActivationFunction::Square, -2, 4);
}

TEST(CPPN, MutateOdds) {
  Genome seed = Genome()
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Output)
    .AddConnection(0, 1, true, 1.0);

  seed.set_generator(std::make_shared<RNG_MersenneTwister>());
  auto prob = std::make_shared<Probabilities>();
  seed.required(prob);

  auto reset_prob = [&]() {
    prob->use_compositional_pattern_producing_networks = true;
    for(auto& obj : prob->cppn_odds) {
      obj.second = 0;
    }
  };


  // One explicitly specified, to make sure that everything is fine with the odds map.
  {
    auto copy = Genome(seed);
    reset_prob();
    prob->cppn_odds[ActivationFunction::Tanh] = 1;
    copy.MutateNode();
    auto net = copy.MakeNet<ConsecutiveNeuralNet>();
    ASSERT_EQ(net->num_nodes(), 3U);
    EXPECT_EQ(net->get_activation_func(2), ActivationFunction::Tanh);
  }

  // Check everything in the odds map.
  for(auto& odds : prob->cppn_odds) {
    auto copy = Genome(seed);
    reset_prob();
    prob->cppn_odds[odds.first] = 1;
    copy.MutateNode();
    auto net = copy.MakeNet<ConsecutiveNeuralNet>();
    ASSERT_EQ(net->num_nodes(), 3U);
    EXPECT_EQ(net->get_activation_func(2), odds.first);
  }
}
