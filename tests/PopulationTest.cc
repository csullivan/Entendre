#include <gtest/gtest.h>
#include "Population.hh"
#include "Timer.hh"

TEST(Population,Construct){
  auto adam = Genome()
    .AddNode(NodeType::Bias)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Hidden)
    .AddNode(NodeType::Output)
    .AddConnection(0,3,true,1.)
    .AddConnection(1,3,true,1.)
    .AddConnection(1,2,true,1.)
    .AddConnection(2,3,true,1.);

  Population pop(adam,
                 std::make_shared<RNG_MersenneTwister>(),
                 std::make_shared<Probabilities>());


  std::vector<float> input_vals = {1};
  pop.Reproduce(
    // fitness lambda function
    [&](ConsecutiveNeuralNet& net) {
      auto outputs = net.evaluate(input_vals);
      return 1.0;
    });
}


TEST(Population,EvaluationTimer){
  auto adam = Genome()
    .AddNode(NodeType::Bias)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Hidden)
    .AddNode(NodeType::Output)
    .AddConnection(0,3,true,1.)
    .AddConnection(1,3,true,1.)
    .AddConnection(1,2,true,1.)
    .AddConnection(2,3,true,1.);


  Population pop(adam,
                 std::make_shared<RNG_MersenneTwister>(),
                 std::make_shared<Probabilities>());


  double tperformance = 0.0;
  auto nTrials = 1000u;
  for (auto i=0u; i < nTrials; i++)
  {


    Timer tbuild([&tperformance](int elapsed) {
        tperformance+=elapsed;
      });

    std::vector<float> input_vals = {1};
    pop.Evaluate(
      // fitness lambda function
      [&](ConsecutiveNeuralNet& net) {
        auto outputs = net.evaluate(input_vals);
        return 1.0;
      });

  }
  std::cout << "                Total time to evaluate 100 network population 1000 times: "
            << tperformance/1.0e6 << " ms"
            << std::endl;

}

TEST(Population, PruneEmptySpecies){
  auto rng = std::make_shared<RNG_MersenneTwister>();
  auto prob = std::make_shared<Probabilities>();
  // Prevent all mutations during this test.
  prob->mutation_prob_adjust_weights = 0;
  prob->mutation_prob_add_connection = 0;
  prob->mutation_prob_add_node = 0;
  prob->mutation_prob_reenable_connection = 0;
  prob->mutation_prob_toggle_connection = 0;

  auto adam = Genome::ConnectedSeed(1,1);
  adam.required(prob);
  adam.set_generator(rng);

  Species full;
  full.organisms.push_back(adam);
  full.id = 5;
  full.representative = adam;
  full.age = 0;
  full.best_fitness = 0;

  Species empty;
  empty.id = 7;
  empty.representative = adam;
  empty.age = 0;
  empty.best_fitness = 0;

  Population gen1({full, empty}, rng, prob);


  {
    prob->keep_empty_species = false;
    Population gen2 = gen1.Reproduce([](ConsecutiveNeuralNet&) { return 1.0; });


    EXPECT_EQ(gen1.GetSpecies().size(), 2u);
    EXPECT_EQ(gen2.GetSpecies().size(), 1u);
    EXPECT_EQ(gen2.GetSpecies()[0].id, 5u);
  }

  {
    prob->keep_empty_species = true;
    Population gen2 = gen1.Reproduce([](ConsecutiveNeuralNet&) { return 1.0; });

    EXPECT_EQ(gen1.GetSpecies().size(), 2u);
    EXPECT_EQ(gen2.GetSpecies().size(), 2u);
    EXPECT_EQ(gen2.GetSpecies()[0].id, 5u);
    EXPECT_EQ(gen2.GetSpecies()[1].id, 7u);
  }
}
