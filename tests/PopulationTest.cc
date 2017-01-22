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


  std::vector<double> input_vals = {1};
  pop.Reproduce(
    // fitness lambda function
    [&](NeuralNet& net) {
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
  auto nTrials = 10000u;
  for (auto i=0u; i < nTrials; i++)
  {


    Timer tbuild([&tperformance](int elapsed) {
        tperformance+=elapsed;
      });

    std::vector<double> input_vals = {1};
    pop.Evaluate(
      // fitness lambda function
      [&](NeuralNet& net) {
        auto outputs = net.evaluate(input_vals);
        return 1.0;
      });

  }
  std::cout << "                Total time to evaluate 100 network population 10000 times: "
            << tperformance/1.0e6 << " ms"
            << std::endl;

}
