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
                 std::make_shared<Uniform>(0,1),
                 std::make_shared<Probabilities>());


  std::vector<double> input_vals = {1};
  auto organisms = pop.Evaluate(
    // fitness lambda function
    [](NeuralNet& net, std::vector<double> inputs) {
      auto outputs = net.evaluate(inputs);
      return 1.0;
    }, input_vals);

  pop.Reproduce(organisms);
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
                 std::make_shared<Uniform>(0,1),
                 std::make_shared<Probabilities>());


  double tperformance = 0.0;
  auto nTrials = 10000;
  for (auto i=0u; i < nTrials; i++)
  {


    Timer tbuild([&tperformance](int elapsed) {
        tperformance+=elapsed;
      });

    std::vector<double> input_vals = {1};
    auto organisms = pop.Evaluate(
      // fitness lambda function
      [](NeuralNet& net, std::vector<double> inputs) {
        auto outputs = net.evaluate(inputs);
        return 1.0;
      }, input_vals);

  }
  std::cout << "                Total time to evaluate 100 network population 10000 times: "
            << tperformance/1.0e6 << " ms"
            << std::endl;

}
