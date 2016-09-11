#include <gtest/gtest.h>
#include "Genome.hh"
#include "NeuralNet.hh"

TEST(Genome,EvaluateNetwork){
    auto sigmoid = [](double val) {return 1/(1 + std::exp(-val));};

    auto genome = Genome()
        .AddNode(NodeType::Bias)
        .AddNode(NodeType::Input)
        .AddNode(NodeType::Hidden)
        .AddNode(NodeType::Output)
        .AddGene(0,3,ConnectionType::Normal,true,1.)
        .AddGene(1,3,ConnectionType::Normal,true,1.)
        .AddGene(1,2,ConnectionType::Normal,true,1.)
        .AddGene(2,3,ConnectionType::Normal,true,1.);
    auto net = NeuralNet(genome);
    net.register_sigmoid(sigmoid);
    auto result = net.evaluate({0.5});

    EXPECT_EQ(result[0],sigmoid(sigmoid(0.5)+1.5));
}
