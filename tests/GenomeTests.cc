#include <gtest/gtest.h>
#include "Genome.hh"
#include "NeuralNet.hh"
#include "Timer.hh"

TEST(Genome,CompareInnovation){
    auto mother = Genome()
        .AddNode(NodeType::Bias)
        .AddNode(NodeType::Input)
        .AddNode(NodeType::Hidden)
        .AddNode(NodeType::Output)
        .AddConnection(0,3,true,1.)
        .AddConnection(1,3,true,1.)
        .AddConnection(1,2,true,1.)
        .AddConnection(2,3,true,1.);
    mother.set_generator(std::make_shared<Uniform>(0,1));

    auto father = mother;
    mother.AddConnection(3,2,true,1.);
    father.AddConnection(0,2,true,1.);

    auto child = mother(father);
}
