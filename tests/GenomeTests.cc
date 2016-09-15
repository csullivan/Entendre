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
        .AddGene(0,3,ConnectionType::Normal,true,1.)
        .AddGene(1,3,ConnectionType::Normal,true,1.)
        .AddGene(1,2,ConnectionType::Normal,true,1.)
        .AddGene(2,3,ConnectionType::Normal,true,1.);
    mother.set_generator(std::make_shared<Uniform>(0,1));

    auto father = mother;
    mother.AddGene(3,2,ConnectionType::Recurrent,true,1.);
    father.AddGene(0,2,ConnectionType::Normal,true,1.);

    auto child = mother(father);
}
