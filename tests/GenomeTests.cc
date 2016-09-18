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
    //Probabilities NEAT = {0.5, 1.0, 0.0, 0.9, 0.0, 0.0, 0.1, 4.0, 1.0, 1.0, 3.0};
    //mother.required(std::make_shared<Probabilities>(NEAT));
    mother.required(std::make_shared<Probabilities>());
    auto father = mother;

    mother.AddConnection(3,2,true,1.);
    father.AddConnection(0,2,true,1.);

    EXPECT_FLOAT_EQ(mother.GeneticDistance(father),1.*2/5+0);

    auto child = mother(father);

    EXPECT_FLOAT_EQ(mother.GeneticDistance(child),0);
    EXPECT_FLOAT_EQ(father.GeneticDistance(child),1.*2/5+0);
    EXPECT_FLOAT_EQ(child.GeneticDistance(mother),0);
    EXPECT_FLOAT_EQ(child.GeneticDistance(father),1.*2/5+0);

}
