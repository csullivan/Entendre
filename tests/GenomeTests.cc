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

    EXPECT_FLOAT_EQ(mother.GeneticDistance(father),1.*2+0);

    auto child = mother.MateWith(father);

    EXPECT_FLOAT_EQ(mother.GeneticDistance(child),0);
    EXPECT_FLOAT_EQ(father.GeneticDistance(child),1.*2+0);
    EXPECT_FLOAT_EQ(child.GeneticDistance(mother),0);
    EXPECT_FLOAT_EQ(child.GeneticDistance(father),1.*2+0);

}

TEST(Genome,MutateConnection) {
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
    mother.required(std::make_shared<Probabilities>());

    //mother.PrintInnovations();
    mother.MutateConnection();
    mother.MutateNode();
    //mother.PrintInnovations();
}

TEST(Genome, GeneticDistanceAfterMutation) {
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
    mother.required(std::make_shared<Probabilities>());
    auto father = mother;

    father.MutateConnection();
    father.MutateNode();
    //std::cout << mother.GeneticDistance(father) << std::endl;

    auto child = mother.MateWith(father);
    EXPECT_FLOAT_EQ(mother.GeneticDistance(child),0.0);
    EXPECT_FLOAT_EQ(mother.GeneticDistance(father), child.GeneticDistance(father));
}

TEST(Genome, ManyMutations) {
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
    mother.required(std::make_shared<Probabilities>());
    auto father = mother;


    for(auto i=0u; i<30; i++) {
        mother.Mutate();
        father.Mutate();
    }
    mother.PrintInnovations();
    father.PrintInnovations();

    std::cout << "                Genetic Distance after 30 generations: " << mother.GeneticDistance(father) << std::endl << std::endl;

}

TEST(Genome, CrossoverAfterManyMutationsNonNEAT) {
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
    Probabilities sNEAT = { 100,5,0.5,0.5,1.0,1.0,0.8,0.9,0.05,0.03,0.25,0.8,0.05,0.05,0.0,0.1,4.0,1.0,1.0,3.0};
    mother.required(std::make_shared<Probabilities>(sNEAT));
    auto father = mother;


    for(auto i=0u; i<5; i++) {
        mother.Mutate();
        father.Mutate();
    }
    mother.PrintInnovations();
    father.PrintInnovations();

    std::cout << "                Genetic Distance after 5 generations: " << mother.GeneticDistance(father) << std::endl << std::endl;

    auto child = mother.MateWith(father);

    child.PrintInnovations();
    std::cout << "                Genetic Distance of child with parents: " << mother.GeneticDistance(child) << " " << father.GeneticDistance(child) << std::endl << std::endl;
}

TEST(Genome, PreserveInnovations) {
  auto parent = Genome()
    .AddNode(NodeType::Bias)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Hidden)
    .AddNode(NodeType::Output)
    .AddConnection(0,3,true,1.)
    .AddConnection(1,3,true,1.)
    .AddConnection(1,2,true,1.)
    .AddConnection(2,3,true,1.);

  EXPECT_TRUE(parent.IsStructurallyEqual(parent));

  parent.set_generator(std::make_shared<Uniform>(0,1));
  auto prob = std::make_shared<Probabilities>();
  prob->mutate_link = 0;
  prob->mutate_node = 0;
  prob->mutate_reenable = 0;
  prob->toggle_status = 0;

  parent.required(prob);

  auto child = parent.MateWith(parent);

  EXPECT_TRUE(parent.IsStructurallyEqual(child));
}

TEST(Genome, ConnectionOrder) {
  auto mother = Genome()
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Output)
    .AddConnection(0, 2, true, 1.0)
    .AddConnection(1, 2, true, 1.0);

  // Connections added in different order
  auto father = Genome()
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Output)
    .AddConnection(1, 2, true, 1.0)
    .AddConnection(0, 2, true, 1.0);

  auto rand = std::make_shared<Uniform>(0,1);
  mother.set_generator(rand);
  father.set_generator(rand);

  auto prob = std::make_shared<Probabilities>();
  prob->match = 0.5;
  prob->single_greater = 1.0;
  prob->single_lesser = 1.0;
  prob->mutate_weights = 0;
  prob->mutate_link = 0;
  prob->mutate_node = 0;
  prob->toggle_status = 0;

  mother.required(prob);
  father.required(prob);

  auto child = mother.MateWith(father);
  child.AssertNoDuplicateConnections();
}
