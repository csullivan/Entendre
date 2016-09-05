#include <gtest/gtest.h>
#include "Node.hh"
#include "Genome.hh"
#include "Gene.hh"

TEST(Genome,Construct){
  auto g =
    make_genome(0)
    .AddNode(Node::Type::Sensor,Node::Function::Bias)
    .AddNode(Node::Type::Sensor,Node::Function::Input)
    .AddNode(Node::Type::Sensor,Node::Function::Input)
    .AddNode(Node::Type::Neuron,Node::Function::Output)
    .AddGene(0,3,0.0,Gene::Status::Enabled,0) // in, out, weight, status, innovation
    .AddGene(1,3,0.0,Gene::Status::Enabled,0)
    .AddGene(2,3,0.0,Gene::Status::Enabled,0);
}
