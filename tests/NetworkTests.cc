#include <gtest/gtest.h>
#include "Genome.hh"
#include "NeuralNet.hh"
#include "Timer.hh"

TEST(NeuralNet,EvaluateNetwork){
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

TEST(NeuralNet,EvaluateRecurrentNetwork){
    auto sigmoid = [](double val) {return 1/(1 + std::exp(-val));};

    auto genome = Genome()
        .AddNode(NodeType::Bias)
        .AddNode(NodeType::Input)
        .AddNode(NodeType::Hidden)
        .AddNode(NodeType::Output)
        .AddGene(0,3,ConnectionType::Normal,true,1.)
        .AddGene(1,3,ConnectionType::Normal,true,1.)
        .AddGene(1,2,ConnectionType::Normal,true,1.)
        .AddGene(2,3,ConnectionType::Normal,true,1.)
        .AddGene(3,2,ConnectionType::Recurrent,true,1.);
    auto net = NeuralNet(genome);
    net.register_sigmoid(sigmoid);
    auto result = net.evaluate({0.5});
    EXPECT_EQ(result[0],sigmoid(sigmoid(0.5)+1.5));
    auto result2 = net.evaluate({0.5});
    EXPECT_EQ(
        result2[0],
        sigmoid(1.5+sigmoid(0.5 + result[0]))
        );

}

TEST(NeuralNet,EvaluateLargeNetwork){
    auto sigmoid = [](double val) {return 1/(1 + std::exp(-val));};

    auto genome = Genome().
        AddNode(NodeType::Bias);

    auto nInputs = 10u;
    auto nHidden = 20u;
    auto nOutputs = 10u;

    auto nTotal = nInputs + nHidden + nOutputs + 1;

    for (auto i=0u;i<nInputs; i++) {
        genome.AddNode(NodeType::Input);
    }
    for (auto i=0u;i<nHidden; i++) {
        genome.AddNode(NodeType::Hidden);
    }
    for (auto i=0u;i<nOutputs; i++) {
        genome.AddNode(NodeType::Output);
    }


    // Connect every input node to every hidden node
    for (auto i=0u;i<nInputs+1; i++) {
        for (auto j=nInputs+1;j<nHidden+nInputs+1; j++) {
            genome.AddGene(i,j,ConnectionType::Normal,true,1.);
        }
    }

    // Connect every hidden node to every output node
    for (auto j=nInputs+1;j<nHidden+nInputs+1; j++) {
        for (auto k=nHidden+nInputs+1; k<nTotal; k++) {
            genome.AddGene(j,k,ConnectionType::Normal,true,1.);
        }
    }

    auto nTrials = 100u;
    double tperformance = 0.0;



    // Time the network construction
    for (auto i=0u; i < nTrials; i++)
    {
        Timer tbuild([&tperformance](int elapsed) {
                tperformance+=elapsed;
        });
        auto net = NeuralNet(genome);
    }
    std::cout << std:: endl
              << "Average time to construct network: "
              << tperformance/nTrials/1.0e6 << " ms\n"
              << std::endl;
    tperformance = 0.0;


    // Time the network evaluation
    for (auto i=0u; i < nTrials; i++)
    {
        auto net = NeuralNet(genome);
        net.register_sigmoid(sigmoid);

        Timer teval([&tperformance](int elapsed) {
                tperformance+=elapsed;
        });

        auto result = net.evaluate({0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5});
    }
    std::cout << std:: endl
              << "Average time to evaluate network: "
              << tperformance/nTrials/1.0e6 << " ms\n"
              << std::endl;


}
