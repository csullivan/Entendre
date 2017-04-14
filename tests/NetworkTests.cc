#include <gtest/gtest.h>
#include "Genome.hh"
#include "ConsecutiveNeuralNet.hh"
#include "ConcurrentNeuralNet.hh"
#include "ConcurrentGPUNeuralNet.hh"
#include "Timer.hh"

TEST(ConsecutiveNeuralNet,EvaluateNetwork){
    auto sigmoid = [](_float_ val) {return 1/(1 + std::exp(-val));};

    auto genome = Genome()
        .AddNode(NodeType::Bias)
        .AddNode(NodeType::Input)
        .AddNode(NodeType::Sigmoid)
        .AddNode(NodeType::Output)
        .AddConnection(0,3,true,1.)
        .AddConnection(1,3,true,1.)
        .AddConnection(1,2,true,1.)
        .AddConnection(2,3,true,1.);
    auto net = genome.MakeNet<ConsecutiveNeuralNet>();
    net->register_sigmoid(sigmoid);
    auto result = net->evaluate({0.5});

    EXPECT_EQ(result[0],sigmoid(sigmoid(0.5)+1.5));
}

TEST(ConsecutiveNeuralNet,EvaluateRecurrentNetwork){
    auto sigmoid = [](_float_ val) {return 1/(1 + std::exp(-val));};

    auto genome = Genome()
        .AddNode(NodeType::Bias)
        .AddNode(NodeType::Input)
        .AddNode(NodeType::Sigmoid)
        .AddNode(NodeType::Output)
        .AddConnection(0,3,true,1.)
        .AddConnection(1,3,true,1.)
        .AddConnection(1,2,true,1.)
        .AddConnection(2,3,true,1.)
        .AddConnection(3,2,true,1.); // recurrent
    auto net = genome.MakeNet<ConsecutiveNeuralNet>();
    net->register_sigmoid(sigmoid);
    auto result = net->evaluate({0.5});
    EXPECT_EQ(result[0],sigmoid(sigmoid(0.5 + 0)+1.5));
    auto result2 = net->evaluate({0.5});
    EXPECT_EQ(
        result2[0],
        sigmoid(1.5+sigmoid(0.5 + result[0]))
        );

}

TEST(ConsecutiveNeuralNet,EvaluateLargeNetwork){
    auto sigmoid = [](_float_ val) {return 1/(1 + std::exp(-val));};

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
        genome.AddNode(NodeType::Sigmoid);
    }
    for (auto i=0u;i<nOutputs; i++) {
        genome.AddNode(NodeType::Output);
    }


    // Connect every input node to every hidden node
    for (auto i=0u;i<nInputs+1; i++) {
        for (auto j=nInputs+1;j<nHidden+nInputs+1; j++) {
            genome.AddConnection(i,j,true,1.);
        }
    }

    // Connect every hidden node to every output node
    for (auto j=nInputs+1;j<nHidden+nInputs+1; j++) {
        for (auto k=nHidden+nInputs+1; k<nTotal; k++) {
            genome.AddConnection(j,k,true,1.);
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
        auto net = genome.MakeNet<ConsecutiveNeuralNet>();
    }
    std::cout << "                Average time to construct network: "
              << tperformance/nTrials/1.0e6 << " ms"
              << std::endl;
    tperformance = 0.0;


    // Time the network evaluation
    for (auto i=0u; i < nTrials; i++)
    {
        auto net = genome.MakeNet<ConsecutiveNeuralNet>();
        net->register_sigmoid(sigmoid);

        // First evaluation includes network sorting (construction)
        // so we will time the evaluations thereafter
        net->evaluate({0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5});

        Timer teval([&tperformance](int elapsed) {
                tperformance+=elapsed;
            });
        auto result2 = net->evaluate({0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5});


    }
    std::cout << "                Average time to evaluate network: "
              << tperformance/nTrials/1.0e6 << " ms"
              << std::endl;


}

TEST(ConcurrentNeuralNet,EvaluateLargeNetwork){
    auto sigmoid = [](_float_ val) {return 1/(1 + std::exp(-val));};

    auto genome = Genome().
        AddNode(NodeType::Bias);

    auto nInputs = 10u;
    auto nHidden = 2000u;
    auto nOutputs = 10u;

    auto nTotal = nInputs + nHidden + nOutputs + 1;

    for (auto i=0u;i<nInputs; i++) {
        genome.AddNode(NodeType::Input);
    }
    for (auto i=0u;i<nHidden; i++) {
        genome.AddNode(NodeType::Sigmoid);
    }
    for (auto i=0u;i<nOutputs; i++) {
        genome.AddNode(NodeType::Output);
    }


    // Connect every input node to every hidden node
    for (auto i=0u;i<nInputs+1; i++) {
        for (auto j=nInputs+1;j<nHidden+nInputs+1; j++) {
            genome.AddConnection(i,j,true,1.);
        }
    }

    // Connect every hidden node to every output node
    for (auto j=nInputs+1;j<nHidden+nInputs+1; j++) {
        for (auto k=nHidden+nInputs+1; k<nTotal; k++) {
            genome.AddConnection(j,k,true,1.);
        }
    }

    auto nTrials = 1u;
    double tperformance = 0.0;



    // Time the network construction
    for (auto i=0u; i < nTrials; i++)
    {
        Timer tbuild([&tperformance](int elapsed) {
                tperformance+=elapsed;
        });
        auto net = genome.MakeNet<ConsecutiveNeuralNet>();
    }
    // std::cout << "                Average time to construct network: "
    //           << tperformance/nTrials/1.0e6 << " ms"
    //           << std::endl;
    tperformance = 0.0;


    // Time the network evaluation
    for (auto i=0u; i < nTrials; i++)
    {
        auto net = genome.MakeNet<ConcurrentNeuralNet>();
        net->register_sigmoid(sigmoid);

        // First evaluation includes network sorting (construction)
        // so we will time the evaluations thereafter
        net->evaluate({1.0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5});

        Timer teval([&tperformance](int elapsed) {
                tperformance+=elapsed;
            });
        auto result2 = net->evaluate({1.0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5});


    }
    std::cout << "                Average time to evaluate network: "
              << tperformance/nTrials/1.0e6 << " ms"
              << std::endl;


}

template<typename Derived, typename Base, typename Del>
std::unique_ptr<Derived, Del>
static_unique_ptr_cast( std::unique_ptr<Base, Del>&& p )
{
        auto d = static_cast<Derived *>(p.release());
        return std::unique_ptr<Derived, Del>(d, std::move(p.get_deleter()));
}

TEST(ConcurrentGPUNeuralNet,EvaluateLargeNetwork){
    auto sigmoid = [](_float_ val) {return 1/(1 + std::exp(-val));};

    auto genome = Genome().
        AddNode(NodeType::Bias);

    auto nInputs = 10u;
    auto nHidden = 2000u;
    auto nOutputs = 10u;

    auto nTotal = nInputs + nHidden + nOutputs + 1;

    for (auto i=0u;i<nInputs; i++) {
        genome.AddNode(NodeType::Input);
    }
    for (auto i=0u;i<nHidden; i++) {
        genome.AddNode(NodeType::Sigmoid);
    }
    for (auto i=0u;i<nOutputs; i++) {
        genome.AddNode(NodeType::Output);
    }


    // Connect every input node to every hidden node
    for (auto i=0u;i<nInputs+1; i++) {
        for (auto j=nInputs+1;j<nHidden+nInputs+1; j++) {
            genome.AddConnection(i,j,true,1.);
        }
    }

    // Connect every hidden node to every output node
    for (auto j=nInputs+1;j<nHidden+nInputs+1; j++) {
        for (auto k=nHidden+nInputs+1; k<nTotal; k++) {
            genome.AddConnection(j,k,true,1.);
        }
    }

    auto nTrials = 1u;
    double tperformance = 0.0;



    // Time the network construction
    for (auto i=0u; i < nTrials; i++)
    {
        Timer tbuild([&tperformance](int elapsed) {
                tperformance+=elapsed;
        });
        auto net = genome.MakeNet<ConsecutiveNeuralNet>();
    }
    // std::cout << "                Average time to construct network: "
    //           << tperformance/nTrials/1.0e6 << " ms"
    //           << std::endl;
    tperformance = 0.0;


    // Time the network evaluation
    for (auto i=0u; i < nTrials; i++)
    {
        auto net = genome.MakeNet<ConcurrentGPUNeuralNet>();
        net->register_sigmoid(sigmoid);

        // First evaluation includes network sorting (construction)
        // so we will time the evaluations thereafter
        net->evaluate({1.0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5});

        Timer teval([&tperformance](int elapsed) {
                tperformance+=elapsed;
            });
        auto result2 = net->evaluate({1.0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5});


    }
    std::cout << "                Average time to evaluate network: "
              << tperformance/nTrials/1.0e6 << " ms"
              << std::endl;


}

TEST(ConcurrentGPUNeuralNet,CompareEvaluation) {
  auto genome = Genome()
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Output)
    .AddNode(NodeType::Sigmoid)
    .AddNode(NodeType::Sigmoid)
    .AddConnection(0,4,true,1.)
    .AddConnection(1,3,true,1.)
    .AddConnection(4,4,true,1.) // self-recurrent
    .AddConnection(4,2,true,1.)
    .AddConnection(3,2,true,1.)
    .AddConnection(2,3,true,1.); // recurrent
  auto consecutive = genome.MakeNet<ConsecutiveNeuralNet>();
  auto concurrent =  genome.MakeNet<ConcurrentNeuralNet>();
  auto concurrentgpu = genome.MakeNet<ConcurrentGPUNeuralNet>();



  auto result = consecutive->evaluate({0.5,1.5});
  auto result2 = concurrent->evaluate({0.5,1.5});
  auto result3 = concurrentgpu->evaluate({0.5,1.5});
  EXPECT_FLOAT_EQ(result[0],result2[0]);
  EXPECT_FLOAT_EQ(result[0],result3[0]);

}
