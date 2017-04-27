#include <memory>
#include <gtest/gtest.h>
#include "Genome.hh"
#include "ConsecutiveNeuralNet.hh"
#include "ConcurrentNeuralNet.hh"
#include "ConcurrentGPUNeuralNet.hh"
#include "CompositeNet.hh"
#include "Timer.hh"

template <typename To, typename From, typename Deleter>
std::unique_ptr<To, Deleter> dynamic_unique_cast(std::unique_ptr<From, Deleter>&& p) {
  if (To* cast = dynamic_cast<To*>(p.get()))
  {
    std::unique_ptr<To, Deleter> result(cast, std::move(p.get_deleter()));
    p.release();
    return result;
  }
  return std::unique_ptr<To, Deleter>(nullptr); // or throw std::bad_cast() if you prefer
}



TEST(ConsecutiveNeuralNet,EvaluateNetwork){
  auto sigmoid = [](_float_ val) {return 1/(1 + std::exp(-val));};

  auto genome = Genome()
    .AddNode(NodeType::Bias)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Hidden)
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
    .AddNode(NodeType::Hidden)
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
    genome.AddNode(NodeType::Hidden);
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
  auto nHidden = 200u;
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
  auto nHidden = 200u;
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

TEST(ConcurrentGPUNeuralNet,CompareEvaluation) {
  auto genome = Genome()
    .AddNode(NodeType::Bias)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Output)
    .AddNode(NodeType::Hidden)
    .AddNode(NodeType::Hidden)
    .AddConnection(0,3,true,1.)
    .AddConnection(1,5,true,1.)
    .AddConnection(2,4,true,1.)
    .AddConnection(5,5,true,1.) // self-recurrent
    .AddConnection(5,3,true,1.)
    .AddConnection(4,3,true,1.)
    .AddConnection(3,4,true,1.); // recurrent
  auto consecutive = genome.MakeNet<ConsecutiveNeuralNet>();
  auto concurrent =  genome.MakeNet<ConcurrentNeuralNet>();
  auto concurrentgpu = genome.MakeNet<ConcurrentGPUNeuralNet>();
  NeuralNet* gpu_raw = concurrentgpu.get();
  ConcurrentGPUNeuralNet* ccgpu_raw = dynamic_cast<ConcurrentGPUNeuralNet*>(gpu_raw);



  auto result = consecutive->evaluate({0.5,1.5});
  auto result2 = concurrent->evaluate({0.5,1.5});
  auto result3 = concurrentgpu->evaluate({0.5,1.5});
  auto result4 = ccgpu_raw->device_evaluate({0.5,1.5});
  EXPECT_FLOAT_EQ(result[0],result2[0]);
  EXPECT_FLOAT_EQ(result[0],result3[0]);
  EXPECT_FLOAT_EQ(result[0],result4[0]);

}

TEST(NeuralNet,BiasSymmetry) {
  auto genome = Genome::ConnectedSeed(2,1);
  auto consecutive = genome.MakeNet<ConsecutiveNeuralNet>();
  auto concurrent =  genome.MakeNet<ConcurrentNeuralNet>();
  auto concurrentgpu = genome.MakeNet<ConcurrentGPUNeuralNet>();
  NeuralNet* gpu_raw = concurrentgpu.get();
  ConcurrentGPUNeuralNet* ccgpu_raw = dynamic_cast<ConcurrentGPUNeuralNet*>(gpu_raw);

  auto result = consecutive->evaluate({0.5,1.5});
  auto result2 = concurrent->evaluate({0.5,1.5});
  auto result3 = concurrentgpu->evaluate({0.5,1.5});
  auto result4 = ccgpu_raw->device_evaluate({0.5,1.5});
  EXPECT_FLOAT_EQ(result[0],result2[0]);
  EXPECT_FLOAT_EQ(result[0],result3[0]);
  EXPECT_FLOAT_EQ(result[0],result4[0]);
}


template<typename NetType>
std::vector<std::pair<_float_,_float_>> CompareCompositeNetEvaluation() {

  std::vector<Genome*> genomes(20);

  auto seed = Genome()
    .AddNode(NodeType::Bias)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Output)
    .AddNode(NodeType::Hidden)
    .AddNode(NodeType::Hidden)
    .AddConnection(1,5,true,1.12)
    .AddConnection(2,4,true,9.9)
    .AddConnection(5,5,true,0.44) // self-recurrent
    .AddConnection(5,3,true,-1.23)
    .AddConnection(4,3,true,3.3)
    .AddConnection(3,4,true,-8.2); // recurrent

  seed.set_generator(std::make_shared<RNG_MersenneTwister>());
  seed.required(std::make_shared<Probabilities>());

  for (auto& genome : genomes) {
    genome = new Genome(seed);
    genome->Mutate();
    genome->Mutate();
    genome->Mutate();
  }

  std::vector<_float_> inputs_single = {0.5,0.8};
  std::vector<_float_> inputs;
  for (auto i=0u;i<genomes.size();i++) {
    std::copy(inputs_single.begin(),inputs_single.end(),std::back_inserter(inputs));
  }

  auto net = BuildCompositeNet<NetType>(genomes,true);
  auto num_evals = 10u;

  std::vector<std::vector<std::vector<_float_>>> single_results(genomes.size());
  for (auto i=0u; i<genomes.size(); i++) {
    auto single_net = genomes[i]->MakeNet<NetType>();
    single_results[i].reserve(num_evals);

    for (auto n = 0u; n<num_evals; n++) {
      single_results[i].push_back(single_net->evaluate(inputs_single));
    }
  }

  std::vector<std::pair<_float_,_float_>> return_vals;
  for (auto n = 0u; n<num_evals; n++) {
    auto composite_result = net->evaluate(inputs);
    for (auto i=0u; i<genomes.size(); i++) {
      for (auto j=0u; j < single_results[i][n].size(); j++) {
        return_vals.push_back(std::pair<_float_,_float_>(single_results[i][n][j], composite_result[i*single_results[i][n].size()+j]));
      }
    }

  }

  return return_vals;

}


TEST(NeuralNet,CompositeNet) {

  // TODO: currently there is an issue with a CompositeNet of NetType=ConsecutiveNeuralNet

  // auto results_consecutive = CompareCompositeNetEvaluation<ConsecutiveNeuralNet>();
  // for (auto& result : results_consecutive) {
  //   EXPECT_FLOAT_EQ(result.first,result.second);
  // }

  auto results_concurrent = CompareCompositeNetEvaluation<ConcurrentNeuralNet>();
  for (auto& result : results_concurrent) {
    EXPECT_FLOAT_EQ(result.first,result.second);
  }

  auto results_concurrent_gpu = CompareCompositeNetEvaluation<ConcurrentGPUNeuralNet>();
  for (auto& result : results_concurrent_gpu) {
    EXPECT_FLOAT_EQ(result.first,result.second);
  }

}
