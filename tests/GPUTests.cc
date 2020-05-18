
#include <gtest/gtest.h>
#include "Genome.hh"
#include "ConcurrentGPUNeuralNet.hh"
#include "Timer.hh"

unsigned int nTrials = 1;



template<typename Derived, typename Base, typename Del>
std::unique_ptr<Derived, Del>
static_unique_ptr_cast( std::unique_ptr<Base, Del>&& p )
{
  auto d = static_cast<Derived *>(p.release());
  return std::unique_ptr<Derived, Del>(d, std::move(p.get_deleter()));
}


template<typename NetType, bool use_gpu=false>
std::vector<_float_> evaluation_performance(const Genome& genome, const std::vector<_float_>& inputs, unsigned int nTrials=1);

TEST(ConcurrentGPUNeuralNet, GEMM){
  ConcurrentGPUNeuralNet gpu;
  gpu.gemm();
}

/*
TEST(ConcurrentGPUNeuralNet,OneToManyNoHidden){

  //auto sigmoid = [](_float_ val) {return 1/(1 + std::exp(-val));};

  auto genome = Genome();

  //auto nInputs = 1u + 1u;
  auto nInputs = 1u;
  auto nOutputs = 10000u;


  std::vector<_float_> inputs = { 1.0 };
  for (auto i=0u;i<nInputs; i++) { // skips bias
    genome.AddNode(NodeType::Input);
  }
  for (auto i=0u;i<nOutputs; i++) {
    genome.AddNode(NodeType::Output);
  }



  // Connect inputs directly to outputs
  for (auto i=0u;i<nInputs; i++) {
    for (auto j=nInputs;j<nInputs+nOutputs; j++) {
      genome.AddConnection(i,j,true,1.);
    }
  }


  {
    auto draw_net = static_unique_ptr_cast<ConcurrentGPUNeuralNet>(genome.MakeNet<ConcurrentGPUNeuralNet>());
    draw_net->device_evaluate(inputs);
    std::cout << *draw_net << std::endl;
  }

  std::vector<std::vector<_float_>> results;
  // {
  //   double tperformance = 0.0;
  //   double tperformance2 = 0.0;
  //   auto net = static_unique_ptr_cast<ConcurrentGPUNeuralNet>(genome.MakeNet<ConcurrentGPUNeuralNet>());
  //   //auto net = genome.MakeNet<ConcurrentGPUNeuralNet>();
  //   // First evaluation includes network sorting (construction)
  //   // so we will time the evaluations thereafter
  //   net->evaluate(inputs);

  //   // Time the network evaluation
  //   for (auto i=0u; i < nTrials; i++)
  //   {


  //     auto  t1 = chrono::high_resolution_clock::now();
  //     auto result = net->evaluate(inputs);
  //     auto  t2 = chrono::high_resolution_clock::now();
  //     tperformance2 += chrono::duration_cast<chrono::nanoseconds>(t2-t1).count();
  //     std::cout << result[0] << "\r" << std::flush;


  //     Timer teval([&tperformance](auto elapsed) {
  //         tperformance+=elapsed;
  //       });
  //     result = net->evaluate(inputs);
  //     std::cout << result[0] << "\r" << std::flush;
  //   }
  //   std::cout << tperformance/nTrials/1.0e6 << " ms"
  //             << " : ConcurrentGPUNeuralNet"
  //             << std::endl;
  //   std::cout << tperformance2/nTrials/1.0e6 << " ms"
  //             << " : ConcurrentGPUNeuralNet (explicit clock)"
  //             << std::endl;


  //   results.push_back(net->evaluate(inputs));
  // }


  results.push_back(evaluation_performance<ConcurrentGPUNeuralNet,true>(genome,inputs,nTrials));
  std::cout << " : ConcurrentGPUNeuralNet: " << std::endl;
  results.push_back(evaluation_performance<ConcurrentGPUNeuralNet>(genome,inputs,nTrials));
  std::cout << " : ConcurrentGPUNeuralNet (CPU): " << std::endl;
  results.push_back(evaluation_performance<ConcurrentNeuralNet>(genome,inputs,nTrials));
  std::cout << " : ConcurrentNeuralNet: " << std::endl;
  results.push_back(evaluation_performance<ConsecutiveNeuralNet>(genome,inputs,nTrials));
  std::cout << " : ConsecutiveNeuralNet: " << std::endl;


  for (auto const& outputs : results) {
    for (auto i=0u;i<results[0].size();i++) {
      EXPECT_EQ(results[0][i],outputs[i]);
    }
  }

}
*/
/*
TEST(ConcurrentGPUNeuralNet,ManyToManyNoHidden){

  //auto sigmoid = [](_float_ val) {return 1/(1 + std::exp(-val));};

  auto genome = Genome();

  //auto nInputs = 1u + 1u;
  auto nInputs  = 1000u;
  auto nOutputs = 1000u;


  std::vector<_float_> inputs(nInputs,0);
  for (auto i=0u;i<nInputs; i++) { // skips bias
    genome.AddNode(NodeType::Input);
  }
  for (auto i=0u;i<nOutputs; i++) {
    genome.AddNode(NodeType::Output);
  }



  // Connect inputs directly to outputs
  for (auto i=0u;i<nInputs; i++) {
    genome.AddConnection(i,nInputs+i,true,1.);
  }


  {
    auto draw_net = static_unique_ptr_cast<ConcurrentGPUNeuralNet>(genome.MakeNet<ConcurrentGPUNeuralNet>());
    draw_net->evaluate(inputs);
    std::cout << *draw_net << std::endl;
  }



  std::vector<std::vector<_float_>> results;
  results.push_back(evaluation_performance<ConcurrentGPUNeuralNet,true>(genome,inputs,nTrials));
  std::cout << " : ConcurrentGPUNeuralNet: " << std::endl;
  results.push_back(evaluation_performance<ConcurrentGPUNeuralNet>(genome,inputs,nTrials));
  std::cout << " : ConcurrentGPUNeuralNet (CPU): " << std::endl;
  results.push_back(evaluation_performance<ConcurrentNeuralNet>(genome,inputs,nTrials));
  std::cout << " : ConcurrentNeuralNet: " << std::endl;
  results.push_back(evaluation_performance<ConsecutiveNeuralNet>(genome,inputs,nTrials));
  std::cout << " : ConsecutiveNeuralNet: " << std::endl;


  for (auto const& outputs : results) {
    for (auto i=0u;i<results[0].size();i++) {
      EXPECT_EQ(results[0][i],outputs[i]);
    }
  }

}
*/

/*
TEST(ConcurrentGPUNeuralNet,MNISTTopology){

  //auto sigmoid = [](_float_ val) {return 1/(1 + std::exp(-val));};

  auto genome = Genome();

  // a typical MNIST network
  auto nInputs = 784u;
  auto nHidden = 10u;
  auto nOutputs = 10u;

  auto nTotal = nInputs + nHidden + nOutputs;

  std::vector<_float_> inputs(nInputs,0);

  for (auto i=0u;i<nInputs; i++) {
    genome.AddNode(NodeType::Input);
  }


  for (auto i=0u;i<nOutputs; i++) {
    genome.AddNode(NodeType::Output);
  }

  for (auto i=0u;i<nHidden; i++) {
    genome.AddNode(NodeType::Hidden);
  }


  // Connect every input node to every hidden node
  for (auto i=0u;i<nInputs; i++) {
    for (auto j=nInputs+nOutputs;j<nTotal; j++) {
      genome.AddConnection(i,j,true,1.);
    }
  }

  // Connect every hidden node to every output node
  for (auto j=nInputs+nOutputs;j<nTotal; j++) {
    for (auto k=nInputs; k<nInputs+nOutputs; k++) {
      genome.AddConnection(j,k,true,1.);
    }
  }



  // {
  //   auto draw_net = static_unique_ptr_cast<ConcurrentGPUNeuralNet>(genome.MakeNet<ConcurrentGPUNeuralNet>());
  //   draw_net->evaluate(inputs);
  //   std::cout << *draw_net << std::endl;
  // }



  std::vector<std::vector<_float_>> results;
  results.push_back(evaluation_performance<ConcurrentGPUNeuralNet,true>(genome,inputs,nTrials));
  std::cout << " : ConcurrentGPUNeuralNet: " << std::endl;
  results.push_back(evaluation_performance<ConcurrentGPUNeuralNet>(genome,inputs,nTrials));
  std::cout << " : ConcurrentGPUNeuralNet (CPU): " << std::endl;
  results.push_back(evaluation_performance<ConcurrentNeuralNet>(genome,inputs,nTrials));
  std::cout << " : ConcurrentNeuralNet: " << std::endl;
  results.push_back(evaluation_performance<ConsecutiveNeuralNet>(genome,inputs,nTrials));
  std::cout << " : ConsecutiveNeuralNet: " << std::endl;


  for (auto const& outputs : results) {
    for (auto i=0u;i<results[0].size();i++) {
      EXPECT_EQ(results[0][i],outputs[i]);
    }
  }

}
*/

/*
TEST(ConcurrentGPUNeuralNet,FakePopulationTest){

  //auto sigmoid = [](_float_ val) {return 1/(1 + std::exp(-val));};

  auto genome = Genome();

  int num_subnets = 1000;

  // Add all the nodes of the subnets
  for (int i = 0; i < num_subnets; i++) {
    genome
      .AddNode(NodeType::Input)
      .AddNode(NodeType::Input);
  }
  for (int i = 0; i < num_subnets; i++) {
    genome
      .AddNode(NodeType::Output);
  }
  for (int i = 0; i < num_subnets; i++) {
    genome
      .AddNode(NodeType::Hidden);
  }

  // connect each subnet
  int subnet_input_node = 0;
  int subnet_output_node = num_subnets*2;
  int subnet_hidden_node = subnet_output_node + num_subnets;
  for (int i = 0; i < num_subnets; i++) {
    genome
      .AddConnection(subnet_input_node+0,subnet_output_node,true,1.)
      .AddConnection(subnet_input_node+1,subnet_output_node,true,1.)
      .AddConnection(subnet_input_node+0,subnet_hidden_node,true,1.)
      .AddConnection(subnet_input_node+1,subnet_hidden_node,true,1.)
      .AddConnection(subnet_hidden_node,subnet_output_node,true,1.);

    subnet_input_node += 2;
    subnet_output_node++;
    subnet_hidden_node++;
  }

  // end genome construction ----------------------------------------------------------

  std::vector<_float_> inputs(num_subnets*2,0.5);
  {
    auto draw_net = static_unique_ptr_cast<ConcurrentGPUNeuralNet>(genome.MakeNet<ConcurrentGPUNeuralNet>());
    draw_net->evaluate(inputs);
    std::cout << *draw_net << std::endl;
  }




  std::vector<std::vector<_float_>> results;
  results.push_back(evaluation_performance<ConcurrentGPUNeuralNet,true>(genome,inputs,nTrials));
  std::cout << " : ConcurrentGPUNeuralNet: " << std::endl;
  results.push_back(evaluation_performance<ConcurrentGPUNeuralNet>(genome,inputs,nTrials));
  std::cout << " : ConcurrentGPUNeuralNet (CPU): " << std::endl;
  results.push_back(evaluation_performance<ConcurrentNeuralNet>(genome,inputs,nTrials));
  std::cout << " : ConcurrentNeuralNet: " << std::endl;
  results.push_back(evaluation_performance<ConsecutiveNeuralNet>(genome,inputs,nTrials));
  std::cout << " : ConsecutiveNeuralNet: " << std::endl;


  for (auto const& outputs : results) {
    for (auto i=0u;i<results[0].size();i++) {
      EXPECT_EQ(results[0][i],outputs[i]);
    }
  }

}
*/

template<bool>
struct Network;

template<>
struct Network<true> {
  static std::vector<_float_> evaluate(auto const& net, const std::vector<_float_>& inputs) {
    return net->evaluate(inputs);
  }
};

template<>
struct Network<false> {
  static std::vector<_float_> evaluate(auto const& net, const std::vector<_float_>& inputs) {
    return net->host_evaluate(inputs);
  }
};


template<typename NetType, bool use_gpu>
std::vector<_float_> evaluation_performance(const Genome& genome, const std::vector<_float_>& inputs, unsigned int nTrials) {

  // Time the network construction ------------------------------------------------------------------
  double tperformance = 0.0;
  {
    Timer teval([&tperformance](auto elapsed) {
        tperformance+=elapsed;
      });
    genome.MakeNet<NetType>();
  } std:: cout << "Network construction time: " <<tperformance/1.0e6 << " ms" << std::endl;
  // END Time the network construction --------------------------------------------------------------

  auto net = static_unique_ptr_cast<NetType>(genome.MakeNet<NetType>());

  // Time the network sorting + first evaluation ----------------------------------------------------
  tperformance = 0.0;
  {
    Timer teval([&tperformance](auto elapsed) {
        tperformance+=elapsed;
      });
    Network<use_gpu>::evaluate(net,inputs);
  } std::cout << "Sorting time (+ initial evaluation): " <<tperformance/1.0e6 << " ms " << std::endl;
  // END Time the network sorting + first evaluation ------------------------------------------------


  // Time the network evaluation --------------------------------------------------------------------
  tperformance = 0.0;
  for (auto i=0u; i < nTrials; i++)
  {

    Timer teval([&tperformance](auto elapsed) {
        tperformance+=elapsed;
      });
    auto result = Network<use_gpu>::evaluate(net,inputs);
    std::cout << result[0] << "\r" << std::flush;

  } std::cout << tperformance/nTrials/1.0e6 << " ms";
  // END Time the network evaluation ----------------------------------------------------------------

  return Network<use_gpu>::evaluate(net,inputs);
}

