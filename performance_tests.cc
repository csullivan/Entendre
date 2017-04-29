#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>

#include "Population.hh"
#include "XorFitness.hh"
#include "Timer.hh"
#include "dummy.hh"
#include "ArgParser.hh"

auto population_from_xor_experiment(unsigned int pop_size = 2000, unsigned int num_generations=10) {
  auto seed = Genome::ConnectedSeed(2,1);

  auto prob = std::make_shared<Probabilities>();
  prob->new_connection_is_recurrent = 0;
  prob->keep_non_matching_father_gene = 0;
  prob->population_size = pop_size/2;
  prob->number_of_children_given_in_nursery = pop_size/2;

  Population pop(seed,
                 std::make_shared<RNG_MersenneTwister>(),
                 prob);

  pop.SetNetType<ConsecutiveNeuralNet>();
  //pop.SetNetType<ConcurrentGPUNeuralNet>();
  //pop.EnableCompositeNet(/*hetero_inputs = */true);


  for (unsigned int generation = 0u; generation < num_generations; generation++) {
    pop = std::move(pop.Reproduce([](){return std::make_unique<XorFitness>();}));

  }
  return pop;
}


int main(int argc, char** argv) {

  bool help = false;
  unsigned int num_trials;
  unsigned int pop_size;
  unsigned int num_networks;
  unsigned int num_gen;

  /////////////////////////////////////////////////////////////////////////////////////
  ArgParser parser;
  parser.option("p population_size", &pop_size)
    .description("Number of networks in population.")
    .default_value(1024);
  parser.option("n num_networks", &num_networks)
    .description("Number of networks to evaluate.")
    .default_value(1024);
  parser.option("N num_trials", &num_trials)
    .description("Number of trials for performance evaluation.")
    .default_value(100);
  parser.option("g num_generations", &num_gen)
    .description("Number of generations to evolve networks over")
    .default_value(20);
  parser.option("h help ?", &help)
    .description("Show the program options help menu.");
  parser.parse(argc,argv);
  if (help) { std::cout << parser << std::endl; return 0;}
  /////////////////////////////////////////////////////////////////////////////////////


  auto xor_pop = std::move(population_from_xor_experiment(pop_size,num_gen));
  auto xor_genomes = xor_pop.GetPopulation();
  std::cout << "/* Seed population built (size: " << xor_genomes.size() << ") */" << std::endl;

  std::vector<Genome> xor_genomes_expanded;
  xor_genomes_expanded.reserve(num_networks);

  for (auto i=0u; i<num_networks; i++) {
    xor_genomes_expanded.push_back(*xor_genomes.at(i%xor_genomes.size()));
  }
  xor_genomes.clear();
  xor_genomes.reserve(xor_genomes_expanded.size());
  for (auto& genome : xor_genomes_expanded) {
    genome.RandomizeWeights();
    xor_genomes.push_back(&genome);
  }
  std::cout << "/* Expanded genome set built (size: " << xor_genomes.size() << ") */" << std::endl;



  std::unique_ptr<NeuralNet> xor_composite_net = nullptr;

  double tperformance = 0.0;
  //----------------------------------------------------------------------------------
  {
    Timer teval([&tperformance](auto elapsed) { tperformance+=elapsed; });
    xor_composite_net = BuildCompositeNet<ConcurrentGPUNeuralNet>(xor_genomes,false);
  } std:: cout << tperformance/1.0e6 << " ms" << " for Composite net construction. " << std::endl;
  //ConcurrentGPUNeuralNet* ccgpu_composite = dynamic_cast<ConcurrentGPUNeuralNet*>((NeuralNet*)xor_composite_net.get());
  //ccgpu_composite->set_threads_per_block(32);

  //----------------------------------------------------------------------------------
  tperformance = 0.0;
  std::vector<std::unique_ptr<NeuralNet>> xor_networks;
  xor_networks.reserve(xor_genomes.size());
  {
    Timer teval([&tperformance](auto elapsed) { tperformance+=elapsed; });
    for (auto& genome : xor_genomes) {
      //xor_networks.emplace_back(genome->MakeNet<ConsecutiveNeuralNet>());
      xor_networks.emplace_back(genome->MakeNet<ConcurrentNeuralNet>());
    }
  } std:: cout << tperformance/1.0e6 << " ms" << " for construction of all networks individually. " << std::endl;

  //----------------------------------------------------------------------------------
  std::vector<_float_> inputs = {1.,1.};
  std::vector<_float_> outputs;
  outputs = xor_composite_net->evaluate(inputs);
  dummy(outputs);

  tperformance = 0.0;
  for (auto i=0u; i<num_trials; i++ ){
    Timer teval([&tperformance](auto elapsed) { tperformance+=elapsed; });
    outputs = xor_composite_net->evaluate(inputs);
    dummy(outputs);
  } std:: cout << tperformance/num_trials/1.0e6 << " ms" << " for composite net evaluation. " << std::endl;
  std::vector<_float_> gpuoutputs = std::move(outputs);
  auto tgpu = tperformance/num_trials/1.0e6;
  //----------------------------------------------------------------------------------

  std::vector<_float_> cpuoutputs;
  cpuoutputs.reserve(gpuoutputs.size());
  for (auto& net : xor_networks) {
    outputs = net->evaluate(inputs);
    std::copy(outputs.begin(),outputs.end(),std::back_inserter(cpuoutputs));
  }
  dummy(outputs);
  tperformance = 0.0;
  for (auto i=0u; i<num_trials; i++ ){
    Timer teval([&tperformance](auto elapsed) { tperformance+=elapsed; });
    for (auto& net : xor_networks) {
      outputs = net->evaluate(inputs);
    }
    dummy(outputs);
  } std:: cout << tperformance/num_trials/1.0e6 << " ms" << " for evaluation of all networks individually. " << std::endl;
  auto tcpu = tperformance/num_trials/1.0e6;


  for (auto i=0u; i<gpuoutputs.size(); i++) {
    assert(std::abs(gpuoutputs[i]-cpuoutputs[i]) < 1e-4);
  }
  std::cout << "/* PASS: outputs from GPU and CPU implementations are identical. */" << std::endl;
  std::cout << "~*~*~*~* GPU speed up: " << tcpu/tgpu << " *~*~*~*~" <<std::endl << std::endl;
  //std::cout << *xor_composite_net << std::endl;
  return 0;
}
