#pragma once
#include "Genome.hh"
#include "NeuralNet.hh"
#include "PopulationHelpers.hh"
#include "FitnessEvaluator.hh"

#include <vector>
#include <limits>
#include <unordered_map>



class Population : public uses_random_numbers,
                   public requires<Probabilities> {
public:
  /// Construct a population, starting from a seed genome
  Population(Genome& first,
             std::shared_ptr<RNG>,std::shared_ptr<Probabilities>);

  /// Construct a population, starting from the specified population of organisms and species
  /**
     Note: Assumes that all genomes have the same RNG and
           Probabilities as are being passed here.
   */
  Population(std::vector<Species> species,
             std::shared_ptr<RNG>,std::shared_ptr<Probabilities>);

  Population(const Population&) = default;
  Population(Population&&) = default;
  Population& operator=(const Population& rhs) = default;
  Population& operator=(Population&&) = default;

  /// Evaluate the fitness function for each neural net.
  template<class Callable>
  void Evaluate(Callable&& fitness) {
    for(auto& spec : species) {
      for(auto& org : spec.organisms) {
        org.fitness = fitness(*org.network());
      }
    }
    CalculateAdjustedFitness();
  }

  void Evaluate(std::function<std::unique_ptr<FitnessEvaluator>(void)> evaluator_factory) {

    struct fitness_kernel {
      NetProxy proxy;
      std::unique_ptr<FitnessEvaluator> eval;
      std::vector<_float_> result;
    };
    std::vector<fitness_kernel> kernels;

    for (auto& spec : species) {
      for (auto& org : spec.organisms) {
        kernels.push_back({ &org, evaluator_factory(), {} });
      }
    }

    while (true) {
      bool continue_looping = false;
      // load one set of inputs for each network
      // or finalize and set fitness value
      for (auto& kernel : kernels) {
        kernel.eval->step(kernel.proxy);
      }

      // eval each network with loaded inputs
      for (auto& kernel : kernels) {
        if (kernel.proxy.has_inputs()) {
          kernel.result = kernel.proxy.evaluate();
          continue_looping = true;
        }
      }

      // if there are no more inputs then
      // the fitness function has been evaluated
      // and we are done
      if (!continue_looping) { break; }

      // call the proxy callbacks
      for (auto& kernel : kernels) {
        kernel.proxy.callback(kernel.result);
        kernel.proxy.clear();
      }
    }

    CalculateAdjustedFitness();
  }

  Population Reproduce(std::function<std::unique_ptr<FitnessEvaluator>(void)> evaluator_factory) {
    Evaluate(evaluator_factory);
    return Reproduce();
  }

  /// Reproduce, using the fitness function given.
  Population Reproduce(std::function<double(NeuralNet&) > fitness) {
    Evaluate(fitness);
    return Reproduce();
  }

  /// Reproduce, using the already evaluated fitness function.
  /**
     Assumes that Evaluate() has already been called.
   */
  Population Reproduce();

  /// Returns the best neural net in the population.
  /**
     Uses the fitness value calculated by the most recent call to Evaluate or Reproduce.
     If neither has been called, returns nullptr.
   */
  NeuralNet* BestNet();

  /// Returns the number of species in the population
  /**
     Uses the speciation from the most recent call to Reproduce.
     If Reproduce has not been called, returns 0.
   */
  unsigned int NumSpecies();
  unsigned int NumViableSpecies();

  unsigned int SpeciesSize(size_t i) const;

  std::pair<double, double> MeanStdDev() const;

  const std::vector<Species>& GetSpecies() const { return species; }

  template<typename NetType>
  void SetNetType() {
    converter = std::make_shared<GenomeConverter_Impl<NetType> >();
  }

private:
  std::vector<Species> MakeNextGenerationSpecies();
  std::vector<Genome> MakeNextGenerationGenomes();
  void DistributeChildrenByRank(std::vector<unsigned int>&) const;
  void DistributeNurseryChildren(std::vector<unsigned int>&) const;


  void Speciate(std::vector<Species>& species,
                const std::vector<Genome>& genomes);
  void CalculateAdjustedFitness();

  std::vector<Species> species;
  std::shared_ptr<GenomeConverter> converter;
};
