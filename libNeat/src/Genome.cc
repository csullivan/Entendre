#include "Genome.hh"
#include <vector>
#include <iostream>

Genome::operator NeuralNet() const {
  NeuralNet net(nodes);
  // add network connections where appropriate
  for (auto const& gene : genes) {
    if (gene.enabled) {
      net.add_connection(gene.link.origin,gene.link.dest,gene.link.weight);
    }
  }

  return net;
}

void Genome::operator=(const Genome& rhs) {
  this->nodes = rhs.nodes;
  this->genes = rhs.genes;
  generator = rhs.generator;
}

// Genome mating
Genome Genome::operator()(const Genome& father) {
  // Implicit assumption: if there is a fitness difference
  // then the mother must be the more fit genome. i.e.
  // child = mother(father) s.t. fitness(mother) > fitness(father)
  auto& mother = *this;
  Genome child;

  auto paternal = father.genes.begin();

  // Mother is most fit, so we will not take
  // any structure above and beyond what is in
  // the mother. Thus, when we run out of mother
  // genes to iterate over, we are done.
  for (auto& maternal : mother.genes) {
    const Gene* candidate = nullptr;

    if (paternal != father.genes.end()){
      if (maternal.innovation_number == paternal->innovation_number) {
        candidate = (random()<Constants::Match) ? &maternal : &(*paternal);
      }
      else {
        candidate = &maternal;
      }
      paternal++;
    } else {
      // no paternal genes left
      candidate =  &maternal;
    }

    //if (!candidate) { continue; }

    // does child already have a gene like the candidate?
    bool unique = true;
    for (auto& gene : child.genes) {
      if (gene == *candidate) {
        unique = false; break;
      }
    }

    // add gene to child
    if (unique) {
      child.genes.push_back(*candidate);
    }
  }

  // copy in nodes from more fit parent (mother)
  child.nodes = mother.nodes;

  // neshima
  return child;
}

Genome& Genome::AddNode(NodeType type) {
  nodes.emplace_back(type);
  return *this;
}

Genome& Genome::AddGene(unsigned int origin, unsigned int dest, ConnectionType type,
             bool status, double weight) {

  unsigned long innovation = 0;
  if (genes.size()) {
    innovation = Hash(dest,origin,genes.back().innovation_number);
  }
  else {
    innovation = Hash(dest,origin,innovation);
  }


  genes.push_back({status,innovation,Connection(origin,dest,type,weight)});
  return *this;
}

void Genome::WeightMutate() {
  if (!generator) { throw std::runtime_error("No RNG engine set. Replace this with class specific exception."); }
  for (auto& gene : genes) {
    // perturb weight by a small amount
    if(random()<Constants::PerturbWeight) {
      gene.link.weight += Constants::StepSize*(2*random()-1);
    } else { // otherwise randomly set weight within reset range
      gene.link.weight = (random() - 0.5)*Constants::ResetWeightScale;
    }
  }
}

void Genome::LinkMutate() {

}

void Genome::NodeMutate() {

}

void Genome::Mutate() {

}

void Genome::PrintInnovations() {
  for (auto const& gene : genes) {
    std::cout << gene.innovation_number << std::endl;
  }
}
