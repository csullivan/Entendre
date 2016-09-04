#include "Neural.hh"
#include <cassert>
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>

using namespace Entendre;

FeedForward::FeedForward(std::vector<uint16_t> composition) {
  assert(composition.size() >= 2);

  // build layers
  for (auto nlayer = 0u; nlayer < composition.size(); nlayer++) {
    m_layers.emplace_back();

    // retrieve # of outputs by counting nodes in next layer
    auto noutputs = (nlayer == composition.size()-1) ? 0 : composition[nlayer+1];

    // add nodes into layers
    for (int i=0; i <= composition[nlayer]; i++) {
      m_layers.back().emplace_back(noutputs,i);  // need constructor args
      // an extra bias node is added to serve as an offset to the activator function
      // the weights act as scales, and the bias node acts as a weighted offset
    }
  }
}

void FeedForward::Feed(const std::vector<double>& inputs) {

  assert(inputs.size() == m_layers.front().size() - 1);

  // set node inputs in the input layer
  for (auto i = 0u; i< inputs.size(); i++) {
    m_layers[0][i].SetOutput(inputs[i]);
  }

  // feed forward to other layers
  for (auto nlayer = 1u; nlayer < m_layers.size(); nlayer++) {

    auto& prev_layer = m_layers[nlayer-1];
    for(auto nnode = 0u; nnode < m_layers[nlayer].size() - 1; nnode++) {
      m_layers[nlayer][nnode].Forward(prev_layer);
    }
  }

}

void FeedForward::BackPropogate(const std::vector<double>& targets) {

  auto& outputs = m_layers.back();

  m_error = FeedForward::Error(outputs,targets);

  for (auto n=0u; n<outputs.size() - 1; n++) {
    outputs[n].OutputGradient(targets[n]);
  }

  // recursively calculate gradients based on the layer in front
  // moving from the back most hidden layer to the front most
  for (auto nlayer = m_layers.size()-2; nlayer > 0; nlayer--) {

    auto& hidden = m_layers[nlayer];
    for (auto n=0u; n < hidden.size(); n++) {
      hidden[n].Gradient(m_layers[nlayer+1]);
    }
  }

  for (auto nlayer = m_layers.size()-1; nlayer > 0; nlayer--) {

    auto& hidden = m_layers[nlayer];
    for (auto n=0u; n<hidden.size()-1; n++) { // bias is excluded, need to check
      hidden[n].UpdateWeights(m_layers[nlayer-1]);
    }
  }

}

void Node::OutputGradient(double target) {
  auto residual = target - m_output;
  m_gradient = residual*Node::ActivateDerivative(m_output);  // B2 = (y-yhat)*dSigma(sk)
}

void Node::Gradient(const Layer& next) {

  double sum = 0.0;
  for (auto n=0u; n < next.size()-1; n++) {
//    std::cout<< m_index << " " << m_weights.size() << std::endl;
    sum += GetWeight(n)*next[n].GetGradient();
  }
  m_gradient = sum*Node::ActivateDerivative(m_output);
}

double FeedForward::Error(const Layer& output, const std::vector<double>& targets) {

  double rms2 = 0.0;
  for (auto n = 0u; n < output.size(); n++) {
    double residual = targets[n] - output[n].GetOutput();
    rms2 += residual*residual;
  }
  rms2 /= (output.size()-1);
  return std::sqrt(rms2);
}

void Node::UpdateWeights(Layer& prev) {

  for (auto& node : prev) {

    auto prev_dweight = node.GetDWeight(m_index);
    auto dweight = eta*node.GetOutput()*m_gradient + alpha*prev_dweight;
    node.SetWeight(m_index,node.GetWeight(m_index)+dweight);
    node.SetDWeight(m_index,dweight);
  }
}

std::vector<double> FeedForward::Results() {

  std::vector<double> results;
  const auto& output = m_layers.back();
  for (auto n = 0u; n<output.size()-1; n++) {
    results.push_back(output[n].GetOutput());
  }
  return results;
}

FeedForward::~FeedForward() { }


Node::Node(uint16_t noutputs, uint16_t index) : m_index(index), m_output(1) {
  std::mt19937 mt(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  for (auto i=0u; i<noutputs; i++) {
    m_weights.push_back(dis(mt));
    m_dweights.push_back(0.);
  }
}

Node::~Node() { }

void Node::Forward(const Layer& prev) {

  double sum = 0.0;
  for (auto const& node : prev) {
    sum += node.GetOutput()*node.GetWeight(m_index);
  }
  m_output = Node::Activate(sum);
}

double Node::Activate(double x) {
  return tanh(x);
}

double Node::ActivateDerivative(double x) {
  return 1-x*x;
}
