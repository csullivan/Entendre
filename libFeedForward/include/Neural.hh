#ifndef _NEURAL_HH_
#define _NEURAL_HH_
#include<vector>
#include <cstdint>


namespace Entendre {

  class Node;
  using Layer = std::vector<Node>;


  class Node {
  public:
    Node(uint16_t, uint16_t);
    ~Node();
    void SetOutput(double output) { m_output = output; }
    double GetOutput() const { return m_output; }
    double GetWeight(int i) const { return m_weights[i]; }
    void SetWeight(int i, double val)  { m_weights[i] = val; }
    double GetDWeight(int i) const { return m_dweights[i]; }
    void SetDWeight(int i, double val)  { m_dweights[i] = val; }
    double GetGradient() const { return m_gradient; }

    void Forward(const Layer&);
    void UpdateWeights(Layer&);
    void OutputGradient(double);
    void Gradient(const Layer&);
    static double Activate(double);
    static double ActivateDerivative(double);

    constexpr static double eta = 0.15;
    constexpr static double alpha = 0.5;

  private:
    int m_index;
    double m_output;
    double m_gradient;
    std::vector<double> m_weights;
    std::vector<double> m_dweights;

  };


  class FeedForward {
  public:
    FeedForward(std::vector<uint16_t>);
    ~FeedForward();
    void Feed(const std::vector<double>&);
    void BackPropogate(const std::vector<double>&);
    static double Error(const Layer&, const std::vector<double>&);
    std::vector<double> Results();

  private:
    std::vector<Layer> m_layers;
    double m_error;

  };

}

#endif //_NEURAL_HH_
