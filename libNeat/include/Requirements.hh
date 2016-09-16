#pragma once
#include <memory>


template <typename T>
class requires {
public:
  std::shared_ptr<T> required() const { return required_; }
  void required(const std::shared_ptr<T>& req) { required_ = req; }
private:
  std::shared_ptr<T> required_;
};

struct Probabilities {
  const float match = 0.0;
  const float single_greater = 0.0;
  const float single_lesser = 0.0;
  const float mutate_weight = 0.9;
  const float mutate_link = 0.0;
  const float mutate_node = 0.0;
  const float step_size = 0.1;
  const float reset_weight = 4.0;
  const float genetic_c1 = 0.0;
  const float genetic_c2 = 0.0;
  const float species_delta = 0.0;
};
