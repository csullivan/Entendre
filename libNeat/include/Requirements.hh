#pragma once
#include <memory>
#include <iostream>


template <typename T>
class requires {
public:
  std::shared_ptr<T> required() const { return required_; }
  void required(const std::shared_ptr<T>& req) { required_ = req; }

private:
  std::shared_ptr<T> required_;
};

struct Probabilities {
  const float match = 0.5;
  const float single_greater = 1.0;
  const float single_lesser = 0.0;
  const float mutate_weights = 0.8;
  const float perturb_weight = 0.9;
  const float mutate_link = 0.05;
  const float mutate_node = 0.03;
  const float mutate_only = 0.25;
  const float mutate_offspring = 0.8;
  const float mutate_reenable = 0.05;
  const float add_recurrent = 0.05;
  const float toggle_status = 0.0; // 0.1 NEAT
  const float step_size = 0.1;
  const float reset_weight = 4.0;
  const float genetic_c1 = 1.0;
  const float genetic_c2 = 1.0;
  const float species_delta = 3.0;
};
