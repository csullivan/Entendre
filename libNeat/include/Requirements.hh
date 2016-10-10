#pragma once
#include <memory>
#include <iostream>


template <typename T>
class requires {
public:
  std::shared_ptr<T> required() const { return required_; }
  void required(const std::shared_ptr<T>& req) { required_ = req; }

protected:
  std::shared_ptr<T> required_;
};

struct Probabilities {
  size_t population_size = 100;
  size_t min_size_for_champion = 5;
  float culling_ratio = 0.5;
  float match = 0.5;
  float single_greater = 1.0;
  float single_lesser = 0.0;
  float mutate_weights = 0.8;
  float perturb_weight = 0.9;
  float mutate_link = 0.05;
  float mutate_node = 0.03;
  float mutate_only = 0.25;
  float mutate_offspring = 0.8;
  float mutate_reenable = 0.05;
  float add_recurrent = 0.05;
  float toggle_status = 0.0; // 0.1 NEAT
  float step_size = 0.1;
  float reset_weight = 4.0;
  float genetic_c1 = 1.0;
  float genetic_c2 = 1.0;
  float species_delta = 3.0;
};
