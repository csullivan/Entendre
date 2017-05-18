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

  size_t stale_species_num_generations = 15;
  float necessary_species_improvement = 0.0;
  float stale_species_penalty = 0.01;
  bool keep_empty_species = false;
  bool species_representative_from_previous_gen = true;

  float matching_gene_choose_mother = 0.5;
  float keep_non_matching_mother_gene = 1.0; // fitness: mother > father
  float keep_non_matching_father_gene = 0.0;

  float mutation_prob_adjust_weights = 0.8;
  float weight_mutation_is_severe = 0.1;
  float weight_mutation_small_adjust = 0.5;
  float weight_mutation_reset_range = 20.0;

  float mutation_prob_add_connection = 0.05;
  float new_connection_is_recurrent = 0.05;
  float mutation_prob_add_node = 0.03;
  //float mutate_offspring = 0.8;
  float mutation_prob_reenable_connection = 0.05;
  float mutation_prob_toggle_connection = 0.0; // 0.1 NEAT

  float genetic_distance_structural = 1.0;
  float genetic_distance_weights = 0.2;
  float genetic_distance_species_threshold = 3.0;

  bool use_compositional_pattern_producing_networks = false;
  
  size_t nursery_age = 15;
  bool fixed_nursery_size = true;
  size_t number_of_children_given_in_nursery = 100;
  float species_survival_percentile = 0.3;

};
