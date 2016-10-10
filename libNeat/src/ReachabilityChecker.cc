#include "ReachabilityChecker.hh"

#include <cassert>

#include "Random.hh"

ReachabilityChecker::ReachabilityChecker(size_t num_nodes, size_t num_inputs)
  : num_nodes(num_nodes), num_inputs(num_inputs),
    num_possible_normal_connections((num_nodes-num_inputs)*(num_nodes-1)),
    num_possible_recurrent_connections(num_nodes-num_inputs),
    mat(num_nodes*num_nodes) {
  // All nodes are reachable from themselves
  // Doesn't count as a connection, though
  for(size_t i=0; i<num_nodes; i++) {
    at(i,i).reachable_normal = true;
    at(i,i).reachable_either = true;
  }
}

void ReachabilityChecker::AddConnection(size_t origin, size_t destination) {
  auto& element = at(origin, destination);

  assert(destination >= num_inputs);
  assert(!element.has_any_connection);
  element.has_any_connection = true;

  bool will_be_recurrent = at(destination,origin).reachable_normal;

  if(will_be_recurrent) {
    element.has_recurrent_connection = true;
    num_possible_recurrent_connections--;
    fill_either_reachable(origin,destination);
  } else {
    element.has_normal_connection = true;
    num_possible_normal_connections--;
    fill_normal_reachable(origin,destination);
    fill_either_reachable(origin,destination);
  }
}

void ReachabilityChecker::fill_normal_reachable(size_t origin, size_t destination) {
  // If we already knew that it was reachable, no more work to be done.
  if(at(origin,destination).reachable_normal) {
    return;
  }

  // We can reach from the origin to the destination
  at(origin,destination).reachable_normal = true;

  //Now that destination is normal-reachable from origin, a connection
  //from destination to origin would now be recurrent, rather than
  //normal.  Only do the modification if there could have been a
  //connection added from destination to origin.  That is, if origin
  //is not an input node.
  if(origin >= num_inputs) {
    num_possible_normal_connections--;
    num_possible_recurrent_connections++;
  }

  // Anywhere that can be reached from the destination, we can reach
  for(size_t i=0; i<num_nodes; i++) {
    if(at(destination,i).reachable_normal) {
      fill_normal_reachable(origin,i);
    }
  }

  // Anywhere that can reach us, can now reach the destination
  for(size_t i=0; i<num_nodes; i++) {
    if(at(i,origin).reachable_normal) {
      fill_normal_reachable(i,destination);
    }
  }
}

void ReachabilityChecker::fill_either_reachable(size_t origin, size_t destination) {
  // If we already knew that it was reachable, no more work to be done.
  if(at(origin,destination).reachable_either) {
    return;
  }

  // We can reach from the origin to the destination
  at(origin,destination).reachable_either = true;

  // Anywhere that can be reached from the destination, we can reach
  for(size_t i=0; i<num_nodes; i++) {
    if(at(destination,i).reachable_either) {
      fill_either_reachable(origin,i);
    }
  }

  // Anywhere that can reach us, can now reach the destination
  for(size_t i=0; i<num_nodes; i++) {
    if(at(i,origin).reachable_either) {
      fill_either_reachable(i,destination);
    }
  }
}

std::pair<int,int> ReachabilityChecker::RandomNormalConnection(RNG& rng) {
  // Nothing left, don't bother
  if(num_possible_normal_connections == 0) {
    return {-1, -1};
  }

  // If we have a decent chance of finding one, try some random elements.
  double prob = double(num_possible_normal_connections)/double(num_nodes*(num_nodes-num_inputs));
  if(prob > 0.25) {
    for(int attempt=0; attempt<10; attempt++) {
      size_t origin = num_nodes*rng();
      size_t destination = num_inputs + (num_nodes-num_inputs)*rng();
      if(could_add_normal(origin, destination)) {
        return {origin, destination};
      }
    }
  }

  // Fallback to a linear search
  size_t desired_index = num_possible_normal_connections*rng();
  size_t index = 0;
  for(size_t origin = 0; origin<num_nodes; origin++) {
    for(size_t destination = num_inputs; destination<num_nodes; destination++) {
      if(could_add_normal(origin, destination)) {
        if(index == desired_index) {
          return {origin, destination};
        }
        index++;
      }
    }
  }

  // Shouldn't ever reach here
  assert(false);
  return {-1,-1};
}

std::pair<int,int> ReachabilityChecker::RandomRecurrentConnection(RNG& rng) {
  // Nothing left, don't bother
  if(num_possible_recurrent_connections == 0) {
    return {-1, -1};
  }

  // If we have a decent chance of finding one, try some random elements.
  double prob = double(num_possible_recurrent_connections)/double(num_nodes*(num_nodes-num_inputs));
  if(prob > 0.25) {
    for(int attempt=0; attempt<10; attempt++) {
      size_t origin = num_nodes*rng();
      size_t destination = num_inputs + (num_nodes-num_inputs)*rng();
      if(could_add_recurrent(origin, destination)) {
        return {origin, destination};
      }
    }
  }

  // Fallback to a linear search
  size_t desired_index = num_possible_recurrent_connections*rng();
  size_t index = 0;
  for(size_t origin = 0; origin<num_nodes; origin++) {
    for(size_t destination = num_inputs; destination<num_nodes; destination++) {
      if(could_add_recurrent(origin, destination)) {
        if(index == desired_index) {
          return {origin, destination};
        }
        index++;
      }
    }
  }

  // Shouldn't ever reach here
  assert(false);
  return {-1,-1};
}
