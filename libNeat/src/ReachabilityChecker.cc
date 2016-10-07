#include "ReachabilityChecker.hh"

#include <cassert>

ReachabilityChecker::ReachabilityChecker(size_t num_nodes)
  : num_nodes(num_nodes), mat(num_nodes*num_nodes) {
  // All nodes are reachable from themselves
  // Doesn't count as a connection, though
  for(size_t i=0; i<num_nodes; i++) {
    at(i,i).reachable_normal = true;
    at(i,i).reachable_either = true;
  }
}

void ReachabilityChecker::AddConnection(size_t origin, size_t destination) {
  auto& element = at(origin, destination);

  assert(!element.has_any_connection);
  element.has_any_connection = true;

  bool will_be_recurrent = at(destination,origin).reachable_normal;

  if(will_be_recurrent) {
    element.has_recurrent_connection = true;
    fill_either_reachable(origin,destination);
  } else {
    element.has_normal_connection = true;
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
