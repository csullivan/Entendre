#include "ConcurrentGPUNeuralNet.hh"

#include <cassert>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>
#include <algorithm>

#include "logging.h"
#include "math.h"


#define cuda_assert(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
//#ifndef NDEBUG
  if (code != cudaSuccess) {
    fprintf(stderr,"cuda_assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) { throw code; }
  }
//#endif
}




__global__ void device_matmul() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
}

static void ConcurrentGPUNeuralNet::matmul()
{
  constexpr size_t num_blocks = 1;
  constexpr size_t num_threads = 32;
  device_matmul<<<num_blocks,num_threads>>>();
}


ConcurrentGPUNeuralNet::~ConcurrentGPUNeuralNet() {
  if (node_) { cuda_assert(cudaFree(node_)); }
  if (origin_) { cuda_assert(cudaFree(origin_)); }
  if (dest_) { cuda_assert(cudaFree(dest_)); }
  if (weight_) { cuda_assert(cudaFree(weight_)); }
  if (action_list_) { cuda_assert(cudaFree(action_list_)); }
}

ConcurrentGPUNeuralNet::EvaluationOrder ConcurrentGPUNeuralNet::compare_connections(const Connection& a, const Connection& b) {
    // A recurrent connection must be used before the origin is overwritten.
  if (a.type == ConnectionType::Recurrent && a.origin == b.dest) { return EvaluationOrder::LessThan; }
  if (b.type == ConnectionType::Recurrent && b.origin == a.dest) { return EvaluationOrder::GreaterThan; }

  // A normal connection must occur after every connection incoming to its origin has completed.
  if (a.type == ConnectionType::Normal && a.dest == b.origin) { return EvaluationOrder::LessThan; }
  if (b.type == ConnectionType::Normal && b.dest == a.origin) { return EvaluationOrder::GreaterThan; }

  // Two connections writing to the same destination must be in different sets.
  if (a.dest == b.dest) {
    // A self-recurrent connection happens at the same time as
    // zero-ing out, and so must occur first of all connections
    // writing to that node.
    if(a.origin == a.dest) {
      return EvaluationOrder::LessThan;
    } else if (b.origin == b.dest) {
      return EvaluationOrder::GreaterThan;
    } else {
      // Choice here is absolutely arbitrary.
      // This is arbitrary, and consistent.
      if(a.origin < b.origin) {
        return EvaluationOrder::GreaterThan;
      } else {
        return EvaluationOrder::LessThan;
      }
    }
  }

  // else a & b are not adjacent and cannot be compared
  return EvaluationOrder::Unknown;
}

void ConcurrentGPUNeuralNet::sort_connections(unsigned int first, unsigned int num_connections) {
  if(connections_sorted) {
    return;
  }

  // if the first connection in the list to sort is not
  // the first connection, and num_connections is zero
  // this is an error
  assert(!(first!=0 && num_connections==0));
  // if num_connections is zero, then we will sort all connections
  num_connections = num_connections > 0 ? num_connections : connections.size();
  // the number of connections to sort cannot be
  // larger than the total number of connections
  assert(first+num_connections <= connections.size());

  // zero out connection set index for use in sorting
  for (auto i=first; i<first+num_connections; i++) {
    connections[i].set = 0;
  }
  for(auto i=first; i<first+num_connections; i++) {
    for(auto j=i+1; j<first+num_connections; j++) {
      Connection& conn1 = connections[i];
      Connection& conn2 = connections[j];
      switch(compare_connections(conn1,conn2)) {
        case EvaluationOrder::GreaterThan:
          conn1.set++;
          break;

        case EvaluationOrder::LessThan:
          conn2.set++;
          break;

        case EvaluationOrder::Unknown:
          break;
      }
    }
  }

  auto split_iter = connections.begin()+first;
  auto last_iter = connections.begin()+first+num_connections;
  size_t current_set_num = 0;
  while(split_iter != last_iter) {
    auto next_split = std::partition(split_iter, last_iter,
                                     [](const Connection& conn) {
                                       return conn.set == 0;
                                     });
    assert(next_split != split_iter);

    // These could be run now, no longer need to track number of
    // depencencies.
    for(auto iter = split_iter; iter<next_split; iter++) {
      iter->set = current_set_num;
    }
    current_set_num++;

    // Decrease number of dependencies for everything else.
    for(auto iter_done = split_iter; iter_done<next_split; iter_done++) {
      for(auto iter_not_done = next_split; iter_not_done<last_iter; iter_not_done++) {
        if (compare_connections(*iter_done,*iter_not_done) == EvaluationOrder::LessThan) {
          iter_not_done->set--;
        }
      }
    }

    split_iter = next_split;
  }

  // build the action list if num_connections was the total set
  // or if this is the last subset of connections (all others are sorted)
  if (first + num_connections == connections.size()) {
    // if first is nonzero then we have been sorting based on subsets and now all subset lock free buckets
    // need to be merged in a sort of the entire connections list where set now is the lock free set index
    // (before it was used as the subnet index)

    if (first != 0) {
      // sort connections based on evaluation set number if not already done
      std::sort(connections.begin(),connections.end(),[](Connection a, Connection b){ return a.set < b.set; });
    }
    // build struct of arrays for use on GPU
    for (auto i=0u; i<connections.size(); i++) {
      auto& conn = connections[i];
      connection_list.add(conn.origin,conn.dest,conn.weight);
    }
    build_action_list();
    connections_sorted = true;
    connections.clear();
    synchronize();
  }
}

void ConcurrentGPUNeuralNet::ConcurrentGPUNeuralNet::build_action_list() {

  unsigned int num_connection_sets = connections.back().set+1;
  std::vector<unsigned int> connection_set_sizes(num_connection_sets, 0);
  for(auto& conn : connections) {
    connection_set_sizes[conn.set]++;
  }

  std::vector<unsigned int> earliest_zero_out_indices(nodes.size(), 0);
  std::vector<unsigned int> earliest_sigmoid_indices(nodes.size(), 0);

  std::vector<unsigned int> latest_zero_out_indices(nodes.size(), num_connection_sets);
  std::vector<unsigned int> latest_sigmoid_indices(nodes.size(), num_connection_sets);

  std::set<unsigned int> self_recurrent_nodes;

  for(auto& conn : connections) {
    // delay earliest possible zeroing of recurrent connections origins
    // until recurrent connections are applied
    if(conn.type == ConnectionType::Recurrent) {
      earliest_zero_out_indices[conn.origin] = std::max(
        earliest_zero_out_indices[conn.origin],
        conn.set + 1);
    }

    earliest_sigmoid_indices[conn.dest] = std::max(
      earliest_sigmoid_indices[conn.dest],
      conn.set + 1);

    latest_zero_out_indices[conn.dest] = std::min(
      latest_zero_out_indices[conn.dest],
      conn.set);

    if(conn.type == ConnectionType::Normal) {
      latest_sigmoid_indices[conn.origin] = std::min(
        latest_sigmoid_indices[conn.origin],
        conn.set);
    }

    if(conn.origin == conn.dest) {
      self_recurrent_nodes.insert(conn.origin);
    }
  }

  std::vector<unsigned int>& zero_out_indices = earliest_zero_out_indices;
  std::vector<unsigned int>& sigmoid_indices = earliest_sigmoid_indices;


  std::vector<std::vector<unsigned int> > zero_out_sets(num_connection_sets+1);
  std::vector<std::vector<unsigned int> > sigmoid_sets(num_connection_sets+1);

  for(unsigned int i=0; i<nodes.size(); i++) {
    bool is_self_recurrent = self_recurrent_nodes.count(i);
    if(!is_self_recurrent && i >= num_inputs) {
      zero_out_sets[zero_out_indices[i]].push_back(i);
    }
    if(i >= num_inputs) {
      sigmoid_sets[sigmoid_indices[i]].push_back(i);
    }
  }

  auto append_zero_out_set = [&](unsigned int i) {
    auto& zero_out_set = zero_out_sets[i];
    action_list.push_back(zero_out_set.size());
    for(unsigned int j : zero_out_set) {
      action_list.push_back(j);
    }
  };

  auto append_sigmoid_set = [&](unsigned int i) {
    auto& sigmoid_set = sigmoid_sets[i];
    action_list.push_back(sigmoid_set.size());
    for(unsigned int j : sigmoid_set) {
      action_list.push_back(j);
    }
  };



  action_list.clear();
  for(unsigned int i=0; i<num_connection_sets; i++) {
    append_zero_out_set(i);
    append_sigmoid_set(i);
    action_list.push_back(connection_set_sizes[i]);
  }

  append_zero_out_set(num_connection_sets);
  append_sigmoid_set(num_connection_sets);



  // print action list
  // for (auto& item : action_list) {
  //   std::cout << item << " ";
  // } std::cout << std::endl;
}


////////////////////////////////////////////////////////////////////////////

void ConcurrentGPUNeuralNet::add_node(NodeType type, ActivationFunction func) {
  switch (type) {
  case NodeType::Bias:
    num_inputs++;
    nodes.push_back(1.0);
    break;
  case NodeType::Input:
    num_inputs++;
    nodes.push_back(0.0);
    break;
  case NodeType::Output:
    num_outputs++;
    nodes.push_back(0.0);
    break;
  case NodeType::Hidden:
    nodes.push_back(0.0);
    break;
  }

  // Only sigmoid nodes implementated for ConcurrentGPUNeuralNet
  assert(func == ActivationFunction::Sigmoid);

}

_float_ sigmoid(_float_ val) {
  return 1/(1 + std::exp(-val));
}

void clear_nodes(unsigned int* list, _float_* nodes, unsigned int n) {
  for(auto i=0u; i<n; i++) {
    nodes[list[i]] = 0;
  }
}

void sigmoid_nodes(unsigned int* list, _float_* nodes, unsigned int n) {
  for(auto i=0u; i<n; i++) {
    nodes[list[i]] = sigmoid(nodes[list[i]]);
  }
}

void apply_connections(_float_* node, unsigned int* origin, unsigned int* dest, _float_* weight, unsigned int n) {
  for(auto i=0u; i<n; i++) {
    auto& conn_origin = origin[i];
    auto& conn_dest = dest[i];
    auto& conn_weight = weight[i];
    if(conn_origin == conn_dest) {
      // Special case for self-recurrent nodes
      // Be sure not to zero-out before this step.
      node[conn_origin] *= conn_weight;
    } else {
      node[conn_dest] += conn_weight*node[conn_origin];
    }
  }
}

__device__ _float_ device_sigmoid(_float_ val) {
  return 1/(1 + expf(-val));
}

__global__ void device_clear_nodes(unsigned int* list, _float_* nodes, unsigned int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i<n) {
    nodes[list[i]] = 0;
  }
}

__global__ void device_sigmoid_nodes(unsigned int* list, _float_* nodes, unsigned int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i<n) {
    nodes[list[i]] = device_sigmoid(nodes[list[i]]);
  }
}

__global__ void device_apply_connections(_float_* node, unsigned int* origin, unsigned int* dest, _float_* weight, unsigned int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i<n) {
    auto& conn_origin = origin[i];
    auto& conn_dest = dest[i];
    auto& conn_weight = weight[i];
    if(conn_origin == conn_dest) {
      // Special case for self-recurrent nodes
      // Be sure not to zero-out before this step.
      node[conn_origin] *= conn_weight;
    } else {
      node[conn_dest] += conn_weight*node[conn_origin];
    }
  }
}

std::vector<_float_> ConcurrentGPUNeuralNet::host_evaluate(std::vector<_float_> inputs) {
  assert(inputs.size() == num_inputs-1);
  sort_connections();

  // copy inputs in to network
  std::copy(inputs.begin(),inputs.end(),nodes.begin()+1);

  auto i = 0u;
  int how_many_zero_out = action_list[i++];
  clear_nodes(&action_list[i], nodes.data(), how_many_zero_out);
  i += how_many_zero_out;

  int how_many_sigmoid = action_list[i++];
  sigmoid_nodes(&action_list[i], nodes.data(), how_many_sigmoid);
  i += how_many_sigmoid;

  int current_conn = 0;
  while(i<action_list.size()) {
    int how_many_conn = action_list[i++];
    apply_connections(nodes.data(), &connection_list.origin[current_conn], &connection_list.dest[current_conn], &connection_list.weight[current_conn], how_many_conn);
    current_conn += how_many_conn;

    int how_many_zero_out = action_list[i++];
    clear_nodes(&action_list[i], nodes.data(), how_many_zero_out);
    i += how_many_zero_out;

    int how_many_sigmoid = action_list[i++];
    sigmoid_nodes(&action_list[i], nodes.data(), how_many_sigmoid);
    i += how_many_sigmoid;
  }

  return std::vector<_float_> (nodes.begin()+num_inputs,nodes.begin()+num_inputs+num_outputs);
}

std::vector<_float_> ConcurrentGPUNeuralNet::evaluate(std::vector<_float_> inputs) {
  assert(inputs.size() == num_inputs-1);
  sort_connections();
  unsigned int num_blocks = 0;

  // copy inputs in to network
  //std::copy(inputs.begin(),inputs.end(),nodes.begin());
  cuda_assert(cudaMemcpy(&node_[1],inputs.data(),inputs.size()*sizeof(_float_),cudaMemcpyHostToDevice));

  auto i = 0u;
  int how_many_zero_out = action_list[i++];
  num_blocks = (how_many_zero_out+num_threads-1)/num_threads;
  if (how_many_zero_out) { device_clear_nodes<<<num_blocks,num_threads>>>(&action_list_[i], node_, how_many_zero_out); }
  i += how_many_zero_out;

  int how_many_sigmoid = action_list[i++];
  num_blocks = (how_many_sigmoid+num_threads-1)/num_threads;
  if (how_many_sigmoid) { device_sigmoid_nodes<<<num_blocks,num_threads>>>(&action_list_[i], node_, how_many_sigmoid); }
  i += how_many_sigmoid;

  int current_conn = 0;
  while(i<action_list.size()) {
    int how_many_conn = action_list[i++];
    num_blocks = (how_many_conn+num_threads-1)/num_threads;
    if (how_many_conn) { device_apply_connections<<<num_blocks,num_threads>>>(node_, &origin_[current_conn], &dest_[current_conn], &weight_[current_conn], how_many_conn); }
    current_conn += how_many_conn;

    int how_many_zero_out = action_list[i++];
    num_blocks = (how_many_zero_out+num_threads-1)/num_threads;
    if (how_many_zero_out) { device_clear_nodes<<<num_blocks,num_threads>>>(&action_list_[i], node_, how_many_zero_out); }
    i += how_many_zero_out;

    int how_many_sigmoid = action_list[i++];
    num_blocks = (how_many_sigmoid+num_threads-1)/num_threads;
    if (how_many_sigmoid) { device_sigmoid_nodes<<<num_blocks,num_threads>>>(&action_list_[i], node_, how_many_sigmoid); }
    i += how_many_sigmoid;
  }
  cuda_assert(cudaDeviceSynchronize());
  std::vector<_float_> outputs(num_outputs,0);
  cuda_assert(cudaMemcpy(outputs.data(),&node_[num_inputs],num_outputs*sizeof(_float_),cudaMemcpyDeviceToHost));

  return outputs;
}

void ConcurrentGPUNeuralNet::add_connection(int origin, int dest, _float_ weight, unsigned int set) {
  if(would_make_loop(origin,dest,set)) {
    connections.emplace_back(origin,dest,ConnectionType::Recurrent,weight,set);
  } else {
    connections.emplace_back(origin,dest,ConnectionType::Normal,weight,set);
  }
}

bool ConcurrentGPUNeuralNet::would_make_loop(unsigned int i, unsigned int j, unsigned int set) {
  // handle the case of a recurrent connection to itself up front
  if (i == j) { return true; }

  if (set == std::numeric_limits<unsigned int>::max()) {

    std::vector<bool> reachable(nodes.size(), false);
    reachable[j] = true;

    while (true) {

      bool found_new_node = false;
      for (auto const& conn : connections) {
        // if the origin of this connection is reachable and its
        // desitination is not, then it should be made reachable
        if (reachable[conn.origin] &&
                    !reachable[conn.dest] &&
            conn.type == ConnectionType::Normal) {
          // if it is a normal node. if it is the origin of the
          // proposed additional connection (i->j) then it would be
          // a loop
          if (conn.dest == i) {
            // the destination of this reachable connection is
            // the origin of the proposed connection. thus there
            // exists a path from j -> i. So this will be a loop.
            return true;
          }
          else {
            reachable[conn.dest] = true;
            found_new_node = true;
          }
        }
      }
      // no loop detected
      if (!found_new_node) {
        return false;
      }

    }

  } else {
    // if set number is not zero, then it is assumed the added connection is
    // part of a subnet that is currently being added.

    std::map<unsigned int,unsigned int> subset_node_map;
    subset_node_map[i] = subset_node_map.size();
    subset_node_map[j] = subset_node_map.size();

    auto conn_iter = connections.end();
    while (conn_iter-- != connections.begin()) {
      auto conn_set = (*conn_iter).set;
      if (conn_set != set){
        break;
      } else {
        auto origin = (*conn_iter).origin;
        auto dest = (*conn_iter).dest;

        if (subset_node_map.count(origin)==0) {
          subset_node_map[origin] = subset_node_map.size();
        }
        if (subset_node_map.count(dest)==0) {
          subset_node_map[dest] = subset_node_map.size();
        }
      }
    }

    std::vector<bool> reachable(subset_node_map.size(), false);
    reachable[subset_node_map[j]] = true;

    while (true) {
      auto conn_start = conn_iter;

      bool found_new_node = false;
      while (++conn_start != connections.end()) {
        auto const& conn = *conn_start;
        assert(conn.set == set);

        // if the origin of this connection is reachable and its
        // desitination is not, then it should be made reachable
        if (reachable[subset_node_map[conn.origin]] &&
            !reachable[subset_node_map[conn.dest]] &&
            conn.type == ConnectionType::Normal) {
          // if it is a normal node. if it is the origin of the
          // proposed additional connection (i->j) then it would be
          // a loop
          if (conn.dest == i) {
            // the destination of this reachable connection is
            // the origin of the proposed connection. thus there
            // exists a path from j -> i. So this will be a loop.
            return true;
          }
          else {
            reachable[subset_node_map[conn.dest]] = true;
            found_new_node = true;
          }
        }
      }
      // no loop detected
      if (!found_new_node) {
        return false;
      }

    }
  }
}

// TODO: implement gpu_smart_pointer to handle GPU memory according to RAII
void ConcurrentGPUNeuralNet::synchronize() {
  cuda_assert(cudaMalloc((void**)&node_,nodes.size()*sizeof(_float_)));
  cuda_assert(cudaMemcpy(node_,nodes.data(),nodes.size()*sizeof(_float_),cudaMemcpyHostToDevice));

  cuda_assert(cudaMalloc((void**)&origin_,connection_list.origin.size()*sizeof(unsigned int)));
  cuda_assert(cudaMemcpy(origin_,connection_list.origin.data(),connection_list.origin.size()*sizeof(unsigned int),cudaMemcpyHostToDevice));

  cuda_assert(cudaMalloc((void**)&dest_,connection_list.dest.size()*sizeof(unsigned int)));
  cuda_assert(cudaMemcpy(dest_,connection_list.dest.data(),connection_list.dest.size()*sizeof(unsigned int),cudaMemcpyHostToDevice));
  
  cuda_assert(cudaMalloc((void**)&weight_,connection_list.weight.size()*sizeof(_float_)));
  cuda_assert(cudaMemcpy(weight_,connection_list.weight.data(),connection_list.weight.size()*sizeof(_float_),cudaMemcpyHostToDevice));

  cuda_assert(cudaMalloc((void**)&action_list_,action_list.size()*sizeof(unsigned int)));
  cuda_assert(cudaMemcpy(action_list_,action_list.data(),action_list.size()*sizeof(unsigned int),cudaMemcpyHostToDevice));
}


void ConcurrentGPUNeuralNet::print_network(std::ostream& os) const {
  std::stringstream ss; ss.str("");
  ss << "Action List: \n\n";

  auto i = 0u;
  int how_many_zero_out = action_list[i++];
  ss << "# Zero out: " << how_many_zero_out << "\n";
  i += how_many_zero_out;

  int how_many_sigmoid = action_list[i++];
  ss << "# Sigmoid: " << how_many_sigmoid << "\n";
  i += how_many_sigmoid;

  std::vector<unsigned int> num_conn_to_apply;
  int current_conn = 0;
  while(i<action_list.size()) {
    int how_many_conn = action_list[i++];
    ss << "# Connections: " << how_many_conn << "\n";
    current_conn += how_many_conn;
    num_conn_to_apply.push_back(how_many_conn);

    int how_many_zero_out = action_list[i++];
    ss << "# Zero out: " << how_many_zero_out << "\n";
    i += how_many_zero_out;

    int how_many_sigmoid = action_list[i++];
    ss << "# Sigmoid: " << how_many_sigmoid << "\n";
    i += how_many_sigmoid;
  }
  os << ss.str();

  // ss.str("");
  // ss << "\nConnection sets:\n";
  // int counter = 0;
  // int num = num_conn_to_apply[counter];
  // for (auto i=0u; i<connection_list.size(); i++) {
  //   ss << connection_list.origin[i] << " -> " << connection_list.dest[i] << "\n";
  //   if (i == num-1) { num += num_conn_to_apply[++counter]; ss << "\n";}
  // }

  // os << ss.str();
}
