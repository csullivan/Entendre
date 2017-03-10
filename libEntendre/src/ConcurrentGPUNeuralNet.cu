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
    if (abort) exit(code);
  }
//#endif
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
      return EvaluationOrder::NotEqual;
    }
  }

  // else a & b are not adjacent and cannot be compared
  return EvaluationOrder::Unknown;
}

void ConcurrentGPUNeuralNet::sort_connections() {
  if(connections_sorted) {
    return;
  }

  unsigned int max_iterations =
    connections.size()*connections.size()*connections.size()+1;

  bool change_applied = false;
  for(auto i_try=0u; i_try < max_iterations; i_try++) {
    change_applied = false;

    for(auto i=0u; i<connections.size(); i++) {
      for(auto j=i+1; j<connections.size(); j++) {
        Connection& conn1 = connections[i];
        Connection& conn2 = connections[j];

        switch(compare_connections(conn1,conn2)) {
        case EvaluationOrder::GreaterThan:
          if (conn1.set <= conn2.set) {
            conn1.set = conn2.set + 1;
            change_applied = true;
          }
          break;

        case EvaluationOrder::LessThan:
          if(conn2.set <= conn1.set) {
            conn2.set = conn1.set + 1;
            change_applied = true;
          }
          break;

        case EvaluationOrder::NotEqual:
          if(conn1.set == conn2.set) {
            conn2.set = conn1.set + 1;
            change_applied = true;
          }
          break;

        case EvaluationOrder::Unknown:
          break;
        }
      }
    }

    if(!change_applied) {
      break;
    }
  }

  if(change_applied) {
    throw std::runtime_error("Sort Error: change_applied == true on last possible iteration");
  }

  // sort connections based on evaluation set number
  std::sort(connections.begin(),connections.end(),[](Connection a, Connection b){ return a.set < b.set; });
  for (auto const& conn : connections) {
    connection_list.add(conn.origin,conn.dest,conn.weight);
  }
  connections_sorted = true;

  build_action_list();
  connections.clear();
  synchronize();
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

void ConcurrentGPUNeuralNet::add_node(const NodeType& type) {
  switch (type) {
  case NodeType::Bias:
  case NodeType::Input:
    num_inputs++;
    break;
  case NodeType::Output:
    num_outputs++;
    break;
  case NodeType::Hidden:
    break;
  };
  nodes.push_back(0.0);
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

std::vector<_float_> ConcurrentGPUNeuralNet::evaluate(std::vector<_float_> inputs) {
  assert(inputs.size() == num_inputs);
  sort_connections();

  // copy inputs in to network
  std::copy(inputs.begin(),inputs.end(),nodes.begin());

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

std::vector<_float_> ConcurrentGPUNeuralNet::device_evaluate(std::vector<_float_> inputs, unsigned int num_threads) {
  assert(inputs.size() == num_inputs);
  sort_connections();
  unsigned int num_blocks = 0;

  // copy inputs in to network
  //std::copy(inputs.begin(),inputs.end(),nodes.begin());
  cuda_assert(cudaMemcpy(node_,inputs.data(),inputs.size()*sizeof(_float_),cudaMemcpyHostToDevice));

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

void ConcurrentGPUNeuralNet::add_connection(int origin, int dest, _float_ weight) {
  if(would_make_loop(origin,dest)) {
    connections.emplace_back(origin,dest,ConnectionType::Recurrent,weight);
  } else {
    connections.emplace_back(origin,dest,ConnectionType::Normal,weight);
  }
}

bool ConcurrentGPUNeuralNet::would_make_loop(unsigned int i, unsigned int j) {
  // handle the case of a recurrent connection to itself up front
  if (i == j) { return true; }

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
  std::stringstream ss;
  ss << "Action List: \n\n";

  auto i = 0u;
  int how_many_zero_out = action_list[i++];
  ss << "# Zero out: " << how_many_zero_out << "\n";
  i += how_many_zero_out;

  int how_many_sigmoid = action_list[i++];
  ss << "# Sigmoid: " << how_many_sigmoid << "\n";
  i += how_many_sigmoid;

  int current_conn = 0;
  while(i<action_list.size()) {
    int how_many_conn = action_list[i++];
    ss << "# Connections: " << how_many_conn << "\n";
    current_conn += how_many_conn;

    int how_many_zero_out = action_list[i++];
    ss << "# Zero out: " << how_many_zero_out << "\n";
    i += how_many_zero_out;

    int how_many_sigmoid = action_list[i++];
    ss << "# Sigmoid: " << how_many_sigmoid << "\n";
    i += how_many_sigmoid;
  }
  os << ss.str();
}