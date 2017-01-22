#include <memory>
#include <sstream>

#include "ReachabilityChecker.hh"
#include "Genome.hh"
#include "NeuralNet.hh"
#include "Population.hh"
#include "Requirements.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

PYBIND11_PLUGIN(pyneat) {
  py::module m("pyneat", "C++ implementation of NEAT");

  py::class_<Population>(m, "Population")
    .def(py::init<Genome&,std::shared_ptr<RNG>,std::shared_ptr<Probabilities>>())
    .def(py::init<const Population&>())
    .def("Evaluate",
         [](Population& pop, std::function<double(const NeuralNet&)> func) {
           pop.Evaluate(func);
         })
    .def("Reproduce",
         [](Population& pop, std::function<double(const NeuralNet&)> func) {
           return pop.Reproduce(func);
         })
    .def("Reproduce",(Population (Population::*)())&Population::Reproduce)
    .def_property_readonly("organisms", &Population::GetOrganisms,
                           py::return_value_policy::reference_internal);

  py::class_<Population::Organism>(m, "Organism")
    .def_readwrite("fitness", &Population::Organism::fitness)
    .def_readwrite("adj_fitness", &Population::Organism::adj_fitness)
    .def_readwrite("species", &Population::Organism::species)
    .def_readwrite("genome", &Population::Organism::genome)
    .def_readwrite("network", &Population::Organism::network);

  py::class_<RNG, std::shared_ptr<RNG> >(m, "RNG")
    .def("uniform",&RNG::uniform,
         py::arg("min") = 0,
         py::arg("max") = 1)
    .def("gaussian",&RNG::gaussian,
         py::arg("mean") = 0,
         py::arg("sigma") = 1)
    .def("__call__", [](RNG& rng){ return rng(); });

  py::class_<RNG_MersenneTwister, RNG, std::shared_ptr<RNG_MersenneTwister> >(m, "RNG_MersenneTwister")
    .def(py::init<>())
    .def(py::init<unsigned long>(),
         py::arg("seed"));

  py::class_<Probabilities, std::shared_ptr<Probabilities> >(m, "Probabilities")
    .def(py::init<>())
    .def_readwrite("population_size",&Probabilities::population_size)
    .def_readwrite("min_size_for_champion",&Probabilities::min_size_for_champion)
    .def_readwrite("culling_ratio",&Probabilities::culling_ratio)
    .def_readwrite("stale_species_num_generations",&Probabilities::stale_species_num_generations)
    .def_readwrite("matching_gene_choose_mother",&Probabilities::matching_gene_choose_mother)
    .def_readwrite("keep_non_matching_mother_gene",&Probabilities::keep_non_matching_mother_gene)
    .def_readwrite("keep_non_matching_father_gene",&Probabilities::keep_non_matching_father_gene)
    .def_readwrite("mutation_prob_adjust_weights",&Probabilities::mutation_prob_adjust_weights)
    .def_readwrite("weight_mutation_is_severe",&Probabilities::weight_mutation_is_severe)
    .def_readwrite("weight_mutation_small_adjust",&Probabilities::weight_mutation_small_adjust)
    .def_readwrite("weight_mutation_reset_range",&Probabilities::weight_mutation_reset_range)
    .def_readwrite("mutation_prob_add_connection",&Probabilities::mutation_prob_add_connection)
    .def_readwrite("new_connection_is_recurrent",&Probabilities::new_connection_is_recurrent)
    .def_readwrite("mutation_prob_add_node",&Probabilities::mutation_prob_add_node)
    .def_readwrite("mutation_prob_reenable_connection",&Probabilities::mutation_prob_reenable_connection)
    .def_readwrite("mutation_prob_toggle_connection",&Probabilities::mutation_prob_toggle_connection)
    .def_readwrite("genetic_distance_structural",&Probabilities::genetic_distance_structural)
    .def_readwrite("genetic_distance_weights",&Probabilities::genetic_distance_weights)
    .def_readwrite("genetic_distance_species_threshold",&Probabilities::genetic_distance_species_threshold);

  py::class_<ReachabilityChecker>(m, "ReachabilityChecker")
    .def(py::init<size_t>())
    .def(py::init<size_t, size_t>())
    .def("AddConnection",&ReachabilityChecker::AddConnection)
    .def("HasConnection",&ReachabilityChecker::HasConnection)
    .def("HasNormalConnection",&ReachabilityChecker::HasNormalConnection)
    .def("HasRecurrentConnection",&ReachabilityChecker::HasRecurrentConnection)
    .def("IsReachableNormal",&ReachabilityChecker::IsReachableNormal)
    .def("IsReachableEither",&ReachabilityChecker::IsReachableEither)
    .def("IsFullyConnected",&ReachabilityChecker::IsFullyConnected)
    .def("NumPossibleNormalConnections",&ReachabilityChecker::NumPossibleNormalConnections)
    .def("NumPossibleRecurrentConnections",&ReachabilityChecker::NumPossibleRecurrentConnections);

  py::enum_<NodeType>(m, "NodeType")
    .value("Input",NodeType::Input)
    .value("Output",NodeType::Output)
    .value("Hidden",NodeType::Hidden)
    .value("Bias",NodeType::Bias);

  py::class_<Genome>(m, "Genome")
    .def(py::init<>())
    .def("AddNode",&Genome::AddNode)
    .def("AddConnection",&Genome::AddConnection)
    .def("Size",&Genome::Size)
    .def_static("ConnectedSeed", &Genome::ConnectedSeed);

  py::class_<NeuralNet>(m, "NeuralNet")
    .def(py::init<const Genome&>())
    .def("evaluate", &NeuralNet::evaluate)
    .def("num_nodes", &NeuralNet::num_nodes)
    .def("num_connections", &NeuralNet::num_connections)
    .def_property_readonly("node_types", &NeuralNet::node_types)
    .def_property_readonly("connections", &NeuralNet::get_connections,
                           py::return_value_policy::reference_internal);

  py::class_<Connection>(m, "Connection")
    .def_readwrite("origin", &Connection::origin)
    .def_readwrite("dest", &Connection::dest)
    .def_readwrite("weight", &Connection::weight)
    .def_readwrite("type", &Connection::type)
    .def("__repr__",
         [](const Connection& conn) {
           std::stringstream ss;
           ss << "Connection(" << conn.origin << " -> " << conn.dest << ", "
              << conn.weight << ", " << (conn.type==ConnectionType::Normal ? "normal" : "recurrent") << ")";
           return ss.str();
         });

  py::enum_<ConnectionType>(m, "ConnectionType")
    .value("Normal", ConnectionType::Normal)
    .value("Recurrent", ConnectionType::Recurrent);

  return m.ptr();
}
