#include "ReachabilityChecker.hh"
#include "Genome.hh"
#include "NeuralNet.hh"
#include "Population.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_PLUGIN(pyneat) {
  py::module m("pyneat", "C++ implementation of NEAT");

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
    .def("AddNode",(Genome& (Genome::*)(NodeType))&Genome::AddNode)
    .def("AddConnection",&Genome::AddConnection)
    .def("Size",&Genome::Size);

  py::class_<NeuralNet>(m, "NeuralNet")
    .def(py::init<const Genome&>())
    .def("evaluate", &NeuralNet::evaluate)
    .def("num_nodes", &NeuralNet::num_nodes)
    .def("num_connections", &NeuralNet::num_connections);

  return m.ptr();
}
