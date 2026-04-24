#include <pybind11/pybind11.h>
#include "mc_kernel.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_fastmc, m) {
    m.doc() = "fastmc — C++ Monte Carlo kernel (pybind11 + OpenMP)";

    py::class_<fastmc::MCResult>(m, "MCResult")
        .def_readonly("price", &fastmc::MCResult::price)
        .def_readonly("stderr", &fastmc::MCResult::stderr)
        .def("__repr__", [](const fastmc::MCResult& r) {
            return "<MCResult price=" + std::to_string(r.price) +
                   " stderr=" + std::to_string(r.stderr) + ">";
        });

    m.def("mc_euro_call", &fastmc::mc_euro_call,
          py::arg("S0"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"),
          py::arg("n_paths"), py::arg("n_steps"),
          py::arg("seed") = 42,
          py::arg("n_threads") = 0,
          "European call MC under GBM (antithetic).");
}
