#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>

#include <extrinsics_calibrator/CheckerboardExtrinsicCalibration.h>

namespace py = pybind11;

PYBIND11_MODULE(ExtrinsicCalibPyModules, m) {
  m.doc() = "Python Wrapper for Extrinsics Calibrator using GTSAM";  // optional module

  pybind11::class_<CheckerboardExtrinsicCalibration>(m, "CheckerboardExtrinsicCalibration")
      .def(pybind11::init<std::string>())
      .def("add_measurement", &CheckerboardExtrinsicCalibration::addMeasurement)
      .def("solve", &CheckerboardExtrinsicCalibration::solve);
}