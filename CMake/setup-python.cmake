###
# Finds the python binaries, libs, include, and site-packages paths
# Ensures that skbuild is installed and its utilities are findable
#
# Calls find_packages to on python interpreter/libraries which defines:
#
#    PYTHON_EXECUTABLE
#    PYTHON_INCLUDE_DIR
#    PYTHON_LIBRARY
#    PYTHON_LIBRARY_DEBUG
#
# Exported variables used by python utility functions are:
#
#    skbuild_location
#      Location of the skbuild library (assumes you have run `pip install scikit-build`)
#
#    skbuild_cmake_dir
#      Location of the skbuild cmake utilities


###
# Private helper function to execute `python -c "<cmd>"`
#
# Runs a python command and populates an outvar with the result of stdout.
# Be careful of indentation if `cmd` is multiline.
#
function(pycmd outvar cmd)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "${cmd}"
    RESULT_VARIABLE _exitcode
    OUTPUT_VARIABLE _output)
  if(NOT ${_exitcode} EQUAL 0)
    message(ERROR "Failed when running python code: \"\"\"
${cmd}\"\"\"")
    message(FATAL_ERROR "Python command failed with error code: ${_exitcode}")
  endif()
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_output}" _output)
  set(${outvar} "${_output}" PARENT_SCOPE)
endfunction()


###
# Find current python major version user option
#

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})


###
# Find scikit-build and include its cmake resource scripts
#
if (NOT SKBUILD)
  pycmd(skbuild_location "import os, skbuild; print(os.path.dirname(skbuild.__file__))")
  set(skbuild_cmake_dir "${skbuild_location}/resources/cmake")
  # If skbuild is not the driver, then we need to include its utilities in our CMAKE_MODULE_PATH
  list(APPEND CMAKE_MODULE_PATH ${skbuild_cmake_dir})
endif()


find_package(PythonExtensions REQUIRED)
