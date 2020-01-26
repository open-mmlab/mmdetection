###
# CMakeLists for mmdet C/C++/Cuda Python extention modules
#
# Notes:
# https://github.com/Erotemic/netharn/issues/7
# https://github.com/amueller/word_cloud/pull/42ci

cmake_minimum_required(VERSION 3.13.0)

project(mmdet LANGUAGES C CXX)

option(USE_CUDA "Build with CUDA" TRUE)
option(MMDET_VERBOSE "Print extra info" FALSE)


if (USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    elseif()
        message(STATUS "CUDA not found, CUDA library will not be built")
        set(USE_CUDA OFF)
    endif()
endif()


# Setup basic python stuff and ensure we have skbuild
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/CMake")
include( setup-python )


###
# Status string for debugging
#
set(PYTHON_SETUP_STATUS "
  * PYTHON_EXECUTABLE = \"${PYTHON_EXECUTABLE}\"

  * PYTHON_INCLUDE_DIR = \"${PYTHON_INCLUDE_DIR}\"
  * PYTHON_LIBRARY = \"${PYTHON_LIBRARY}\"
  * PYTHON_LIBRARY_DEBUG = \"${PYTHON_LIBRARY_DEBUG}\"

  * skbuild_location = \"${skbuild_location}\"
  * skbuild_cmake_dir = \"${skbuild_cmake_dir}\"
")

# Ensure we can find Torch
#~/.local/conda/envs/py38/lib/python3.8/site-packages/torch/share/cmake/Torch/TorchConfig.cmake
pycmd(torch_location "import os, torch; print(os.path.dirname(torch.__file__))")
set(torch_cmake_dir "${torch_location}/share/cmake/Torch")
set(Torch_DIR "${torch_cmake_dir}")


#find_package(pybind11 REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(Torch REQUIRED)

# sudo apt install pybind11-dev
#FIND_PACKAGE( pybind11 CONFIG REQUIRED )
#message(STATUS "pybind11_FOUND = ${pybind11_FOUND}")
#message(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
#message(STATUS "pybind11_DIR = ${pybind11_DIR}")


function(make_cython_ext module_name dpath rel_sources)

  list(APPEND cython_sources "")
  foreach(src IN LISTS rel_sources)
    list(APPEND cython_sources "${dpath}/${src}")
  endforeach()

  # hack: make a cmake lists so scikit-build doesn't complain
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${dpath}/src")
  file(TOUCH "${dpath}/src/CMakeLists.txt")
  add_subdirectory(${dpath}/src)

  # Translate Cython into C/C++
  add_cython_target(${module_name} "${cython_sources}" C OUTPUT_VAR sources)
  # Create C++ library. Specify include dirs and link libs as normal
  add_library(${module_name} MODULE ${sources})
  # Ensure the library is output in the specified directory
  set_target_properties(${module_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${dpath})
  target_include_directories(
    ${module_name}
    PUBLIC
      ${NumPy_INCLUDE_DIRS}
      #${PYTHON_INCLUDE_DIRS}
  )
  #target_link_libraries(${module_name} ${PYTHON_LIBRARIES})
  #${PYTHON_LIBRARIES})
  target_compile_options(${module_name} PUBLIC "-Wno-unused-function" "-Wno-write-strings")
  target_compile_definitions(${module_name} PUBLIC "NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")

  # Transform the C++ library into an importable python module
  python_extension_module(${module_name})

  # Install the C++ module to the correct relative location
  # (this will be an inplace build if you use `pip install -e`)
  #file(RELATIVE_PATH _install_dest "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  #install(TARGETS ${module_name} LIBRARY DESTINATION "${_install_dest}")
  install(TARGETS ${module_name} LIBRARY DESTINATION "${dpath}")
endfunction(make_cython_ext)


function(make_cuda_ext module_name dpath rel_sources )

  list(APPEND sources "")
  foreach(src IN LISTS rel_sources)
    list(APPEND sources "${dpath}/${src}")
  endforeach()

  add_library(${module_name} MODULE ${sources})
  #pybind11_add_module(${module_name} MODULE ${sources})

  # Ensure the library is output in the specified directory
  set_target_properties(${module_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${dpath})

  target_include_directories(${module_name} PUBLIC
    ${TORCH_INCLUDE_DIRS}
    #${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    #${NumPy_INCLUDE_DIRS}
    #${PYTHON_INCLUDE_DIRS}
    #${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(${module_name} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
  # ${PYTHON_LIBRARIES}

  target_compile_definitions(${module_name}
    PUBLIC "TORCH_EXTENSION_NAME=${module_name}"  # Extremely important. Causes dynamic module does not define module export function otherwise
    PUBLIC "WITH_CUDA=1"
    PUBLIC "__CUDA_NO_HALF_OPERATORS__"
    PUBLIC "__CUDA_NO_HALF_CONVERSIONS__"
    PUBLIC "__CUDA_NO_HALF2_OPERATORS__"
  )

  target_compile_definitions(${module_name} PUBLIC "NPY_NO_DEPRECATED_API")

  # Required for pybind11 when building STATIC libraries.
  # It's ON by default with SHARED or MODULE libraries.
  #set_target_properties(${module_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  #set_target_properties(${module_name} PROPERTIES CXX_VISIBILITY_PRESET "hidden")
  #set_target_properties(${module_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")

  # Transform the C++ library into an importable python module
  #python_extension_module(${module_name})

  # Install the C++ module to the correct relative location
  # (this will be an inplace build if you use `pip install -e`)
  #file(RELATIVE_PATH _install_dest "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  install(TARGETS ${module_name} LIBRARY DESTINATION "${dpath}")

endfunction(make_cuda_ext)


make_cuda_ext("compiling_info" "mmdet/ops/utils"
  "src/compiling_info.cpp")

make_cython_ext("soft_nms_cpu" "mmdet/ops/nms"
  "src/soft_nms_cpu.pyx")

make_cuda_ext("nms_cpu" "mmdet/ops/nms"
  "src/nms_cpu.cpp")

make_cuda_ext("nms_cuda" "mmdet/ops/nms"
  "src/nms_cuda.cpp;src/nms_kernel.cu")

make_cuda_ext("roi_align_cuda" "mmdet/ops/roi_align"
  "src/roi_align_cuda.cpp;src/roi_align_kernel.cu")

make_cuda_ext("roi_pool_cuda" "mmdet/ops/roi_pool"
  "src/roi_pool_cuda.cpp;src/roi_pool_kernel.cu")

make_cuda_ext("deform_conv_cuda" "mmdet/ops/dcn"
  "src/deform_conv_cuda.cpp;src/deform_conv_cuda_kernel.cu")

make_cuda_ext("deform_pool_cuda" "mmdet/ops/dcn"
  "src/deform_pool_cuda.cpp;src/deform_pool_cuda_kernel.cu")

make_cuda_ext("sigmoid_focal_loss_cuda" "mmdet/ops/sigmoid_focal_loss"
  "src/sigmoid_focal_loss.cpp;src/sigmoid_focal_loss_cuda.cu")

make_cuda_ext("masked_conv2d_cuda" "mmdet/ops/masked_conv"
  "src/masked_conv2d_cuda.cpp;src/masked_conv2d_kernel.cu")


set(MMDET_CONFIG_STATUS "
PYTHON_CONFIG_STATUS
====================

Include Dirs:
  * CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES = \"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}\"
  * NumPy_INCLUDE_DIRS = \"${NumPy_INCLUDE_DIRS}\"
  * PYTHON_INCLUDE_DIRS = \"${PYTHON_INCLUDE_DIRS}\"

status(Torch):
  * TORCH_FOUND = \"${TORCH_FOUND}\"
  * TORCH_INCLUDE_DIRS = \"${TORCH_INCLUDE_DIRS}\"
  * TORCH_LIBRARIES = \"${TORCH_LIBRARIES}\"
  * TORCH_CXX_FLAGS = \"${TORCH_CXX_FLAGS}\"

status(Cython):
  * CYTHON_FOUND = \"${CYTHON_FOUND}\"
  * CYTHON_EXECUTABLE = \"${CYTHON_EXECUTABLE}\"
  * CYTHON_VERSION = \"${CYTHON_VERSION}\"

status(NumPy):
  * NumPy_FOUND = \"${NumPy_FOUND}\"
  * NumPy_INCLUDE_DIRS = \"${NumPy_INCLUDE_DIRS}\"
  * NumPy_VERSION = \"${NumPy_VERSION}\"
  * NumPy_CONV_TEMPLATE_EXECUTABLE = \"${NumPy_CONV_TEMPLATE_EXECUTABLE}\"
  * NumPy_FROM_TEMPLATE_EXECUTABLE = \"${NumPy_FROM_TEMPLATE_EXECUTABLE}\"

status(PythonExtensions):
  * PYTHON_PREFIX = \"${PYTHON_PREFIX}\"
  * PYTHON_SITE_PACKAGES_DIR = \"${PYTHON_SITE_PACKAGES_DIR}\"
  * PYTHON_RELATIVE_SITE_PACKAGES_DIR = \"${PYTHON_RELATIVE_SITE_PACKAGES_DIR}\"
  * PYTHON_SEPARATOR = \"${PYTHON_SEPARATOR}\"
  * PYTHON_PATH_SEPARATOR = \"${PYTHON_PATH_SEPARATOR}\"
  * PYTHON_EXTENSION_MODULE_SUFFIX = \"${PYTHON_EXTENSION_MODULE_SUFFIX}\"

status(python-setup)
${PYTHON_SETUP_STATUS}
")

message(STATUS "MMDET_VERBOSE = ${MMDET_VERBOSE}")
if (MMDET_VERBOSE)
  message(STATUS ${MMDET_CONFIG_STATUS})
endif ()
