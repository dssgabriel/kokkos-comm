# Find NVIDIA NCCL
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#
# Adapted from PyTorch: https://github.com/pytorch/pytorch/blob/main/cmake/Modules/FindNCCL.cmake

set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "NVIDIA NCCL headers directory")
set(NCCL_LIB_DIR $ENV{NCCL_LIB_DIR} CACHE PATH "NVIDIA NCCL library directory")
set(NCCL_VERSION $ENV{NCCL_VERSION} CACHE STRING "NVIDIA NCCL version")

if($ENV{NCCL_ROOT_DIR})
  message(WARNING "`NCCL_ROOT_DIR` is deprecated. Please set `NCCL_ROOT` instead.")
endif()
list(
  APPEND
  NCCL_ROOT
  $ENV{NCCL_ROOT_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR}
)
# Compatible layer for CMake <3.12.
# `NCCL_ROOT` will be accounted in when searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${NCCL_ROOT})

find_path(NCCL_INCLUDE_DIRS NAMES nccl.h HINTS ${NCCL_INCLUDE_DIR})

set(NCCL_LIBNAME "nccl")
# Prefer the versioned library if a specific NCCL version is specified
if(NCCL_VERSION)
  set(
    CMAKE_FIND_LIBRARY_SUFFIXES
    ".so.${NCCL_VERSION}"
    ${CMAKE_FIND_LIBRARY_SUFFIXES}
  )
endif()

find_library(NCCL_LIBRARIES NAMES ${NCCL_LIBNAME} HINTS ${NCCL_LIB_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NCCL
  DEFAULT_MSG
  NCCL_INCLUDE_DIRS
  NCCL_LIBRARIES
)

# Determining NCCL version and sanity checks
if(NCCL_FOUND)
  set(NCCL_HEADER_FILE "${NCCL_INCLUDE_DIRS}/nccl.h")
  message(STATUS "Determining NCCL version from ${NCCL_HEADER_FILE}...")
  set(OLD_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
  list(APPEND CMAKE_REQUIRED_INCLUDES ${NCCL_INCLUDE_DIRS})

  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(
    NCCL_VERSION_CODE
    nccl.h
    NCCL_VERSION_DEFINED
  )

  if(NCCL_VERSION_DEFINED)
    set(file "${PROJECT_BINARY_DIR}/detect_nccl_version.cc")
    file(
      WRITE
      ${file}
      "
      #include <iostream>
      #include <nccl.h>
      int main() {
        std::cout << NCCL_MAJOR << '.' << NCCL_MINOR << '.' << NCCL_PATCH << '\n';
        int x;
        ncclGetVersion(&x);
        return x == NCCL_VERSION_CODE;
      }
"
    )
    try_run(
      NCCL_VERSION_MATCHED
      compile_result
      ${PROJECT_BINARY_DIR}
      ${file}
      RUN_OUTPUT_VARIABLE NCCL_VERSION_FROM_HEADER
      CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES=${NCCL_INCLUDE_DIRS}"
      LINK_LIBRARIES
        ${NCCL_LIBRARIES}
    )
    if(NOT NCCL_VERSION_MATCHED)
      message(
        FATAL_ERROR
        "Found NCCL version does not match:\n\
\tENV NCCL_VERSION:     ${NCCL_VERSION}\n\
\tNCCL_VERSION_CODE:    ${NCCL_VERSION_CODE}\n\
\tNCCL_VERSION_DEFINED: ${NCCL_VERSION_DEFINED}\n\
\tNCCL_VERSION_MATCHED: ${NCCL_VERSION_MATCHED}\n\
Please set `NCCL_INCLUDE_DIR` and `NCCL_LIBRARY` manually (include: ${NCCL_INCLUDE_DIRS}, lib: ${NCCL_LIBRARIES})."
      )
    endif()
    message(STATUS "NCCL version: ${NCCL_VERSION_FROM_HEADER}")
  else()
    message(STATUS "NCCL version < 2.3.5-5")
  endif()
  set(CMAKE_REQUIRED_INCLUDES ${OLD_CMAKE_REQUIRED_INCLUDES})

  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(
    NCCL_ROOT_DIR
    NCCL_INCLUDE_DIRS
    NCCL_LIBRARIES
  )
endif()

# -------------------------------------------------------------------------------------------------------------------- #
# FIXME: REWRITE ATTEMPT BELOW
# -------------------------------------------------------------------------------------------------------------------- #
# find_path(NCCL_INCLUDE_DIR NAMES nccl.h HINTS ENV ${NCCL_INCLUDE_DIRS})
# mark_as_advanced(NCCL_INCLUDE_DIR)

# find_library(
#   NCCL_LIBRARY
#   NAMES
#     nccl
#     libnccl
#   HINTS
#   ENV ${NCCL_LIBRARY_DIRS}
# )
# mark_as_advanced(NCCL_LIBRARY)

# if(NOT $ENV{NCCL_VERSION})
#   set(NCCL_VERSION $ENV{NCCL_VERSION})
#   mark_as_advanced(NCCL_VERSION)
# endif()

# include(FindPackageHandleStandardArgs)
# find_package_handle_standard_args(
#   NCCL
#   FOUND_VAR NCCL_FOUND
#   REQUIRED_VARS
#     NCCL_LIBRARY
#     NCCL_INCLUDE_DIR
#   VERSION_VAR NCCL_VERSION
# )

# if(NCCL_FOUND)
#   set(NCCL_LIBRARIES ${NCCL_LIBRARY})
#   set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})

#   if(NOT TARGET nccl::nccl)
#     add_library(nccl::nccl UNKNOWN IMPORTED)
#     set_target_properties(
#       nccl::nccl
#       PROPERTIES
#         INTERFACE_INCLUDE_DIRECTORIES
#           ${NCCL_INCLUDE_DIRS}
#     )

#     if(EXISTS "${NCCL_LIBRARY}")
#       set_target_properties(
#         nccl::nccl
#         PROPERTIES
#           IMPORTED_LINK_INTERFACE_LANGUAGES
#             "CXX"
#           IMPORTED_LOCATION
#             "${NCCL_LIBRARY}"
#       )
#     endif()
#   endif()
# endif()
