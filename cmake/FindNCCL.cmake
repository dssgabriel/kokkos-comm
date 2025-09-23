# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# # FindNCCL
#
# Finds the NVIDIA NCCL library.
#
# ## Imported Targets
#
# This module provides the following imported targets, if found:
# - `NCCL::NCCL`: The NCCL library
#
# ## Result Variables
#
# This will define the following variables:
# - `NCCL_FOUND`: True if the system has the NCCL library.
# - `NCCL_VERSION`: The version of the NCCL library which was found.
# - `NCCL_INCLUDE_DIRS`: Include directories needed to use NCCL.
# - `NCCL_LIBRARIES`: Libraries needed to link to NCCL.
#
# ## Cache Variables
#
# The following cache variables may also be set:
# - `NCCL_INCLUDE_DIR`: The directory containing `nccl.h`.
# - `NCCL_LIBRARY`: The path to the NCCL library.

find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_NCCL QUIET NCCL)
endif()

find_path(NCCL_INCLUDE_DIR NAMES nccl.h HINTS ${PC_NCCL_INCLUDE_DIRS} PATH_SUFFIXES nccl)
mark_as_advanced(NCCL_INCLUDE_DIR)

find_library(NCCL_LIBRARY NAMES nccl HINTS ENV ${PC_NCCL_LIBRARY_DIRS})
mark_as_advanced(NCCL_LIBRARY)

if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
  # Extract version components from the header file
  file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" NCCL_VERSION_MAJOR REGEX "^#define[ \t]+NCCL_MAJOR[ \t]+" LIMIT_COUNT 1)
  file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" NCCL_VERSION_MINOR REGEX "^#define[ \t]+NCCL_MINOR[ \t]+" LIMIT_COUNT 1)
  file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" NCCL_VERSION_PATCH REGEX "^#define[ \t]+NCCL_PATCH[ \t]+" LIMIT_COUNT 1)
  file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" NCCL_VERSION_SUFFIX REGEX "^#define[ \t]+NCCL_SUFFIX[ \t]+" LIMIT_COUNT 1)

  # Extract just the values from each line
  if(NCCL_VERSION_MAJOR)
    string(REGEX REPLACE "^#define[ \t]+NCCL_MAJOR[ \t]+(.+)" "\\1" NCCL_VERSION_MAJOR "${NCCL_VERSION_MAJOR}")
    # Remove any template variables like ${nccl:Major}
    string(REGEX REPLACE "\\$\\{[^}]*\\}" "" NCCL_VERSION_MAJOR "${NCCL_VERSION_MAJOR}")
    string(STRIP "${NCCL_VERSION_MAJOR}" NCCL_VERSION_MAJOR)
  endif()

  if(NCCL_VERSION_MINOR)
    string(REGEX REPLACE "^#define[ \t]+NCCL_MINOR[ \t]+(.+)" "\\1" NCCL_VERSION_MINOR "${NCCL_VERSION_MINOR}")
    # Remove any template variables like ${nccl:Minor}
    string(REGEX REPLACE "\\$\\{[^}]*\\}" "" NCCL_VERSION_MINOR "${NCCL_VERSION_MINOR}")
    string(STRIP "${NCCL_VERSION_MINOR}" NCCL_VERSION_MINOR)
  endif()

  if(NCCL_VERSION_PATCH)
    string(REGEX REPLACE "^#define[ \t]+NCCL_PATCH[ \t]+(.+)" "\\1" NCCL_VERSION_PATCH "${NCCL_VERSION_PATCH}")
    # Remove any template variables like ${nccl:Patch}
    string(REGEX REPLACE "\\$\\{[^}]*\\}" "" NCCL_VERSION_PATCH "${NCCL_VERSION_PATCH}")
    string(STRIP "${NCCL_VERSION_PATCH}" NCCL_VERSION_PATCH)
  endif()

  if(NCCL_VERSION_SUFFIX)
    string(REGEX REPLACE "^#define[ \t]+NCCL_SUFFIX[ \t]+(.+)" "\\1" NCCL_VERSION_SUFFIX "${NCCL_VERSION_SUFFIX}")
    # Remove quotes and template variables like "${nccl:Suffix}"
    string(REGEX REPLACE "\"([^\"]*)\"" "\\1" NCCL_VERSION_SUFFIX "${NCCL_VERSION_SUFFIX}")
    string(REGEX REPLACE "\\$\\{[^}]*\\}" "" NCCL_VERSION_SUFFIX "${NCCL_VERSION_SUFFIX}")
    string(STRIP "${NCCL_VERSION_SUFFIX}" NCCL_VERSION_SUFFIX)
  endif()

  # Construct the full version string if we have valid numeric values
  if(
    NCCL_VERSION_MAJOR
      MATCHES
      "^[0-9]+$"
    AND
      NCCL_VERSION_MINOR
        MATCHES
        "^[0-9]+$"
    AND
      NCCL_VERSION_PATCH
        MATCHES
        "^[0-9]+$"
  )
    set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")

    # Append suffix if it exists and is not empty
    if(DEFINED NCCL_VERSION_SUFFIX AND NOT NCCL_VERSION_SUFFIX STREQUAL "")
      set(NCCL_VERSION "${NCCL_VERSION}-${NCCL_VERSION_SUFFIX}")
    endif()

    # Make version string available
    set(NCCL_VERSION_STRING ${NCCL_VERSION})
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NCCL
  FOUND_VAR NCCL_FOUND
  REQUIRED_VARS
    NCCL_LIBRARY
    NCCL_INCLUDE_DIR
  VERSION_VAR NCCL_VERSION
)

if(NCCL_FOUND)
  set(NCCL_LIBRARIES ${NCCL_LIBRARY})
  set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})

  if(NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL UNKNOWN IMPORTED)
    set_target_properties(
      NCCL::NCCL
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
          ${NCCL_INCLUDE_DIRS}
    )

    if(EXISTS "${NCCL_LIBRARY}")
      set_target_properties(
        NCCL::NCCL
        PROPERTIES
          IMPORTED_LINK_INTERFACE_LANGUAGES
            "CXX"
          IMPORTED_LOCATION
            "${NCCL_LIBRARY}"
      )
    endif()
  endif()
endif()
