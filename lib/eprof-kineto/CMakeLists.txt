# Vendored libkineto, plus the one dependency it does not carry.
#
# Driven by build.rs, not by a project-wide cmake tree: libkineto belongs to
# this crate, so building it is this crate's business. Everything variable --
# the compiler, CUPTI's location, which backend to build for -- arrives as a
# cache entry from build.rs, which asked the environment for it.
cmake_minimum_required(VERSION 3.25)
project(eprof_kineto LANGUAGES CXX C)

# libkineto links dynolog_ipcfabric_lib and includes FabricManager.h from it,
# but does not add the directory itself -- it expects its embedder to.
if(NOT DEFINED EPROF_THIRD_PARTY)
  message(FATAL_ERROR "EPROF_THIRD_PARTY must point at the repository's third_party/")
endif()
set(DYNOLOG_INCLUDE_DIR "${EPROF_THIRD_PARTY}/dynolog")
set(IPCFABRIC_INCLUDE_DIR "${DYNOLOG_INCLUDE_DIR}/dynolog/src/ipcfabric/")
if(NOT EXISTS "${IPCFABRIC_INCLUDE_DIR}/FabricManager.h")
  message(FATAL_ERROR
    "dynolog is missing from ${IPCFABRIC_INCLUDE_DIR}. Run:\n"
    "  git submodule update --init --recursive")
endif()
add_subdirectory("${IPCFABRIC_INCLUDE_DIR}" ipcfabric)

add_subdirectory(libkineto)
# The archive is linked into a shared object, so every object in it must be.
set_property(TARGET kineto PROPERTY POSITION_INDEPENDENT_CODE ON)
