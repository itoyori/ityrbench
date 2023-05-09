#!/bin/bash
[[ -z "${PS1+x}" ]] && set -euo pipefail

MPICC=${MPICC:-mpicc}
MPICXX=${MPICXX:-mpicxx}

$MPICXX --version

if [[ ! -z ${KOCHI_INSTALL_PREFIX_ITOYORI+x} ]]; then
  CXXFLAGS="${CXXFLAGS:+$CXXFLAGS} -DITYR_PROFILER_MODE=$KOCHI_PARAM_PROF"
  CXXFLAGS="${CXXFLAGS:+$CXXFLAGS} -DITYR_ITO_DAG_PROF=$KOCHI_PARAM_DAG_PROF"
  CXXFLAGS="${CXXFLAGS:+$CXXFLAGS} -DITYR_ORI_BLOCK_SIZE=$KOCHI_PARAM_BLOCK_SIZE"

  CMAKE_OPTIONS="${CMAKE_OPTIONS:+$CMAKE_OPTIONS} -Ditoyori_DIR=$KOCHI_INSTALL_PREFIX_ITOYORI/lib/cmake/itoyori"
  CMAKE_OPTIONS="${CMAKE_OPTIONS:+$CMAKE_OPTIONS} -DITYRBENCH_DEFAULT_MEM_MAPPER=$KOCHI_PARAM_DIST_POLICY"
fi
