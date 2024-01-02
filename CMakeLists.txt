cmake_minimum_required(VERSION 3.13)

project(ityrbench VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_VERBOSE_MAKEFILE 1)

# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE RelWithDebInfo)
# set(CMAKE_BUILD_TYPE Debug)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG -g")

set(ITYRBENCH_DEFAULT_MEM_MAPPER default CACHE STRING "Default memory mapper (distribution policy) for Itoyori")

find_package(itoyori REQUIRED)

# just for debugging with UCX logger
if(DEFINED ENV{KOCHI_INSTALL_PREFIX_UCX})
  target_include_directories(itoyori INTERFACE $ENV{KOCHI_INSTALL_PREFIX_UCX}/include)
  target_link_libraries(itoyori INTERFACE $ENV{KOCHI_INSTALL_PREFIX_UCX}/lib/libucs.so)
endif()

add_subdirectory(third-party/backward-cpp)

include(GNUInstallDirs)

function(build_bench_each bench_name config_name)
  set(options)
  set(single_value_args)
  set(multi_value_args FILES INCLUDE_DIRS CXXFLAGS)

  cmake_parse_arguments(PARSE_ARGV 0 ARG "${options}" "${single_value_args}" "${multi_value_args}")

  separate_arguments(policy_options UNIX_COMMAND "${ARG_CXXFLAGS}")

  set(target "${config_name}")

  add_executable(${target}.out "${ARG_FILES}")
  target_link_libraries(${target}.out itoyori)
  target_include_directories(${target}.out PRIVATE ${CMAKE_SOURCE_DIR}/third-party/pcg-cpp/include ${ARG_INCLUDE_DIRS})
  target_compile_options(${target}.out PRIVATE ${policy_options})
  target_link_options(${target}.out PRIVATE ${policy_options})
  add_backward(${target}.out)

  install(TARGETS ${target}.out
          DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/ityrbench")

  add_dependencies(${bench_name} ${target}.out)
endfunction()

function(build_bench_policy bench_name config_name)
  set(options)
  set(single_value_args)
  set(multi_value_args FILES INCLUDE_DIRS CXXFLAGS)

  cmake_parse_arguments(PARSE_ARGV 0 ARG "${options}" "${single_value_args}" "${multi_value_args}")

  build_bench_each(${bench_name} "${config_name}_nocache" FILES ${ARG_FILES} INCLUDE_DIRS ${ARG_INCLUDE_DIRS}
                   CXXFLAGS "${ARG_CXXFLAGS} -DITYR_ORI_CORE=nocache -DITYR_ORI_FORCE_GETPUT=1 -DITYR_RMA_IMPL=mpi")

  build_bench_each(${bench_name} "${config_name}_writethrough" FILES ${ARG_FILES} INCLUDE_DIRS ${ARG_INCLUDE_DIRS}
                   CXXFLAGS "${ARG_CXXFLAGS} -DITYR_ORI_ENABLE_LAZY_RELEASE=0 -DITYR_ORI_ENABLE_WRITE_THROUGH=1")

  build_bench_each(${bench_name} "${config_name}_writeback" FILES ${ARG_FILES} INCLUDE_DIRS ${ARG_INCLUDE_DIRS}
                   CXXFLAGS "${ARG_CXXFLAGS} -DITYR_ORI_ENABLE_LAZY_RELEASE=0")

  build_bench_each(${bench_name} "${config_name}_writeback_lazy" FILES ${ARG_FILES} INCLUDE_DIRS ${ARG_INCLUDE_DIRS}
                   CXXFLAGS "${ARG_CXXFLAGS}")

  build_bench_each(${bench_name} "${config_name}_getput" FILES ${ARG_FILES} INCLUDE_DIRS ${ARG_INCLUDE_DIRS}
                   CXXFLAGS "${ARG_CXXFLAGS} -DITYR_ORI_FORCE_GETPUT=1 -DITYR_ORI_ENABLE_VM_MAP=0")
endfunction()

function(build_bench_sched bench_name config_name)
  set(options)
  set(single_value_args)
  set(multi_value_args FILES INCLUDE_DIRS CXXFLAGS)

  cmake_parse_arguments(PARSE_ARGV 0 ARG "${options}" "${single_value_args}" "${multi_value_args}")

  if(ITYRBENCH_DEFAULT_MEM_MAPPER STREQUAL "default")
    set(default_mem_mapper_randws "cyclic")
    set(default_mem_mapper_adws   "block_adws")
  else()
    set(default_mem_mapper_randws ${ITYRBENCH_DEFAULT_MEM_MAPPER})
    set(default_mem_mapper_adws   ${ITYRBENCH_DEFAULT_MEM_MAPPER})
  endif()

  build_bench_policy(${bench_name} "${config_name}_randws" FILES ${ARG_FILES} INCLUDE_DIRS ${ARG_INCLUDE_DIRS}
                     CXXFLAGS "${ARG_CXXFLAGS} -DITYR_ITO_SCHEDULER=randws -DITYR_ORI_DEFAULT_MEM_MAPPER=${default_mem_mapper_randws}")

  build_bench_policy(${bench_name} "${config_name}_adws" FILES ${ARG_FILES} INCLUDE_DIRS ${ARG_INCLUDE_DIRS}
                     CXXFLAGS "${ARG_CXXFLAGS} -DITYR_ITO_SCHEDULER=adws -DITYR_ORI_DEFAULT_MEM_MAPPER=${default_mem_mapper_adws}")
endfunction()

function(build_bench bench_name config_name)
  set(options)
  set(single_value_args)
  set(multi_value_args FILES INCLUDE_DIRS CXXFLAGS)

  cmake_parse_arguments(PARSE_ARGV 0 ARG "${options}" "${single_value_args}" "${multi_value_args}")

  build_bench_each(${bench_name} "${config_name}_serial" FILES ${ARG_FILES} INCLUDE_DIRS ${ARG_INCLUDE_DIRS}
                   CXXFLAGS "${ARG_CXXFLAGS} -DITYR_ITO_SCHEDULER=serial -DITYR_ORI_CORE=serial")

  build_bench_sched(${bench_name} ${config_name} FILES ${ARG_FILES} INCLUDE_DIRS ${ARG_INCLUDE_DIRS} CXXFLAGS "${ARG_CXXFLAGS}")
endfunction()

add_custom_target(cilksort)
add_custom_target(heat)
add_custom_target(pagerank)
add_custom_target(anns_hnsw)
add_custom_target(anns_hcnng)
add_custom_target(uts)
add_custom_target(uts++)

build_bench(cilksort   cilksort   FILES cilksort.cpp)
build_bench(heat       heat       FILES heat.cpp)
build_bench(pagerank   pagerank   FILES pagerank.cpp)
build_bench(anns_hnsw  anns_hnsw  FILES anns/benchmarks/ANN/HNSW/calc_recall.cpp  INCLUDE_DIRS anns/benchmarks/ANN/bench)
build_bench(anns_hcnng anns_hcnng FILES anns/benchmarks/ANN/bench/neighborsTime.C INCLUDE_DIRS anns/benchmarks/ANN/HCNNG)
build_bench(uts        uts        FILES uts/main.cc   uts/uts.c uts/rng/brg_sha1.c CXXFLAGS -DBRG_RNG=1)
build_bench(uts++      uts++      FILES uts/main++.cc uts/uts.c uts/rng/brg_sha1.c CXXFLAGS -DBRG_RNG=1)
set_source_files_properties(uts/uts.c uts/rng/brg_sha1.c PROPERTIES LANGUAGE CXX)

add_subdirectory(exafmm)
