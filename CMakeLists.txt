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

find_package(itoyori REQUIRED)

include(GNUInstallDirs)

set(benchmarks cilksort)

foreach(bench IN LISTS benchmarks)
  add_executable(${bench}.out ${bench}.cpp)
  target_link_libraries(${bench}.out itoyori)
  target_include_directories(${bench}.out PRIVATE pcg-cpp/include)

  add_executable(${bench}_serial.out ${bench}.cpp)
  target_link_libraries(${bench}_serial.out itoyori)
  target_include_directories(${bench}_serial.out PRIVATE pcg-cpp/include)
  target_compile_options(${bench}_serial.out PRIVATE -DITYR_ITO_SCHEDULER=serial)

  add_executable(${bench}_prof_stats.out ${bench}.cpp)
  target_link_libraries(${bench}_prof_stats.out itoyori)
  target_include_directories(${bench}_prof_stats.out PRIVATE pcg-cpp/include)
  target_compile_options(${bench}_prof_stats.out PRIVATE -DITYR_PROFILER_MODE=stats)

  add_executable(${bench}_prof_trace.out ${bench}.cpp)
  target_link_libraries(${bench}_prof_trace.out itoyori)
  target_include_directories(${bench}_prof_trace.out PRIVATE pcg-cpp/include)
  target_compile_options(${bench}_prof_trace.out PRIVATE -DITYR_PROFILER_MODE=trace)

  install(TARGETS ${bench}.out
                  ${bench}_serial.out
                  ${bench}_prof_stats.out
                  ${bench}_prof_trace.out
          DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/ityrbench")
endforeach()
