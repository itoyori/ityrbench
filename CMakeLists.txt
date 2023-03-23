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

set(policies "\
serial        , -DITYR_ITO_SCHEDULER=serial -DITYR_ORI_CORE=serial;\
nocache       , -DITYR_ORI_CORE=nocache -DITYR_ORI_FORCE_GETPUT=1;\
writethrough  , -DITYR_ORI_ENABLE_LAZY_RELEASE=0 -DITYR_ORI_ENABLE_WRITE_THROUGH=1;\
writeback     , -DITYR_ORI_ENABLE_LAZY_RELEASE=0;\
writeback_lazy, ;\
getput        , -DITYR_ORI_FORCE_GETPUT=1 -DITYR_ORI_ENABLE_VM_MAP=0")

foreach(bench IN LISTS benchmarks)
  foreach(policy_tuple IN LISTS policies)
    string(REPLACE "," ";" pair ${policy_tuple})
    list(GET pair 0 policy_name)
    list(GET pair 1 policy_cxxflags)

    string(STRIP "${policy_name}" policy_name)
    separate_arguments(policy_options UNIX_COMMAND "${policy_cxxflags}")

    set(target "${bench}_${policy_name}")

    add_executable(${target}.out ${bench}.cpp)
    target_link_libraries(${target}.out itoyori)
    target_include_directories(${target}.out PRIVATE pcg-cpp/include)
    target_compile_options(${target}.out PRIVATE ${policy_options})

    install(TARGETS ${target}.out
            DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/ityrbench")
  endforeach()
endforeach()
