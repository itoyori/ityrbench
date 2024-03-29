cmake_minimum_required(VERSION 3.1)

function(build_exafmm_kernel kernel_name)
  set(options)
  set(single_value_args)
  set(multi_value_args CXXFLAGS)

  cmake_parse_arguments(PARSE_ARGV 0 ARG "${options}" "${single_value_args}" "${multi_value_args}")

  # ./myconfig/config.h was generated by us
  set(EXAFMM_INCLUDE_DIRS ./include ./kernels ./vectorclass ./myconfig)

  build_bench(exafmm exafmm_${kernel_name}
    FILES        examples/fmm.cxx
    INCLUDE_DIRS ${EXAFMM_INCLUDE_DIRS}
    CXXFLAGS     "${ARG_CXXFLAGS} -ffast-math -funroll-loops"
  )
endfunction()

add_custom_target(exafmm)
build_exafmm_kernel(laplace    CXXFLAGS -DEXAFMM_LAPLACE=1)
build_exafmm_kernel(helmholtz  CXXFLAGS -DEXAFMM_HELMHOLTZ=1)
build_exafmm_kernel(biotsavart CXXFLAGS -DEXAFMM_BIOTSAVART=1)
