SHELL=/bin/bash
.SHELLFLAGS += -eu -o pipefail -c
.ONESHELL:

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

UTH_PATH     := ${KOCHI_INSTALL_PREFIX_MASSIVETHREADS_DM}
UTH_CXXFLAGS := -I$(UTH_PATH)/include -fno-stack-protector -Wno-register
UTH_LDFLAGS  := -L$(UTH_PATH)/lib
UTH_LIBS     := -luth -lmcomm

PCAS_PATH     := ${KOCHI_INSTALL_PREFIX_PCAS}
PCAS_CXXFLAGS := -I$(PCAS_PATH)/include
PCAS_LDFLAGS  :=
PCAS_LIBS     := -lrt

LIBUNWIND_PATH     := ${KOCHI_INSTALL_PREFIX_LIBUNWIND}
LIBUNWIND_CXXFLAGS := -I$(LIBUNWIND_PATH)/include
LIBUNWIND_LDFLAGS  := -L$(LIBUNWIND_PATH)/lib -Wl,-R$(LIBUNWIND_PATH)/lib
LIBUNWIND_LIBS     := -lunwind

BACKWARD_PATH     := ${KOCHI_INSTALL_PREFIX_BACKWARD_CPP}
BACKWARD_CXXFLAGS := -I$(BACKWARD_PATH)/include
BACKWARD_LDFLAGS  :=
BACKWARD_LIBS     := -lbfd

PCG_PATH     := ${KOCHI_INSTALL_PREFIX_PCG}
PCG_CXXFLAGS := -I$(PCG_PATH)/include
PCG_LDFLAGS  :=
PCG_LIBS     :=

# TODO: remove it when boost dependency is removed
BOOST_PATH     := ${KOCHI_INSTALL_PREFIX_BOOST}
BOOST_CXXFLAGS := -I$(BOOST_PATH)/include
BOOST_LDFLAGS  := -L$(BOOST_PATH)/lib -Wl,-R$(BOOST_PATH)/lib
BOOST_LIBS     := -lboost_container

CXXFLAGS := $(UTH_CXXFLAGS) $(PCAS_CXXFLAGS) $(LIBUNWIND_CXXFLAGS) $(BACKWARD_CXXFLAGS) $(PCG_CXXFLAGS) $(BOOST_CXXFLAGS) -I. -std=c++17 -O3 -g -gdwarf-4 -Wall $(CXXFLAGS) $(CFLAGS)
LDFLAGS  := $(UTH_LDFLAGS) $(PCAS_LDFLAGS) $(LIBUNWIND_LDFLAGS) $(BACKWARD_LDFLAGS) $(PCG_LDFLAGS) $(BOOST_LDFLAGS) -Wl,-export-dynamic
LIBS     := $(UTH_LIBS) $(PCAS_LIBS) $(LIBUNWIND_LIBS) $(BACKWARD_LIBS) $(PCG_LIBS) $(BOOST_LIBS) -lpthread -lm -ldl

MPICXX := $(or ${MPICXX},mpicxx)

SRCS := $(wildcard ./*.cpp)
HEADERS := $(wildcard ./ityr/**/*.hpp)

MAIN_TARGETS := $(patsubst %.cpp,%.out,$(SRCS)) uts.out uts++.out

all: $(MAIN_TARGETS) exafmm

%.out: %.cpp $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

uts.out: uts/uts.c uts/rng/brg_sha1.c uts/main.cc $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -DBRG_RNG=1 -o $@ uts/uts.c uts/rng/brg_sha1.c uts/main.cc $(LDFLAGS) $(LIBS)

uts++.out: uts/uts.c uts/rng/brg_sha1.c uts/main++.cc $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -DBRG_RNG=1 -o $@ uts/uts.c uts/rng/brg_sha1.c uts/main++.cc $(LDFLAGS) $(LIBS)

.PHONY: exafmm
exafmm:
	cd exafmm
	[ -f Makefile ] || ./configure --disable-simd --enable-mpi CXXFLAGS="-I$(THIS_DIR) $(CXXFLAGS)" LDFLAGS="$(LDFLAGS)" LIBS="$(LIBS)"
	make -j

clean:
	rm -rf $(MAIN_TARGETS)
	[ ! -f exafmm/Makefile ] || make clean -C exafmm

distclean:
	rm -rf $(MAIN_TARGETS)
	[ ! -f exafmm/Makefile ] || make distclean -C exafmm
