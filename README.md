# ityrbench

```sh
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Ditoyori_DIR=$ITOYORI_HOME/lib/cmake/itoyori ..
make -j
```

## List of Itoyori Benchmarks

- Cilksort
    - config: `cilksort.yaml`
    - code: `cilksort.cpp`
    - source: Cilk v5.4.6 examples
- Heat
    - config: `heat.yaml`
    - code: `heat.cpp`
    - source: Cilk v5.4.6 examples
- ExaFMM
    - config: `exafmm.yaml`
    - code: `exafmm/`
    - source: https://github.com/exafmm/exafmm-beta
- UTS/UTS++ (UTS-Mem)
    - config: `uts.yaml`, `uts++.yaml`
    - code: `uts/main.cc`, `uts/main++.cc`
    - source: https://sourceforge.net/p/uts-benchmark/wiki/Home/
- ANNS Graph Construction (HNSW and HCNNG)
    - config: `anns_hnsw.yaml`, `anns_hcnng.yaml`
    - code: `anns/`
    - source: https://github.com/cmuparlay/pbbsbench-vldb2024
- PageRank
    - config: `pagerank.yaml`
    - code: `pagerank.cpp`
    - source: https://github.com/souravpati/GPOP
