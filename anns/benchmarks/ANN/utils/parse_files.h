// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <algorithm>
#if 0
#include "common/geometry.h"
#include "common/geometryIO.h"
#endif
#include "common/parse_command_line.h"
#include "types.h"
// #include "common/time_loop.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if 0

using namespace benchIO;
// *************************************************************
// Parsing code (should move to common?)
// *************************************************************

// returns a pointer and a length
std::pair<char*, size_t> mmapStringFromFile(const char* filename) {
  struct stat sb;
  int fd = open(filename, O_RDONLY);
  if (fd == -1) {
    perror("open");
    exit(-1);
  }
  if (fstat(fd, &sb) == -1) {
    perror("fstat");
    exit(-1);
  }
  if (!S_ISREG(sb.st_mode)) {
    perror("not a file\n");
    exit(-1);
  }
  char* p =
      static_cast<char*>(mmap(0, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
  if (p == MAP_FAILED) {
    perror("mmap");
    exit(-1);
  }
  if (close(fd) == -1) {
    perror("close");
    exit(-1);
  }
  size_t n = sb.st_size;
  return std::make_pair(p, n);
}
#endif

// *************************************************************
//  GRAPH TOOLS
// *************************************************************

template<typename TvecPoint>
inline auto add_null_graph(ityr::global_span<TvecPoint> points, int maxDeg){
  ityr::global_vector_options global_vec_coll_opts(true, 1024);

  ityr::global_vector<int> out_nbh(global_vec_coll_opts, maxDeg*points.size(), -1);
  ityr::global_span<int> out_nbh_ref(out_nbh);
  ityr::root_exec([=] {
    ityr::for_each(
        ityr::execution::parallel_policy(1024),
        ityr::make_global_iterator(points.begin(), ityr::checkout_mode::read_write),
        ityr::make_global_iterator(points.end()  , ityr::checkout_mode::read_write),
        ityr::count_iterator<std::size_t>(0),
        [=](TvecPoint& p, std::size_t i) {
          p.out_nbh = out_nbh_ref.subspan(maxDeg*i, maxDeg);
        });
  });
  return out_nbh;
}

#if 0
template<typename T>
int add_saved_graph(parlay::sequence<Tvec_point<T>> &points, const char* gFile){
    /* auto [graphptr, graphlength] = mmapStringFromFile(gFile); */
    auto graphptr_length = mmapStringFromFile(gFile);
    auto& graphptr = std::get<0>(graphptr_length);
    int maxDeg = *((int*)(graphptr+4));
    int num_points = *((int*)graphptr);
    if(num_points != points.size()){
        std::cout << "ERROR: graph file and data file do not match" << std::endl;
        abort();
    }
    parlay::parallel_for(0, points.size(), [&] (size_t i){
        int* start_graph = (int*)(graphptr + 8 + 4*maxDeg*i);
        int* end_graph = start_graph + maxDeg;
        points[i].out_nbh = parlay::make_slice(start_graph, end_graph);
    });
    return maxDeg;
}

//graph file format begins with number of points N, then max degree
//then N+1 offsets indicating beginning and end of each vector
//then the IDs in the vector
//assumes user correctly matches data file and graph file
template<typename T>
void write_graph(parlay::sequence<Tvec_point<T>*> &v, char* outFile, int maxDeg){
  int n = static_cast<int>(v.size());
  std::cout << "Writing graph with " << n << " points and max degree " << maxDeg << std::endl;
  parlay::sequence<int> preamble = {n, maxDeg};
  int* preamble_data = preamble.begin();
  int* graph_data = v[0]->out_nbh.begin();
  std::ofstream writer;
  writer.open(outFile, std::ios::binary | std::ios::out);
  writer.write((char *) preamble_data, 2*sizeof(int));
  writer.write((char *) graph_data, v.size()*maxDeg*sizeof(int));
  writer.close();
}
#endif

// *************************************************************
//  BINARY TOOLS: uint8, int8, float32, int32
// *************************************************************

template <typename TvecPoint>
inline auto parse_bin(const char* filename, const char* gFile, int maxDeg){
  using T = typename TvecPoint::value_type;

  auto ufp = ityr::make_unique_file<char>(filename);
  char* fileptr = ufp.get();

    int num_vectors = *((int*) fileptr);
    int d = *((int*) (fileptr+4));

    if (ityr::is_master()) {
      std::cout << "Detected " << num_vectors << " points with dimension " << d << std::endl;
    }

    ityr::global_vector_options global_vec_coll_opts(true, 1024);

    ityr::global_vector<TvecPoint> points(global_vec_coll_opts, num_vectors);
    ityr::global_span<TvecPoint> points_ref(points);

    ityr::root_exec([=] {
      ityr::transform(
          ityr::execution::parallel_policy(1024),
          ityr::count_iterator<int>(0),
          ityr::count_iterator<int>(num_vectors),
          points_ref.begin(),
          [=](int i) {
            TvecPoint p;
            p.id = i;

            T* start = (T*)(fileptr + 8 + i*d*sizeof(T)); //8 bytes at the start for size + dimension
            T* end = start + d;
            p.coordinates = {start, end};
            return p;
          });
    });

#if 0
    if(maxDeg != 0){
        if(gFile != NULL){int md = add_saved_graph(points, gFile); maxDeg = md;}
        else{add_null_graph(points, maxDeg);}
    }
#else
    if(maxDeg != 0){
      if(gFile != NULL){
        std::cout << "Error: gFile not supported" << std::endl;
        abort();
      }
      ityr::global_vector<int> out_nbh = add_null_graph<TvecPoint>(points, maxDeg);
      return std::make_tuple(std::move(ufp), std::move(out_nbh), maxDeg, std::move(points));
    } else {
      return std::make_tuple(std::move(ufp), ityr::global_vector<int>{}, maxDeg, std::move(points));
    }
#endif
}

// //this is a hack that does some copying, but the sequences are short so it's ok
// //it also won't work if there is overflow but we're doing <1 billion points 
// //please make sure that this is accurate to your specific use case
// auto parse_ibin(const char* filename){
//     auto [fileptr, length] = mmapStringFromFile(filename);

//     int num_vectors = *((int*) fileptr);
//     int d = *((int*) (fileptr+4));

//     std::cout << "Detected " << num_vectors << " points with number of results " << d << std::endl;

//     parlay::sequence<int> &groundtruth = *new parlay::sequence<int>(d*num_vectors);
//     parlay::parallel_for(0, d*num_vectors, [&] (size_t i){
//       long int* p = (long int*)(fileptr+8+8*i);
//       long int j = *(p);
//       groundtruth[i] = static_cast<int>(j);
//     });
//     parlay::sequence<ivec_point> points(num_vectors);

//     parlay::parallel_for(0, num_vectors, [&] (size_t i) {
//         points[i].id = i; 
//         points[i].coordinates = parlay::make_slice(groundtruth.begin()+d*i, groundtruth.begin()+d*(i+1));
//     });

//     return points;
// }

inline auto parse_ibin(const char* filename){
  auto ufp = ityr::make_unique_file<char>(filename);
  char* fileptr = ufp.get();

    int num_vectors = *((int*) fileptr);
    int d = *((int*) (fileptr+4));

    if (ityr::is_master()) {
      std::cout << "Detected " << num_vectors << " points with number of results " << d << std::endl;
    }

    ityr::global_vector_options global_vec_coll_opts(true, 1024);

    ityr::global_vector<ivec_point> points(global_vec_coll_opts, num_vectors);
    ityr::global_span<ivec_point> points_ref(points);

    ityr::root_exec([=] {
      ityr::transform(
          ityr::execution::parallel_policy(1024),
          ityr::count_iterator<int>(0),
          ityr::count_iterator<int>(num_vectors),
          points_ref.begin(),
          [=](int i) {
            ivec_point p;
            p.id = i;

            int* start = (int*)(fileptr + 8 + 4*i*d); //8 bytes at the start for size + dimension
            int* end = start + d;
            float* dist_start = (float*)(fileptr+ 8 + num_vectors*4*d + 4*i*d);
            float* dist_end = dist_start+d; 
            p.coordinates = {start, end};
            p.distances = {dist_start, dist_end};
            return p;
          });
    });

    return std::make_tuple(std::move(ufp), std::move(points));
}

#if 0

// *************************************************************
//  XVECS TOOLS: uint8, float32, int32
// *************************************************************

auto parse_fvecs(const char* filename, const char* gFile, int maxDeg) {
  /* auto [fileptr, length] = mmapStringFromFile(filename); */
  auto fileptr_length = mmapStringFromFile(filename);
  auto& fileptr = std::get<0>(fileptr_length);
  auto& length = std::get<1>(fileptr_length);

  // Each vector is 4 + 4*d bytes.
  // * first 4 bytes encode the dimension (as an integer)
  // * next d values are floats representing vector components
  // See http://corpus-texmex.irisa.fr/ for more details.

  int d = *((int*)fileptr);

  size_t vector_size = 4 + 4*d;
  size_t num_vectors = length / vector_size;
  // std::cout << "Num vectors = " << num_vectors << std::endl;

  parlay::sequence<Tvec_point<float>> points(num_vectors);

  parlay::parallel_for(0, num_vectors, [&] (size_t i) {
    size_t offset_in_bytes = vector_size * i + 4;  // skip dimension
    float* start = (float*)(fileptr + offset_in_bytes);
    float* end = start + d;
    points[i].id = i; 
    points[i].coordinates = parlay::make_slice(start, end);  
  });

  if(maxDeg != 0){
    if(gFile != NULL){int md = add_saved_graph(points, gFile); maxDeg = md;}
    else{add_null_graph(points, maxDeg);}
  }

  return std::make_pair(maxDeg, points);
}

auto parse_bvecs(const char* filename, const char* gFile, int maxDeg) {

  /* auto [fileptr, length] = mmapStringFromFile(filename); */
  auto fileptr_length = mmapStringFromFile(filename);
  auto& fileptr = std::get<0>(fileptr_length);
  auto& length = std::get<1>(fileptr_length);
  // Each vector is 4 + d bytes.
  // * first 4 bytes encode the dimension (as an integer)
  // * next d values are unsigned chars representing vector components
  // See http://corpus-texmex.irisa.fr/ for more details.

  int d = *((int*)fileptr);
  size_t vector_size = 4 + d;
  size_t num_vectors = length / vector_size;

  parlay::sequence<Tvec_point<uint8_t>> points(num_vectors);

  parlay::parallel_for(0, num_vectors, [&] (size_t i) {
    size_t offset_in_bytes = vector_size * i + 4;  // skip dimension
    uint8_t* start = (uint8_t*)(fileptr + offset_in_bytes);
    uint8_t* end = start + d;
    points[i].id = i; 
    points[i].coordinates = parlay::make_slice(start, end);  
  });

  if(maxDeg != 0){
        if(gFile != NULL){int md = add_saved_graph(points, gFile); maxDeg = md;}
        else{add_null_graph(points, maxDeg);}
    }

  return std::make_pair(maxDeg, points);
}

auto parse_ivecs(const char* filename) {
  /* auto [fileptr, length] = mmapStringFromFile(filename); */
  auto fileptr_length = mmapStringFromFile(filename);
  auto& fileptr = std::get<0>(fileptr_length);
  auto& length = std::get<1>(fileptr_length);

  // Each vector is 4 + 4*d bytes.
  // * first 4 bytes encode the dimension (as an integer)
  // * next d values are floats representing vector components
  // See http://corpus-texmex.irisa.fr/ for more details.

  int d = *((int*)fileptr);

  size_t vector_size = 4 + 4*d;
  size_t num_vectors = length / vector_size;  

  parlay::sequence<ivec_point> points(num_vectors);

  parlay::parallel_for(0, num_vectors, [&] (size_t i) {
    size_t offset_in_bytes = vector_size * i + 4;  // skip dimension
    int* start = (int*)(fileptr + offset_in_bytes);
    int* end = start + d;
    points[i].id = i; 
    points[i].coordinates = parlay::make_slice(start, end);
  });

  return points;
}

// *************************************************************
//  RANGERES TOOLS: int32
// *************************************************************

auto parse_rangeres(const char* filename){
    /* auto [fileptr, length] = mmapStringFromFile(filename); */
    auto fileptr_length = mmapStringFromFile(filename);
    auto& fileptr = std::get<0>(fileptr_length);
    int num_points = *((int*) fileptr);
    int num_matches = *((int*) (fileptr+4));
    
    std::cout << "Detected " << num_points << " query points with " << num_matches << " unique matches" << std::endl;
    int* start = (int*)(fileptr+8);
    int* end = start + num_points;
    parlay::slice<int*, int*> num_results = parlay::make_slice(start, end);
    /* auto [offsets, total] = parlay::scan(num_results); */
    auto offsets_total = parlay::scan(num_results);
    auto& offsets = std::get<0>(offsets_total);
    auto& total = std::get<1>(offsets_total);
    offsets.push_back(total);
    parlay::sequence<ivec_point> points(num_points);

    auto id_offset = 4*num_points+8;
    auto dist_offset = id_offset + 4*num_matches; 
    parlay::parallel_for(0, num_points, [&] (size_t i) {
        int* start = (int*)(fileptr + id_offset + 4*offsets[i]); 
        int* end = (int*)(fileptr + id_offset + 4*offsets[i+1]);
        points[i].id = i;
        points[i].coordinates = parlay::make_slice(start, end);
    });
    return points;
}

#endif
