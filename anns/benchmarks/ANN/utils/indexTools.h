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

#ifndef INDEXTOOLS
#define INDEXTOOLS

#include <algorithm>
#if 0
#include "common/geometry.h"
#include <random>
#endif
#include "../utils/types.h"

//special size function
template<typename T>
int size_of(ityr::global_span<T> nbh){
  auto nbh_cs = ityr::make_checkout(nbh, ityr::checkout_mode::read);
	int size = 0;
        std::size_t i=0;
	while(i<nbh_cs.size() && nbh_cs[i] != -1) {size++; i++;}
	return size;
}

//adding more neighbors
template<typename T>
void add_nbh(int nbh, Tvec_point<T>& p){
	if((size_t)size_of(p.out_nbh) >= p.out_nbh.size()){
		std::cout << "error: tried to exceed degree bound " << p.out_nbh.size() << std::endl;
		abort();
	}
	p.out_nbh[size_of(p.out_nbh)].put(nbh);
}

template<typename T>
void add_out_nbh(const std::vector<int>& nbh, Tvec_point<T>& p){
  if (nbh.size() > p.out_nbh.size()) {
    std::cout << "oversize" << std::endl;
    abort();
  }
  auto out_nbh_cs = ityr::make_checkout(p.out_nbh, ityr::checkout_mode::write);
	for(std::size_t i=0; i<out_nbh_cs.size(); i++){
		out_nbh_cs[i] = -1;
	}
	for(std::size_t i=0; i<nbh.size(); i++){
		out_nbh_cs[i] = nbh[i];
	}
}

#if 0
template<typename T>
void add_new_nbh(parlay::sequence<int> nbh, Tvec_point<T> *p){
  if (nbh.size() > p->new_nbh.size()) {
    std::cout << "oversize" << std::endl;
    abort();
  }
	for(int i=0; i<p->new_nbh.size(); i++){
		p->new_nbh[i] = -1;
	}
	for(int i=0; i<nbh.size(); i++){
		p->new_nbh[i] = nbh[i];
	}
}

template<typename T>
void synchronize(Tvec_point<T> *p){
	std::vector<int> container = std::vector<int>();
	for(int j=0; j<p->new_nbh.size(); j++) {
		container.push_back(p->new_nbh[j]); 
	}
	for(int j=0; j<p->new_nbh.size(); j++){
		p->out_nbh[j] = container[j];
	}
	p->new_nbh = parlay::make_slice<int*, int*>(nullptr, nullptr);
}

//synchronization function
template<typename T>
void synchronize(parlay::sequence<Tvec_point<T>*> &v){
	size_t n = v.size();
	parlay::parallel_for(0, n, [&] (size_t i){
		synchronize(v[i]);
	});
}
#endif

template<typename T>
void clear(Tvec_point<T>& p){
  auto nbh_cs = ityr::make_checkout(p.out_nbh, ityr::checkout_mode::write);
	for(std::size_t j=0; j<nbh_cs.size(); j++) nbh_cs[j] = -1;
} 

template<typename T>
void clear(ityr::global_span<Tvec_point<T>> v){
  ityr::root_exec([=] {
        ityr::for_each(
            ityr::execution::parallel_policy(1024),
            ityr::make_global_iterator(v.begin(), ityr::checkout_mode::read),
            ityr::make_global_iterator(v.end()  , ityr::checkout_mode::read),
            [=](Tvec_point<T>& p) {
              clear(p);
            });
  });
} 

#endif  

