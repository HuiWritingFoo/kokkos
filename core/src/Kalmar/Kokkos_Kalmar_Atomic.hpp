/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <hc.hpp>
#include <hsa_atomic.h>

//#ifdef KOKKOS_USE_ATOMICS_KALMAR_GPU
namespace Kokkos {
  //Kalmar can do:
  //Types int/unsigned int
  //variants: atomic_exchange/compare_exchange/fetch_add/fetch_sub/fetch_max/fetch_min/fetch_and/fetch_or/fetch_xor/fetch_inc/fetch_dec 


  KOKKOS_INLINE_FUNCTION
  int atomic_exchange(int* dest, const int& val) {
    return ::__hsail_atomic_exchange_int(dest, val);
  }

  KOKKOS_INLINE_FUNCTION
  unsigned int atomic_exchange(unsigned int* dest, const unsigned int& val) {
    return ::__hsail_atomic_exchange_unsigned(dest, val);
  }

  KOKKOS_INLINE_FUNCTION
  int64_t atomic_exchange(int64_t* dest, const int64_t& val) {
    return ::__hsail_atomic_exchange_int64(dest, val);
  }

  KOKKOS_INLINE_FUNCTION
  int atomic_compare_exchange(int* dest, int compare, const int& val);

  KOKKOS_INLINE_FUNCTION
  int64_t atomic_compare_exchange(int64_t* dest, int64_t compare, const int64_t& val);

  template<class T>
  KOKKOS_INLINE_FUNCTION
  T atomic_exchange(T* dest, typename std::enable_if<sizeof(T) == sizeof(int), const T&>::type val) {
    union U {
      int i ;
      T t ;
      KOKKOS_INLINE_FUNCTION U() {};
    } assume , oldval , newval ;

    oldval.t = *dest ;
    assume.i = oldval.i ;
    newval.t = val ;
    Kokkos::atomic_compare_exchange( reinterpret_cast<int*>(dest) , assume.i, newval.i );

    return oldval.t ;    
  }

  template<class T>
  KOKKOS_INLINE_FUNCTION
  T atomic_exchange(T* dest, typename std::enable_if<sizeof(T) != sizeof(int) && sizeof(T) == sizeof(int64_t), const T&>::type val) {
    union U {
      int64_t i ;
      T t ;
      KOKKOS_INLINE_FUNCTION U() {};
    } assume , oldval , newval ;

    oldval.t = *dest ;

    assume.i = oldval.i ;
    newval.t = val ;
    Kokkos::atomic_compare_exchange( reinterpret_cast<int64_t*>(dest) , assume.i, newval.i );

    return oldval.t ;    
  }

  KOKKOS_INLINE_FUNCTION
  int atomic_compare_exchange(int* dest, int compare, const int& val) {
    return ::__hsail_atomic_compare_exchange_int(dest, compare, val);
  }

  KOKKOS_INLINE_FUNCTION
  unsigned int atomic_compare_exchange(unsigned int* dest, unsigned int compare, const unsigned int& val) {
    return ::__hsail_atomic_compare_exchange_unsigned(dest, compare, val);
  }

  KOKKOS_INLINE_FUNCTION
  int64_t atomic_compare_exchange(int64_t* dest, int64_t compare, const int64_t& val) {
    return ::__hsail_atomic_compare_exchange_int64(dest, compare, val);
  }

  template<typename T>
  KOKKOS_INLINE_FUNCTION
  T atomic_compare_exchange(T* dest, T& compare , typename std::enable_if<sizeof(T) == sizeof(int), const T&>::type val) {
    union U {
      int i ;
      T t ;
      KOKKOS_INLINE_FUNCTION U() {};
    } assume , oldval, newval ;

    oldval.t = *dest ;
    assume.t = compare ;
    newval.t = val ;

    Kokkos::atomic_compare_exchange( reinterpret_cast<int*>(dest) , assume.i , newval.i );

    return oldval.t ;    
  }

  template<typename T>
  KOKKOS_INLINE_FUNCTION
  T atomic_compare_exchange(T* dest, T& compare , typename std::enable_if<sizeof(T) != sizeof(int) && sizeof(T) == sizeof(int64_t), const T&>::type val) {
    union U {
      int64_t i ;
      T t ;
      KOKKOS_INLINE_FUNCTION U() {};
    } assume , oldval, newval ;

    oldval.t = *dest ;
    assume.t = compare ;
    newval.t = val ;

    Kokkos::atomic_compare_exchange( reinterpret_cast<int64_t*>(dest) , assume.i , newval.i );

    return oldval.t ;    
  }

  KOKKOS_INLINE_FUNCTION
  int atomic_fetch_add(int* dest, const int& val) {
    return ::__hsail_atomic_fetch_add_int(dest, val);
  }
  
  KOKKOS_INLINE_FUNCTION
  unsigned int atomic_fetch_add(unsigned int* dest, const unsigned int& val) {
    return ::__hsail_atomic_fetch_add_unsigned(dest, val);
  }

  KOKKOS_INLINE_FUNCTION
  int64_t atomic_fetch_add(int64_t* dest, const int64_t& val) {
    return ::__hsail_atomic_fetch_add_int64(dest, val);
  }

  template<class T>
  KOKKOS_INLINE_FUNCTION
  T atomic_fetch_add(T* dest, typename std::enable_if<sizeof(T) == sizeof(int), const T&>::type val) {
    union U {
      int i ;
      T t ;
      KOKKOS_INLINE_FUNCTION U() {};
    } assume , oldval , newval ;

    oldval.t = *dest ;

    do {
      assume.i = oldval.i ;
      newval.t = assume.t + val ;
      oldval.i = Kokkos::atomic_compare_exchange( reinterpret_cast<int*>(dest) , assume.i , newval.i );
    } while ( assume.i != oldval.i );

    return oldval.t ;    
  }

  template<class T>
  KOKKOS_INLINE_FUNCTION
  T atomic_fetch_add(T* dest, typename std::enable_if<sizeof(T) != sizeof(int) && sizeof(T) == sizeof(int64_t), const T&>::type val) {
    union U {
      int64_t i ;
      T t ;
      KOKKOS_INLINE_FUNCTION U() {};
    } assume , oldval , newval ;

    oldval.t = *dest ;

    do {
      assume.i = oldval.i ;
      newval.t = assume.t + val ;
      oldval.i = Kokkos::atomic_compare_exchange( (int64_t*)dest , assume.i , newval.i );
    } while ( assume.i != oldval.i );

    return oldval.t ;    
  }

#if 0
  KOKKOS_INLINE_FUNCTION
  int atomic_fetch_sub(int* dest, int val) {
    #ifdef __KALMAR_ACCELERATOR__
    return ::atomic_fetch_sub(dest,val);
    #endif
  }

  KOKKOS_INLINE_FUNCTION
  unsigned int atomic_fetch_sub(unsigned int* dest, unsigned int& val) {
    #ifdef __KALMAR_ACCELERATOR__ 
    return ::atomic_fetch_sub(dest,val);
    #endif
  }

  template<class T>
  KOKKOS_INLINE_FUNCTION
  T atomic_fetch_sub(T* dest, typename std::enable_if<sizeof(T) == sizeof(int),T>::type & val) {
    union U {
      int i ;
      T t ;
      KOKKOS_INLINE_FUNCTION U() {};
    } assume , oldval , newval ;

    oldval.t = *dest ;

    do {
      assume.i = oldval.i ;
      newval.t = assume.t - val ;
      oldval.i = Kokkos::atomic_compare_exchange( (int*)dest , assume.i , newval.i );
    } while ( assume.i != oldval.i );

    return oldval.t ;
  }
#endif
}

