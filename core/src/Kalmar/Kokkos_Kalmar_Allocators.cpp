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

#include <Kokkos_Macros.hpp>

#if ! defined( KOKKOS_USING_EXPERIMENTAL_VIEW )

/* only compile this file if Kalmar is enabled for Kokkos */
#ifdef KOKKOS_HAVE_KALMAR

#include "hc.hpp"
#include <impl/Kokkos_Error.hpp>
#include <Kalmar/Kokkos_Kalmar_Allocators.hpp>

#include <sstream>

namespace Kokkos { namespace Impl {

/*--------------------------------------------------------------------------*/

void * KalmarAllocator::allocate( size_t size )
{
  hc::array<char>* ptr = new hc::array<char>(size);
  assert( ptr && "array pointer is null");
  return ptr->accelerator_pointer();
}

void KalmarAllocator::deallocate( void * ptr, size_t /*size*/ )
{
  try {
    // TODO: Release hc containter pointer
  } catch(...) {}
}

void * KalmarAllocator::reallocate(void * old_ptr, size_t old_size, size_t new_size)
{
  void * ptr = old_ptr;
  if (old_size != new_size) {
    ptr = allocate( new_size );
    size_t copy_size = old_size < new_size ? old_size : new_size;
    hc::array<char> Old(old_size, old_ptr);
    hc::array<char> Now(new_size, ptr);
    hc::copy(Old, Now);
    deallocate( old_ptr, old_size );
  }
  return ptr;
}

/*--------------------------------------------------------------------------*/

}} // namespace Kokkos::Impl

#endif //KOKKOS_HAVE_CUDA

#endif /* #if ! defined( KOKKOS_USING_EXPERIMENTAL_VIEW ) */

