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

#ifndef KOKKOS_EXPERIMENTAL_KALMAR_VIEW_HPP
#define KOKKOS_EXPERIMENTAL_KALMAR_VIEW_HPP

/* only compile this file if Kalmar is enabled for Kokkos */
#if defined( KOKKOS_HAVE_KALMAR )

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

template< class >
struct ViewOperatorBoundsErrorAbort ;

template<>
struct ViewOperatorBoundsErrorAbort< Kokkos::KalmarSpace > {
 static void apply( const size_t rank
                  , const size_t n0 , const size_t n1
                  , const size_t n2 , const size_t n3
                  , const size_t n4 , const size_t n5
                  , const size_t n6 , const size_t n7
                  , const size_t i0 , const size_t i1
                  , const size_t i2 , const size_t i3
                  , const size_t i4 , const size_t i5
                  , const size_t i6 , const size_t i7 );
};

  KOKKOS_INLINE_FUNCTION
  static void apply( const size_t rank
                   , const size_t n0 , const size_t n1
                   , const size_t n2 , const size_t n3
                   , const size_t n4 , const size_t n5
                   , const size_t n6 , const size_t n7
                   , const size_t i0 , const size_t i1
                   , const size_t i2 , const size_t i3
                   , const size_t i4 , const size_t i5
                   , const size_t i6 , const size_t i7 ) restrict(amp, cpu)
    {
#if !defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_KALMAR_GPU )
      char buffer[512];

      snprintf( buffer , sizeof(buffer)
          , "Kalmar View operator bounds error : rank(%lu) dim(%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu) index(%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu)"
          , rank , n0 , n1 , n2 , n3 , n4 , n5 , n6 , n7
                 , i0 , i1 , i2 , i3 , i4 , i5 , i6 , i7 );

      Kokkos::Impl::throw_runtime_exception( buffer );
#endif
    }


} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_HAVE_KALMAR ) */
#endif /* #ifndef KOKKOS_EXPERIMENTAL_KALMAR_VIEW_HPP */

