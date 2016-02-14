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

#ifndef KOKKOS_KALMAR_ERROR_HPP
#define KOKKOS_KALMAR_ERROR_HPP

#include "hc_am.hpp"
#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Error.hpp>

#include <iostream>
#include <sstream>
#include <string>

/* only compile this file if Kalmar is enabled for Kokkos */
#ifdef KOKKOS_HAVE_KALMAR

namespace Kokkos { namespace Impl {

static void kalmar_internal_error_throw( am_status_t e , const char * name, const char * file = NULL, const int line = 0 ) {
  std::ostringstream out ;
  out << name << " error( " << "hc am error" << "): ";
  if (file) {
    out << " " << file << ":" << line;
  }
  throw_runtime_exception( out.str() );

}

inline void kalmar_internal_safe_call( am_status_t e , const char * name, const char * file = NULL, const int line = 0)
{
  if ( AM_SUCCESS != e ) { kalmar_internal_error_throw( e , name, file, line ); }
}

#define KALMAR_SAFE_CALL( call )  \
	Kokkos::Impl::kalmar_internal_safe_call( call , #call, __FILE__, __LINE__ )

#define KALMAR_ASSERT(exp)                                                   \
        Kokkos::Impl::kalmar_internal_safe_call( (exp) ? AM_SUCCESS : AM_ERROR_MISC, #exp, __FILE__, __LINE__ );

}} // namespace Kokkos::Impl

#endif //KOKKOS_HAVE_KALMAR
#endif //KOKKOS_KALMAR_ERROR_HPP
