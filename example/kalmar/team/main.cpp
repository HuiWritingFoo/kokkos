
#include "Kokkos_Core.hpp"


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


typedef Kokkos::TeamPolicy<Kokkos::Kalmar> Policy;

template<class T>
struct functor_type_range_for {

  Kokkos::View<T**,Kokkos::Kalmar> data ;

  functor_type_range_for()
    : data( "data" , 10 ,10)
    {
    }

  KOKKOS_INLINE_FUNCTION
//  void operator() (Kokkos::kalmar_team_member_type team) const
  void operator() (Kokkos::Impl::KalmarTeamMember team) const
  {
    if(team.league_rank()<10)
    if(team.team_rank()<10)
    data(team.league_rank(),team.team_rank()) = team.league_rank()*100 + team.team_rank();
  }
};

int main() {
  Kokkos::initialize();

  functor_type_range_for<int> f ;

  for ( int i = 0 ; i < 10 ; ++i ) 
  for ( int j = 0 ; j < 10 ; ++j ) 
    f.data(i,j) = 0 ;


  Kokkos::parallel_for(Policy(10,10),f);

  for ( int i = 0 ; i < 10 ; ++i ) 
  for ( int j = 0 ; j < 10 ; ++j ) 
  {
    printf("f.data[%d] = %d\n",i,f.data(i,j));
  }

  Kokkos::finalize();

   
}

