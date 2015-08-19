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

#include <typeinfo>
#include <Kalmar/Kokkos_Kalmar_Reduce.hpp>

namespace Kokkos {
namespace Impl {
  struct KalmarTeamMember ;
}

template< class Arg0 , class Arg1 >
class TeamPolicy< Arg0 , Arg1 , Kokkos::Kalmar > {
public:
  enum { TEAM_SIZE = 256 };
  int m_league_size ;
  int m_team_size ;

  using execution_policy = TeamPolicy ;
  using execution_space  = Kokkos::Kalmar ;
  using work_tag         = void ;

  TeamPolicy( const int arg_league_size
            , const int arg_team_size )
    : m_league_size( arg_league_size ), m_team_size( arg_team_size )
    {}

  KOKKOS_INLINE_FUNCTION int team_size() const { return m_team_size ; }
  KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size ; }

  //This is again a reference thing from other module error 
  //We used auto last time to work around it.
  /*
  struct member_type {
    KOKKOS_INLINE_FUNCTION int league_rank() const ;
    KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size ; }
    KOKKOS_INLINE_FUNCTION int team_rank() const ;
    KOKKOS_INLINE_FUNCTION int team_size() const { return TEAM_SIZE ; }

    KOKKOS_INLINE_FUNCTION
    member_type( const TeamPolicy & arg_policy
               , const hc::tiled_index< TEAM_SIZE > & arg_idx )
      : m_league_size( arg_policy.league_size() )
      , m_league_rank( arg_idx.tile[0]  )
      , m_team_rank( arg_idx.local[0] )
      {}

  private:
    int m_league_size ;
    int m_league_rank ;
    int m_team_rank ;
  };
  */
  typedef Impl::KalmarTeamMember member_type;
};

namespace Impl {
  struct KalmarTeamMember {
    typedef TeamPolicy<Kokkos::Kalmar,void,Kokkos::Kalmar> TeamPolicy;
    KOKKOS_INLINE_FUNCTION int league_rank() const { return idx.tile[0] ; }
    KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size ; }
    KOKKOS_INLINE_FUNCTION int team_rank() const { return idx.local[0]/m_vector_length ; }
    KOKKOS_INLINE_FUNCTION int team_size() const { return m_team_size ; }


    KOKKOS_INLINE_FUNCTION
    KalmarTeamMember( const hc::tiled_index< 1 > & arg_idx, int league_size_,int team_size_ )
      : m_league_size( league_size_ )
      , m_team_size( team_size_ )
      , m_vector_length( 1 )
      , idx( arg_idx )
      {}

    KOKKOS_INLINE_FUNCTION
    KalmarTeamMember( const hc::tiled_index< 1 > & arg_idx, int league_size_,int team_size_,int vector_length_ )
      : m_league_size( league_size_ )
      , m_team_size( team_size_ )
      , m_vector_length( vector_length_ )
      , idx( arg_idx )
      {}

    KOKKOS_INLINE_FUNCTION
    void team_barrier() const {
      idx.barrier.wait();
    }

    
  private:
    int m_league_size ;
    int m_team_size ;
  public:
    int m_vector_length ;
    hc::tiled_index<1> idx;
  };
}
} // namespace Kokkos

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------

template< class FunctorType , class Arg0 , class Arg1 , class Arg2 >
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Arg0 , Arg1 , Arg2 , Kokkos::Kalmar > >
{
private:

  typedef Kokkos::RangePolicy< Arg0 , Arg1 , Arg2 , Kokkos::Kalmar > Policy ;

  const FunctorType& m_functor ;
  typename Policy::member_type m_offset ;

public:
  template<typename Tag>
  KOKKOS_INLINE_FUNCTION
  static
  void driver(const FunctorType& functor,
              typename std::enable_if< std::is_same<Tag, void>::value,
                                       typename Policy::member_type const & >::type index) { functor(index); }

  template<typename Tag>
  KOKKOS_INLINE_FUNCTION
  static
  void driver(const FunctorType& functor,
              typename std::enable_if< !std::is_same<Tag, void>::value,
                                       typename Policy::member_type const & >::type index) { functor(Tag(), index); }

  KOKKOS_INLINE_FUNCTION
  void operator()( const hc::index<1> & idx ) const
    {
       ParallelFor::template driver<typename Policy::work_tag> (m_functor, idx[0] + m_offset);
    }

  inline
  ParallelFor( const FunctorType & functor
             , const Policy      & policy )
     : m_functor( functor ),
       m_offset( policy.begin() )
    {

#if 0
      auto make_lambda = [this]( const hc::index<1> & idx ) restrict(amp) {
        this->operator() (idx);
      };
      hc::parallel_for_each( hc::extent<1>(
         policy.end()-policy.begin()) , make_lambda);
#else
if(policy.end()-policy.begin()==0) return;

      hc::completion_future fut = hc::parallel_for_each( hc::extent<1>(
         policy.end()-policy.begin()) , *this);
      fut.wait();
#endif

    }
};

//----------------------------------------------------------------------------

template< class FunctorType , class Arg0 , class Arg1 >
class ParallelFor< FunctorType
                 , Kokkos::TeamPolicy< Arg0 , Arg1 , Kokkos::Kalmar > >
{
  using Policy = Kokkos::TeamPolicy< Arg0 , Arg1 , Kokkos::Kalmar > ;
  const FunctorType& m_functor ;
  int league_size;
  int team_size;
public:
  template<typename Tag>
  KOKKOS_INLINE_FUNCTION
  static
  void driver(const FunctorType& functor,
              typename std::enable_if< std::is_same<Tag, void>::value,
                                       typename Policy::member_type const & >::type index) { functor(index); }

  template<typename Tag>
  KOKKOS_INLINE_FUNCTION
  static
  void driver(const FunctorType& functor,
              typename std::enable_if< !std::is_same<Tag, void>::value,
                                       typename Policy::member_type const & >::type index) { functor(Tag(), index); }

//  KOKKOS_INLINE_FUNCTION
//  static
//  void driver(const FunctorType& functor, typename Policy::member_type const& index) { functor(index); }
  KOKKOS_INLINE_FUNCTION
  void operator()( const hc::tiled_index<1> & idx ) const
    {
       ParallelFor::template driver<void> (m_functor, typename Policy::member_type(idx,league_size,team_size));
    }


  inline
  ParallelFor( const FunctorType & functor
             , const Policy      & policy )
    :m_functor(functor)
    {
#if 0
      auto make_lambda =
        [&]( const hc::tiled_index< 1 > & idx ) restrict(amp)
      {
        using member_type = typename Policy::member_type ;
        
        this->m_functor( KalmarTeamMember( policy , idx ) );
      };

      hc::extent< 1 >
        flat_extent( policy.league_size() * 256 );

      hc::tiled_extent< 1 > team_extent =
        flat_extent.tile(256);

      hc::parallel_for_each( team_extent , make_lambda );
#else
      league_size = policy.league_size();
      team_size = policy.team_size();
      hc::extent< 1 >
        flat_extent( policy.league_size() * policy.team_size() );

      hc::tiled_extent< 1 > team_extent =
        flat_extent.tile(policy.team_size());

      hc::completion_future fut = hc::parallel_for_each( team_extent , *this );
      fut.wait();
#endif
    }

};

//----------------------------------------------------------------------------

template< class FunctorType , class Arg0 , class Arg1 , class Arg2 >
class ParallelReduce<
  FunctorType , Kokkos::RangePolicy< Arg0 , Arg1 , Arg2 , Kokkos::Kalmar > >
{
public:

  typedef Kokkos::RangePolicy< Arg0 , Arg1 , Arg2 , Kokkos::Kalmar > Policy ;

  template< class ViewType >
  inline
  ParallelReduce( typename Impl::enable_if<
                    ( Impl::is_view< ViewType >::value &&
                      Impl::is_same< typename ViewType::memory_space , HostSpace >::value
                    ), const FunctorType & >::type functor
                , const Policy    & policy
                , const ViewType  & result_view )
    {
if(policy.end()-policy.begin()==0) return;
      Kokkos::Impl::reduce_enqueue
        ( policy.end() - policy.begin()
        , functor
        , result_view.ptr_on_device()
        , result_view.dimension_0()
        );
    }
};

}
}

namespace Kokkos {
namespace Impl {
  template<typename iType>
  struct TeamThreadRangeBoundariesStruct<iType,KalmarTeamMember> {
    typedef iType index_type;
    const iType start;
    const iType end;
    const iType increment;
    const KalmarTeamMember& thread;

    KOKKOS_INLINE_FUNCTION
    TeamThreadRangeBoundariesStruct (const KalmarTeamMember& thread_, const iType& count):
      start( thread_.team_rank() ),
      end( count ),
      increment( thread_.team_size() ),
      thread(thread_)
    {}
    KOKKOS_INLINE_FUNCTION
    TeamThreadRangeBoundariesStruct (const KalmarTeamMember& thread_,  const iType& begin_, const iType& end_):
      start( begin_ + thread_.team_rank() ),
      end( end_ ),
      increment( thread_.team_size() ),
      thread(thread_)
    {}
  };

}
}

namespace Kokkos {

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct<iType,Impl::KalmarTeamMember>
  TeamThreadRange(const Impl::KalmarTeamMember& thread, const iType& count) {
  return Impl::TeamThreadRangeBoundariesStruct<iType,Impl::KalmarTeamMember>(thread,count);
}

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct<iType,Impl::KalmarTeamMember>
  TeamThreadRange(const Impl::KalmarTeamMember& thread, const iType& begin, const iType& end) {
  return Impl::TeamThreadRangeBoundariesStruct<iType,Impl::KalmarTeamMember>(thread,begin,end);
}

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::KalmarTeamMember >
  ThreadVectorRange(const Impl::KalmarTeamMember& thread, const iType& count) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::KalmarTeamMember >(thread,count);
}

KOKKOS_INLINE_FUNCTION
Impl::ThreadSingleStruct<Impl::KalmarTeamMember> PerTeam(const Impl::KalmarTeamMember& thread) {
  return Impl::ThreadSingleStruct<Impl::KalmarTeamMember>(thread);
}

KOKKOS_INLINE_FUNCTION
Impl::VectorSingleStruct<Impl::KalmarTeamMember> PerThread(const Impl::KalmarTeamMember& thread) {
  return Impl::VectorSingleStruct<Impl::KalmarTeamMember>(thread);
}


template<class FunctorType>
KOKKOS_INLINE_FUNCTION
void single(const Impl::VectorSingleStruct<Impl::KalmarTeamMember>& single_struct, const FunctorType& lambda) {
  if( single_struct.team_member.idx.local[0]%single_struct.team_member.m_vector_length == 0) lambda();
}

template<class FunctorType>
KOKKOS_INLINE_FUNCTION
void single(const Impl::ThreadSingleStruct<Impl::KalmarTeamMember>& single_struct, const FunctorType& lambda) {
  if( single_struct.team_member.idx.tile[0] == 0 ) lambda();
}

//template<class FunctorType, class ValueType>
//KOKKOS_INLINE_FUNCTION
//void single(const Impl::VectorSingleStruct<Impl::KalmarTeamMember>& single_struct, const FunctorType& lambda, ValueType& val) {
//  if( single_struct.team_member.idx.local[0]%single_struct.team_member.idx.m_vector_length == 0) lambda(val);
//  val = shfl(val,0,blockDim.x);
//}

//template<class FunctorType, class ValueType>
//KOKKOS_INLINE_FUNCTION
//void single(const Impl::ThreadSingleStruct<Impl::KalmarTeamMember>& single_struct, const FunctorType& lambda, ValueType& val) {
//  if( single_struct.team_member.idx.local[0] == 0 ) {
//    lambda(val);
//  }
//  single_struct.team_member.team_broadcast(val,0);
//}

}

namespace Kokkos {
  /** \brief  Inter-thread parallel_for. Executes lambda(iType i) for each i=0..N-1.
   *
   * The range i=0..N-1 is mapped to all threads of the the calling thread team.
   * This functionality requires C++11 support.*/
template<typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION
void parallel_for(const Impl::TeamThreadRangeBoundariesStruct<iType,Impl::KalmarTeamMember>& loop_boundaries, const Lambda& lambda) {
  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment)
    lambda(i);
}

}

namespace Kokkos {
/** \brief  Intra-thread vector parallel_for. Executes lambda(iType i) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread.
 * This functionality requires C++11 support.*/
template<typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION
void parallel_for(const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::KalmarTeamMember >&
    loop_boundaries, const Lambda& lambda) {
  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment)
    lambda(i);
}
}

