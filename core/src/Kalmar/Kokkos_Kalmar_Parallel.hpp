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

#include <algorithm>
#include <typeinfo>
#include <Kalmar/Kokkos_Kalmar_Reduce.hpp>
#include <Kalmar/Kokkos_Kalmar_Scan.hpp>

namespace Kokkos {
namespace Impl {

template<>
struct VerifyExecutionCanAccessMemorySpace
  < Kokkos::Kalmar::memory_space
  , Kokkos::Kalmar::scratch_memory_space
  >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

  struct KalmarTeamMember ;
}

template< class Arg0 , class Arg1 >
class TeamPolicy< Arg0 , Arg1 , Kokkos::Kalmar > {
public:
  int m_league_size ;
  int m_team_size ;
  int m_vector_length ;

  using execution_policy = TeamPolicy ;
  using execution_space  = Kokkos::Kalmar ;
  using work_tag = typename Impl::if_c< ! Impl::is_same< Kokkos::Kalmar , Arg0 >::value , Arg0 , Arg1 >::type;

  TeamPolicy( const int arg_league_size
            , const int arg_team_size )
    : m_league_size( arg_league_size ), m_team_size( arg_team_size )
    {}
  TeamPolicy( const int arg_league_size
            , const int arg_team_size
            , const int vector_length_request)
    : m_league_size( arg_league_size ),
      m_team_size( arg_team_size ),
      m_vector_length (vector_length_request)
    {}

  TeamPolicy( const int arg_league_size
            , const Kokkos::AUTO_t )
    : m_league_size( arg_league_size ), m_team_size( -1 )
    {}
  TeamPolicy( const int arg_league_size
            , const Kokkos::AUTO_t
            , const int vector_length_request)
    : m_league_size( arg_league_size ),
      m_team_size( -1 ),
      m_vector_length (vector_length_request)
    {}

  template< class Functor_Type>
  KOKKOS_INLINE_FUNCTION static
  int team_size_max( const Functor_Type & functor)
  {
    typedef typename Kokkos::Impl::FunctorValueTraits<Functor_Type, void>::value_type value_type;
    return team_size_recommended(functor);
    // return std::min(Kokkos::Impl::get_max_tile_size() / sizeof(value_type), Kokkos::Impl::get_max_tile_thread());
  }

  template< class Functor_Type>
  KOKKOS_INLINE_FUNCTION static int team_size_recommended(const Functor_Type & functor)
  { return Kokkos::Impl::get_tile_size<typename Kokkos::Impl::FunctorValueTraits<Functor_Type, void>::value_type>(); }

  template< class Functor_Type >
  KOKKOS_INLINE_FUNCTION static int team_size_recommended(const Functor_Type &functor, const int vector_length)
 {
   int max = team_size_recommended( functor )/vector_length;
   if(max < 1) max = 1;
   return(max);
 }

  template<class F>
  KOKKOS_INLINE_FUNCTION int team_size(const F& f) const { return (m_team_size > 0) ? m_team_size : team_size_recommended(f); }
  KOKKOS_INLINE_FUNCTION int team_size() const { return (m_team_size > 0) ? m_team_size : Impl::get_max_tile_thread(); ; }
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
    typedef Kokkos::ScratchMemorySpace<Kokkos::Kalmar> scratch_memory_space ;
    typedef TeamPolicy<Kokkos::Kalmar,void,Kokkos::Kalmar> TeamPolicy;

    KOKKOS_INLINE_FUNCTION
    const scratch_memory_space & team_shmem() const { return m_space; }

    KOKKOS_INLINE_FUNCTION int league_rank() const { return idx.tile[0]; }
    KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size; }
    KOKKOS_INLINE_FUNCTION int team_rank() const { return idx.local[0] / m_vector_length; }
    KOKKOS_INLINE_FUNCTION int team_size() const { return m_team_size; }


    KOKKOS_INLINE_FUNCTION
    KalmarTeamMember( const hc::tiled_index< 1 > & arg_idx, int league_size_,int team_size_ )
      : m_league_size( league_size_ )
      , m_team_size( team_size_ )
      , m_space( nullptr, 0 )
      , m_vector_length( 1 )
      , idx( arg_idx )
      {}

    KOKKOS_INLINE_FUNCTION
    KalmarTeamMember( const hc::tiled_index< 1 > & arg_idx, int league_size_,int team_size_, char * shmem, std::size_t shsize )
      : m_league_size( league_size_ )
      , m_team_size( team_size_ )
      , m_space( shmem + arg_idx.tile[0] * shsize, shsize )
      , m_vector_length( 1 )
      , idx( arg_idx )
      {}

    // KOKKOS_INLINE_FUNCTION
    // KalmarTeamMember( const hc::tiled_index< 1 > & arg_idx, int league_size_,int team_size_,int vector_length_ )
    //   : m_league_size( league_size_ )
    //   , m_team_size( team_size_ )
    //   , m_vector_length( vector_length_ )
    //   , idx( arg_idx )
    //   , m_space( nullptr, 0 )
    //   {}

    KOKKOS_INLINE_FUNCTION
    void team_barrier() const {
      idx.barrier.wait();
    }

    template<class ValueType>
    KOKKOS_INLINE_FUNCTION
    void team_broadcast(const ValueType& value, const int& thread_id ) const 
    {
      static_assert(std::is_trivially_default_constructible<ValueType>(), "Only trivial constructible types can be broadcasted");
      tile_static ValueType local_value;
      zero_init(local_value);
      if (this->team_rank() == thread_id) {
        local_value = value;
      }
      this->team_barrier();
      value = local_value;
    }

    template< class ValueType, class JoinOp >
    KOKKOS_INLINE_FUNCTION
    ValueType team_reduce( const ValueType & value , const JoinOp & op_in) const
    {
      typedef JoinLambdaAdapter<ValueType,JoinOp> JoinOpFunctor ;
      const JoinOpFunctor op(op_in);

      const auto local = idx.local[0];
      tile_static ValueType buffer[128];
      const std::size_t size = next_pow_2(m_team_size+1)/2;
      lds_for(buffer[local], [&](ValueType& x)
      {
          x = value;
      });
      idx.barrier.wait();

      for(std::size_t s = 1; s < size; s *= 2)
      {
          const std::size_t index = 2 * s * local;
          if (index < size)
          {
              lds_for(buffer[index], [&](ValueType& x)
              {
                  lds_for(buffer[index+s], [&](ValueType& y)
                  {
                      op.join(x, y);
                  });
              });
          }
          idx.barrier.wait();
      }

      if (local == 0)
      {
          buffer[0] = std::accumulate(buffer+size, buffer+m_team_size, buffer[0], [&](ValueType x, ValueType y)
          {
              op.join(x, y);
              return x;
          });
      }
      idx.barrier.wait();
      return buffer[0];
    }

    /** \brief  Intra-team exclusive prefix sum with team_rank() ordering
     *          with intra-team non-deterministic ordering accumulation.
     *
     *  The global inter-team accumulation value will, at the end of the
     *  league's parallel execution, be the scan's total.
     *  Parallel execution ordering of the league's teams is non-deterministic.
     *  As such the base value for each team's scan operation is similarly
     *  non-deterministic.
     */
    template< typename Type >
    KOKKOS_INLINE_FUNCTION Type team_scan( const Type & value , Type * const global_accum = nullptr ) const
    {
  #if 0
      const auto local = idx.local[0];
      const auto last = m_team_size - 1;
      const auto init = 0;
      tile_static Type buffer[256];

      if (local == last) buffer[0] = init;
      else buffer[local] = value;

      idx.barrier.wait();

      for(std::size_t s = 1; s < m_team_size; s *= 2)
      {
          if (local >= s) buffer[local] += buffer[local - s];
          idx.barrier.wait();
      }

      if ( global_accum )
      { 
         if(local == last)
         {
            atomic_fetch_add(global_accum, buffer[local] + value);
         }
         idx.barrier.wait();
         buffer[local] += *global_accum;
      }
      idx.barrier.wait();
      return buffer[local];
#else
      tile_static Type sarray[2][256+1];
      int lid = idx.local[0];
      int lp1 = lid+1;

      int toggle = 1;
      int _toggle = 0;
      idx.barrier.wait();

      if(lid == 0) 
      {
         sarray[1][0] = 0;
         sarray[0][0] = 0;
      }
      sarray[1][lp1] = value;

      idx.barrier.wait();
      for(int stride = 1; stride < m_team_size; stride*=2)
      {
         if(lid >= stride)
         {
            sarray[_toggle][lp1] =
                          sarray[toggle][lp1]+sarray[toggle][lp1-stride];
         }
         else
         {
            sarray[_toggle][lp1] = sarray[toggle][lp1];
         }
         toggle = _toggle;
         _toggle = 1-toggle;
         idx.barrier.wait();
      }

      if ( global_accum )
      { 
         if(m_team_size == lp1)
         {
            sarray[toggle][m_team_size] = atomic_fetch_add(global_accum,sarray[toggle][m_team_size]);
         }
         idx.barrier.wait();
         sarray[toggle][lid] += sarray[toggle][m_team_size];
      }
      idx.barrier.wait();
      return sarray[toggle][lid];
#endif
    }

  private:
    int m_league_size ;
    int m_team_size ;
    const scratch_memory_space  m_space;
  public:
    int m_vector_length;
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

public:

  inline
  ParallelFor( const FunctorType & f
             , const Policy      & policy )
    {


      const auto len = policy.end()-policy.begin();
      const auto offset = policy.begin();
      if(len == 0) return;

      hc::parallel_for_each(hc::extent<1>(len) , [&](const hc::index<1> & idx) restrict(amp)
      {
        kalmar_invoke<typename Policy::work_tag>(f, idx[0] + offset);
      }).wait();

    }

  KOKKOS_INLINE_FUNCTION
  void execute() const {}

};

//----------------------------------------------------------------------------

template< class F , class Arg0 , class Arg1 >
class ParallelFor< F
                 , Kokkos::TeamPolicy< Arg0 , Arg1 , Kokkos::Kalmar > >
{
  using Policy = Kokkos::TeamPolicy< Arg0 , Arg1 , Kokkos::Kalmar > ;
  typedef Kokkos::Impl::FunctorValueTraits<F, typename Policy::work_tag> ValueTraits;

public:
  inline
  ParallelFor( const F & f
             , const Policy      & policy )
    {
      const auto league_size = policy.league_size();
      const auto team_size = policy.team_size();
      const auto total_size = league_size * team_size;

      if(total_size == 0) return;

      const auto shared_size = FunctorTeamShmemSize< F >::value( f , team_size );

      std::vector<char> scratch(shared_size * total_size);

      hc::extent< 1 > flat_extent( total_size );

      hc::tiled_extent< 1 > team_extent = flat_extent.tile(team_size);

      hc::parallel_for_each( team_extent , [&](hc::tiled_index<1> idx) restrict(amp)
      {
        kalmar_invoke<typename Policy::work_tag>(f, typename Policy::member_type(idx, league_size, team_size, scratch.data(), shared_size));
      }).wait();
    }

  KOKKOS_INLINE_FUNCTION
  void execute() const {}

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
                    ), const FunctorType & >::type f
                , const Policy    & policy
                , const ViewType  & result_view )
    {
      typedef typename Policy::work_tag Tag;
      typedef Kokkos::Impl::FunctorValueTraits< FunctorType , Tag > ValueTraits;
      typedef typename ValueTraits::reference_type reference_type;
if(policy.end()-policy.begin()==0) return;
      Kokkos::Impl::reduce_enqueue< Tag >
        ( policy.end() - policy.begin()
        , f
        , [](hc::tiled_index<1> idx, int, int) { return idx.global[0]; }
        , result_view.ptr_on_device()
        , result_view.dimension_0()
        );
    }

  KOKKOS_INLINE_FUNCTION
  void execute() const {}

};

template< class FunctorType, class Arg0, class Arg1 >
class ParallelReduce<
   FunctorType , Kokkos::TeamPolicy< Arg0, Arg1, Kokkos::Kalmar > >
{
  using Policy = Kokkos::TeamPolicy< Arg0, Arg1, Kokkos::Kalmar >;
  typedef Kokkos::Impl::FunctorValueTraits<FunctorType, typename Policy::work_tag> ValueTraits;

public:
  template< class ViewType >
  inline
  ParallelReduce( typename Impl::enable_if<
                  ( Impl::is_view< ViewType >::value &&
                    Impl::is_same< typename ViewType::memory_space, HostSpace >::value), const FunctorType &>::type f
                  , const Policy &policy
                  , const ViewType & result_view)
    {
      const int league_size = policy.league_size();
      const int team_size = policy.team_size(f);
      const int total_size = league_size * team_size;

      if(total_size == 0) return;

      const int reduce_size = ValueTraits::value_size( f );
      const int shared_size = FunctorTeamShmemSize< FunctorType >::value( f , team_size );

      std::vector<char> scratch(reduce_size * shared_size * total_size);
       
      Kokkos::Impl::reduce_enqueue< typename Policy::work_tag >
      ( total_size 
        , f
        , [&](hc::tiled_index<1> idx, int n_teams, int n_leagues) { return typename Policy::member_type(idx, n_leagues, n_teams, scratch.data(), shared_size); }
        , result_view.ptr_on_device()
        , result_view.dimension_0()
     );

    }

  KOKKOS_INLINE_FUNCTION
  void execute() const {}

};


template< class FunctionType , class Arg0 , class Arg1 , class Arg2 >
class ParallelScan< FunctionType , Kokkos::RangePolicy< Arg0 , Arg1 , Arg2 , Kokkos::Kalmar > >
{
private:

  typedef Kokkos::RangePolicy< Arg0 , Arg1 , Arg2 , Kokkos::Kalmar > Policy;
  typedef typename Policy::work_tag Tag;
  typedef Kokkos::Impl::FunctorValueTraits< FunctionType, Tag>  ValueTraits;

public:

  //----------------------------------------

  inline
  ParallelScan( const FunctionType & f
              , const Policy      & policy )
  {
    const auto len = policy.end()-policy.begin();

    scan_enqueue<Tag>(len, f, [](hc::tiled_index<1> idx, int, int) { return idx.global[0]; });
  }

  KOKKOS_INLINE_FUNCTION
  void execute() const {}

  //----------------------------------------
};

template< class FunctionType , class Arg0 , class Arg1>
class ParallelScan< FunctionType , Kokkos::TeamPolicy< Arg0 , Arg1 , Kokkos::Kalmar > >
{
private:

  typedef Kokkos::TeamPolicy< Arg0 , Arg1 , Kokkos::Kalmar > Policy;
  typedef typename Policy::work_tag Tag;
  typedef Kokkos::Impl::FunctorValueTraits< FunctionType, Tag>  ValueTraits;

public:

  //----------------------------------------

  inline
  ParallelScan( const FunctionType & f
              , const Policy      & policy )
  {
    const auto league_size = policy.league_size();
    const auto team_size = policy.team_size(f);
    const auto len  = league_size * team_size;
      
    if(len == 0) return;

    scan_enqueue<Tag>(len, f, [&](hc::tiled_index<1> idx, int n_teams, int n_leagues) { return typename Policy::member_type(idx,n_leagues,n_teams); });
  }

  KOKKOS_INLINE_FUNCTION
  void execute() const {}

  //----------------------------------------
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
  lambda();
}

template<class FunctorType>
KOKKOS_INLINE_FUNCTION
void single(const Impl::ThreadSingleStruct<Impl::KalmarTeamMember>& single_struct, const FunctorType& lambda) {
  if(single_struct.team_member.team_rank()==0) lambda();
}

template<class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION
void single(const Impl::VectorSingleStruct<Impl::KalmarTeamMember>& single_struct, const FunctorType& lambda, ValueType& val) {
  lambda(val);
}

template<class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION
void single(const Impl::ThreadSingleStruct<Impl::KalmarTeamMember>& single_struct, const FunctorType& lambda, ValueType& val) {
  if(single_struct.team_member.team_rank()==0) {
    lambda(val);
  }
  single_struct.team_member.team_broadcast(val,0);
}

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

/** \brief  Inter-thread vector parallel_reduce. Executes lambda(iType i, ValueType & val) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all threads of the the calling thread team and a summation of
 * val is performed and put into result. This functionality requires C++11 support.*/
template< typename iType, class Lambda, typename ValueType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<iType,Impl::KalmarTeamMember>& loop_boundaries,
                     const Lambda & lambda, ValueType& result) {

  result = ValueType();

  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    ValueType tmp = ValueType();
    lambda(i,tmp);
    result+=tmp;
  }

  result = loop_boundaries.thread.team_reduce(result,Impl::JoinAdd<ValueType>());
}

/** \brief  Intra-thread vector parallel_reduce. Executes lambda(iType i, ValueType & val) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread and a reduction of
 * val is performed using JoinType(ValueType& val, const ValueType& update) and put into init_result.
 * The input value of init_result is used as initializer for temporary variables of ValueType. Therefore
 * the input value should be the neutral element with respect to the join operation (e.g. '0 for +-' or
 * '1 for *'). This functionality requires C++11 support.*/
template< typename iType, class Lambda, typename ValueType, class JoinType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<iType,Impl::KalmarTeamMember>& loop_boundaries,
                     const Lambda & lambda, const JoinType& join, ValueType& init_result) {

  ValueType result = init_result;

  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    ValueType tmp = ValueType();
    lambda(i,tmp);
    join(result,tmp);
  }

  init_result = loop_boundaries.thread.team_reduce(result,join);
}

} //namespace Kokkos


namespace Kokkos {
/** \brief  Intra-thread vector parallel_for. Executes lambda(iType i) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread.
 * This functionality requires C++11 support.*/
template<typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION
void parallel_for(const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::KalmarTeamMember >&
    loop_boundaries, const Lambda& lambda) {
  #ifdef KOKKOS_HAVE_PRAGMA_IVDEP
  #pragma ivdep
  #endif
  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment)
    lambda(i);
}

/** \brief  Intra-thread vector parallel_reduce. Executes lambda(iType i, ValueType & val) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread and a summation of
 * val is performed and put into result. This functionality requires C++11 support.*/
template< typename iType, class Lambda, typename ValueType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::KalmarTeamMember >&
      loop_boundaries, const Lambda & lambda, ValueType& result) {
  result = ValueType();
#ifdef KOKKOS_HAVE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    ValueType tmp = ValueType();
    lambda(i,tmp);
    result+=tmp;
  }
}

/** \brief  Intra-thread vector parallel_reduce. Executes lambda(iType i, ValueType & val) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread and a reduction of
 * val is performed using JoinType(ValueType& val, const ValueType& update) and put into init_result.
 * The input value of init_result is used as initializer for temporary variables of ValueType. Therefore
 * the input value should be the neutral element with respect to the join operation (e.g. '0 for +-' or
 * '1 for *'). This functionality requires C++11 support.*/
template< typename iType, class Lambda, typename ValueType, class JoinType >
KOKKOS_INLINE_FUNCTION
void parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::KalmarTeamMember >&
      loop_boundaries, const Lambda & lambda, const JoinType& join, ValueType& init_result) {

  ValueType result = init_result;
#ifdef KOKKOS_HAVE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    ValueType tmp = ValueType();
    lambda(i,tmp);
    join(result,tmp);
  }
  init_result = result;
}

/** \brief  Intra-thread vector parallel exclusive prefix sum. Executes lambda(iType i, ValueType & val, bool final)
 *          for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes in the thread and a scan operation is performed.
 * Depending on the target execution space the operator might be called twice: once with final=false
 * and once with final=true. When final==true val contains the prefix sum value. The contribution of this
 * "i" needs to be added to val no matter whether final==true or not. In a serial execution
 * (i.e. team_size==1) the operator is only called once with final==true. Scan_val will be set
 * to the final sum value over all vector lanes.
 * This functionality requires C++11 support.*/
template< typename iType, class FunctorType >
KOKKOS_INLINE_FUNCTION
void parallel_scan(const Impl::ThreadVectorRangeBoundariesStruct<iType,Impl::KalmarTeamMember >&
      loop_boundaries, const FunctorType & lambda) {

  typedef Kokkos::Impl::FunctorValueTraits< FunctorType , void > ValueTraits ;
  typedef typename ValueTraits::value_type value_type ;

  value_type scan_val = value_type();

#ifdef KOKKOS_HAVE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for( iType i = loop_boundaries.start; i < loop_boundaries.end; i+=loop_boundaries.increment) {
    lambda(i,scan_val,true);
  }
}

} // namespace Kokkos

