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

#ifndef KOKKOS_KALMARSPACE_HPP
#define KOKKOS_KALMARSPACE_HPP

#include <Kokkos_Core_fwd.hpp>

#if defined( KOKKOS_HAVE_KALMAR )

#include <iosfwd>
#include <typeinfo>
#include <string>

#include <Kokkos_HostSpace.hpp>

#include <impl/Kokkos_AllocationTracker.hpp>


/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

} // namespace Impl
} // namespace Kokkos

namespace Kokkos {
/** \brief Memory management for Kalmar Non-UVM */

class KalmarSpace {
public:

  //! Tag this class as a kokkos memory space
  typedef KalmarSpace  memory_space ;
  typedef size_t     size_type ;
  typedef Kokkos::Kalmar   execution_space ;
  //! This memory space preferred device_type
  typedef Kokkos::Device<execution_space,memory_space> device_type;

  /*--------------------------------*/
#if ! defined( KOKKOS_USING_EXPERIMENTAL_VIEW )


  /** \brief  Allocate a contiguous block of memory.
   *
   *  The input label is associated with the block of memory.
   *  The block of memory is tracked via reference counting where
   *  allocation gives it a reference count of one.
   */
  static Impl::AllocationTracker allocate_and_track( const std::string & label, const size_t size );
#endif /* #if ! defined( KOKKOS_USING_EXPERIMENTAL_VIEW ) */

  /*--------------------------------*/

  KalmarSpace();
  KalmarSpace( const KalmarSpace & rhs ) = default ;
  KalmarSpace & operator = ( const KalmarSpace & rhs ) = default ;
  ~KalmarSpace() = default ;

  /**\brief  Allocate untracked memory in the Kalmar space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the Kalmar space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /*--------------------------------*/
  /** \brief  Error reporting for HostSpace attempt to access KalmarSpace */
  static void access_error();
  static void access_error( const void * const );

private:

  int  m_device ; ///< Which Kalmar device

  // friend class Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::KalmarSpace , void > ;
};

} // namespace Kokkos


/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

void DeepCopyAsyncKalmar( void * dst , const void * src , size_t n);

template<> struct DeepCopy< KalmarSpace , KalmarSpace , Kalmar>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kalmar & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< KalmarSpace , HostSpace , Kalmar >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kalmar & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , KalmarSpace , Kalmar >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Kalmar & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace> struct DeepCopy< KalmarSpace , KalmarSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< KalmarSpace , KalmarSpace , Kalmar >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncKalmar (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< KalmarSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< KalmarSpace , HostSpace , Kalmar>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncKalmar (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , KalmarSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , KalmarSpace , Kalmar >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncKalmar (dst,src,n);
  }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** Running in KalmarSpace attempting to access HostSpace: error */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::KalmarSpace , Kokkos::HostSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("Kalmar code attempted to access HostSpace memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("Kalmar code attempted to access HostSpace memory"); }
};
/** Running in KalmarSpace attempting to access an unknown space: error */
template< class OtherSpace >
struct VerifyExecutionCanAccessMemorySpace<
  typename enable_if< ! is_same<Kokkos::KalmarSpace,OtherSpace>::value , Kokkos::KalmarSpace >::type ,
  OtherSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("Kalmar code attempted to access unknown Space memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("Kalmar code attempted to access unknown Space memory"); }
};

//----------------------------------------------------------------------------
/** Running in HostSpace attempting to access KalmarSpace */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::KalmarSpace >
{
  enum { value = false };
  inline static void verify( void ) __CPU__ __HC__ { KalmarSpace::access_error(); }
  inline static void verify( const void * p ) __CPU__ __HC__ { KalmarSpace::access_error(p); }
};
} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

template<>
class SharedAllocationRecord< Kokkos::KalmarSpace , void >
  : public SharedAllocationRecord< void , void >
{
private:

  friend Kokkos::KalmarSpace;

  typedef SharedAllocationRecord< void , void >  RecordBase ;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

  /**\brief  Root record for tracked allocations from this KalmarSpace instance */
  static RecordBase s_root_record ;
  const Kokkos::KalmarSpace m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_space() {}

  SharedAllocationRecord( const Kokkos::KalmarSpace        & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  std::string get_label() const ;

  KOKKOS_INLINE_FUNCTION static
  SharedAllocationRecord * allocate( const Kokkos::KalmarSpace &  arg_space
                                          , const std::string       &  arg_label
                                          , const size_t               arg_alloc_size );

  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::KalmarSpace & arg_space
                         , const std::string & arg_label
                         , const size_t arg_alloc_size );

  /**\brief  Reallocate tracked memory in the space */
  static
  void * reallocate_tracked( void * const arg_alloc_ptr
                           , const size_t arg_alloc_size );

  /**\brief  Deallocate tracked memory in the space */
  static
  void deallocate_tracked( void * const arg_alloc_ptr );

  static SharedAllocationRecord * get_record( void * arg_alloc_ptr );

  static void print_records( std::ostream & , const Kokkos::KalmarSpace & , bool detail = false );
};

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_HAVE_KALMAR ) */
#endif /* #define KOKKOS_KALMARSPACE_HPP */

