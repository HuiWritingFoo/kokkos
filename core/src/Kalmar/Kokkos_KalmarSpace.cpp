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

#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <Kokkos_Macros.hpp>

/* only compile this file if Kalmar is enabled for Kokkos */
#ifdef KOKKOS_HAVE_KALMAR

#include <Kokkos_Kalmar.hpp>
#include <Kokkos_KalmarSpace.hpp>

#include <impl/Kokkos_BasicAllocators.hpp>
#include <impl/Kokkos_Error.hpp>

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

DeepCopy<KalmarSpace,KalmarSpace,Kalmar>::DeepCopy( void * dst , const void * src , size_t n )
{
  hc::am_copy(dst, (void*)src, n);
}

DeepCopy<HostSpace,KalmarSpace,Kalmar>::DeepCopy( void * dst , const void * src , size_t n )
{
  hc::am_copy(dst, (void*)src, n);
}

DeepCopy<KalmarSpace,HostSpace,Kalmar>::DeepCopy( void * dst , const void * src , size_t n )
{
  hc::am_copy(dst, (void*)src, n);
}

DeepCopy<KalmarSpace,KalmarSpace,Kalmar>::DeepCopy( const Kalmar & instance , void * dst , const void * src , size_t n )
{
  // TODO: multiple devices support in HCC
  hc::am_copy(dst, (void*)src, n);
}

DeepCopy<HostSpace,KalmarSpace,Kalmar>::DeepCopy( const Kalmar & instance , void * dst , const void * src , size_t n )
{
  // TODO: multiple devices support in HCC
  hc::am_copy(dst, (void*)src, n);
}

DeepCopy<KalmarSpace,HostSpace,Kalmar>::DeepCopy( const Kalmar & instance , void * dst , const void * src , size_t n )
{
  // TODO: multiple devices support in HCC
  hc::am_copy(dst, (void*)src, n);
}

void DeepCopyAsyncKalmar( void * dst , const void * src , size_t n) 
{
  //
}
} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

#if ! defined( KOKKOS_USING_EXPERIMENTAL_VIEW )

namespace {



} // unnamed namespace

/*--------------------------------------------------------------------------*/

Impl::AllocationTracker KalmarSpace::allocate_and_track( const std::string & label, const size_t size )
{
  return Impl::AllocationTracker( allocator(), size, label);
}

#endif /* #if ! defined( KOKKOS_USING_EXPERIMENTAL_VIEW ) */

/*--------------------------------------------------------------------------*/



} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

KalmarSpace::KalmarSpace()
  : m_device( 0 )
{
}


void * KalmarSpace::allocate( const size_t arg_alloc_size ) const
{
  return hc::am_alloc( arg_alloc_size, hc::accelerator(), 0 );
}


void KalmarSpace::deallocate( void * const arg_alloc_ptr , const size_t /* arg_alloc_size */ ) const
{
  try {
    hc::am_free( (void*) arg_alloc_ptr );
  } catch(...) {}
}

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

SharedAllocationRecord< void , void >
SharedAllocationRecord< Kokkos::KalmarSpace , void >::s_root_record ;

std::string
SharedAllocationRecord< Kokkos::KalmarSpace , void >::get_label() const
{
  SharedAllocationHeader header ;

  Kokkos::Impl::DeepCopy< Kokkos::HostSpace , Kokkos::KalmarSpace >( & header , RecordBase::head() , sizeof(SharedAllocationHeader) );

  return std::string( header.m_label );
}
SharedAllocationRecord< Kokkos::KalmarSpace , void > *
SharedAllocationRecord< Kokkos::KalmarSpace , void >::
allocate( const Kokkos::KalmarSpace &  arg_space
        , const std::string       &  arg_label
        , const size_t               arg_alloc_size
        )
{
  return new SharedAllocationRecord( arg_space , arg_label , arg_alloc_size );
}

void
SharedAllocationRecord< Kokkos::KalmarSpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< Kokkos::KalmarSpace , void >::
~SharedAllocationRecord()
{
  m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::KalmarSpace , void >::
SharedAllocationRecord( const Kokkos::KalmarSpace & arg_space
                      , const std::string       & arg_label
                      , const size_t              arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void >
      ( & SharedAllocationRecord< Kokkos::KalmarSpace , void >::s_root_record
      , reinterpret_cast<SharedAllocationHeader*>( arg_space.allocate( sizeof(SharedAllocationHeader) + arg_alloc_size ) )
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , arg_dealloc
      )
  , m_space( arg_space )
{
  SharedAllocationHeader header ;

  // Fill in the Header information
  header.m_record = static_cast< SharedAllocationRecord< void , void > * >( this );

  strncpy( header.m_label
          , arg_label.c_str()
          , SharedAllocationHeader::maximum_label_length
          );

  // Copy to device memory
  Kokkos::Impl::DeepCopy<KalmarSpace,HostSpace>::DeepCopy( RecordBase::m_alloc_ptr , & header , sizeof(SharedAllocationHeader) );
}
//----------------------------------------------------------------------------

void * SharedAllocationRecord< Kokkos::KalmarSpace , void >::
allocate_tracked( const Kokkos::KalmarSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_space , arg_alloc_label , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< Kokkos::KalmarSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< Kokkos::KalmarSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->m_space , r_old->get_label() , arg_alloc_size );

  Kokkos::Impl::DeepCopy<KalmarSpace,KalmarSpace>( r_new->data() , r_old->data()
                                             , std::min( r_old->size() , r_new->size() ) );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}


//----------------------------------------------------------------------------

SharedAllocationRecord< Kokkos::KalmarSpace , void > *
SharedAllocationRecord< Kokkos::KalmarSpace , void >::get_record( void * alloc_ptr )
{
  using Header     = SharedAllocationHeader ;
  using RecordBase = SharedAllocationRecord< void , void > ;
  using RecordKalmar = SharedAllocationRecord< Kokkos::KalmarSpace , void > ;

#if 0
  // Copy the header from the allocation
  Header head ;

  Header const * const head_kalmar = alloc_ptr ? Header::get_header( alloc_ptr ) : (Header*) 0 ;

  if ( alloc_ptr ) {
    Kokkos::Impl::DeepCopy<HostSpace,KalmarSpace>::DeepCopy( & head , head_kalmar , sizeof(SharedAllocationHeader) );
  }

  RecordKalmar * const record = alloc_ptr ? static_cast< RecordKalmar * >( head.m_record ) : (RecordKalmar *) 0 ;

  if ( ! alloc_ptr || record->m_alloc_ptr != head_kalmar ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::KalmarSpace , void >::get_record ERROR" ) );
  }

#else

  // Iterate the list to search for the record among all allocations
  // requires obtaining the root of the list and then locking the list.

  RecordKalmar * const record = static_cast< RecordKalmar * >( RecordBase::find( & s_root_record , alloc_ptr ) );

  if ( record == 0 ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::KalmarSpace , void >::get_record ERROR" ) );
  }

#endif

  return record ;
}

// Iterate records to print orphaned memory ...
void
SharedAllocationRecord< Kokkos::KalmarSpace , void >::
print_records( std::ostream & s , const Kokkos::KalmarSpace & space , bool detail )
{
  SharedAllocationRecord< void , void > * r = & s_root_record ;

  char buffer[256] ;

  SharedAllocationHeader head ;

  if ( detail ) {
    do {
      if ( r->m_alloc_ptr ) {
        Kokkos::Impl::DeepCopy<HostSpace,KalmarSpace>::DeepCopy( & head , r->m_alloc_ptr , sizeof(SharedAllocationHeader) );
      }
      else {
        head.m_label[0] = 0 ;
      }

      //Formatting dependent on sizeof(uintptr_t)
      const char * format_string;

      if (sizeof(uintptr_t) == sizeof(unsigned long)) { 
        format_string = "Kalmar addr( 0x%.12lx ) list( 0x%.12lx 0x%.12lx ) extent[ 0x%.12lx + %.8ld ] count(%d) dealloc(0x%.12lx) %s\n";
      }
      else if (sizeof(uintptr_t) == sizeof(unsigned long long)) { 
        format_string = "Kalmar addr( 0x%.12llx ) list( 0x%.12llx 0x%.12llx ) extent[ 0x%.12llx + %.8ld ] count(%d) dealloc(0x%.12llx) %s\n";
      }

      snprintf( buffer , 256 
              , format_string
              , reinterpret_cast<uintptr_t>( r )
              , reinterpret_cast<uintptr_t>( r->m_prev )
              , reinterpret_cast<uintptr_t>( r->m_next )
              , reinterpret_cast<uintptr_t>( r->m_alloc_ptr )
              , r->m_alloc_size
              , r->m_count
              , reinterpret_cast<uintptr_t>( r->m_dealloc )
              , head.m_label
              );
      std::cout << buffer ;
      r = r->m_next ;
    } while ( r != & s_root_record );
  }
  else {
    do {
      if ( r->m_alloc_ptr ) {

        Kokkos::Impl::DeepCopy<HostSpace,KalmarSpace>::DeepCopy( & head , r->m_alloc_ptr , sizeof(SharedAllocationHeader) );

        //Formatting dependent on sizeof(uintptr_t)
        const char * format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) { 
          format_string = "Kalmar [ 0x%.12lx + %ld ] %s\n";
        }
        else if (sizeof(uintptr_t) == sizeof(unsigned long long)) { 
          format_string = "Kalmar [ 0x%.12llx + %ld ] %s\n";
        }

        snprintf( buffer , 256 
                , format_string
                , reinterpret_cast< uintptr_t >( r->data() )
                , r->size()
                , head.m_label
                );
      }
      else {
        snprintf( buffer , 256 , "Kalmar [ 0 + 0 ]\n" );
      }
      std::cout << buffer ;
      r = r->m_next ;
    } while ( r != & s_root_record );
  }
}
} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace {
}

namespace Impl {


}
}
#endif // KOKKOS_HAVE_KALMAR

