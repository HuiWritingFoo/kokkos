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

#ifndef KOKKOS_FUNCTORADAPTER_HPP
#define KOKKOS_FUNCTORADAPTER_HPP

#include <cstddef>
#include <Kokkos_Core_fwd.hpp>
#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Tags.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ArgTag , class Enable = void >
struct FunctorDeclaresValueType : public Impl::false_type {};

template< class FunctorType , class ArgTag >
struct FunctorDeclaresValueType< FunctorType , ArgTag
                               , typename Impl::enable_if_type< typename FunctorType::value_type >::type >
  : public Impl::true_type {};


/** \brief  Query Functor and execution policy argument tag for value type.
 *
 *  If C++11 enabled and 'value_type' is not explicitly declared then attempt
 *  to deduce the type from FunctorType::operator().
 */
template< class FunctorType , class ArgTag , bool Dec = FunctorDeclaresValueType<FunctorType,ArgTag>::value >
struct FunctorValueTraits
{
  typedef void value_type ;
  typedef void pointer_type ;
  typedef void reference_type ;

  enum { StaticValueSize = 0 };

  KOKKOS_FORCEINLINE_FUNCTION static
  unsigned value_count( const FunctorType & ) { return 0 ; }

  KOKKOS_FORCEINLINE_FUNCTION static
  unsigned value_size( const FunctorType & ) { return 0 ; }
};

template<class ArgTag>
struct FunctorValueTraits<void, ArgTag,false>
{
  typedef void reference_type;
};

/** \brief  FunctorType::value_type is explicitly declared so use it.
 *
 * Two options for declaration
 *
 *   1) A plain-old-data (POD) type
 *        typedef {pod_type} value_type ;
 *
 *   2) An array of POD of a runtime specified count.
 *        typedef {pod_type} value_type[] ;
 *        const unsigned     value_count ;
 */
template< class FunctorType , class ArgTag >
struct FunctorValueTraits< FunctorType , ArgTag , true /* == exists FunctorType::value_type */ >
{
  typedef typename Impl::remove_extent< typename FunctorType::value_type >::type  value_type ;

  // If not an array then what is the sizeof(value_type)
  enum { StaticValueSize = Impl::is_array< typename FunctorType::value_type >::value ? 0 : sizeof(value_type) };

  typedef value_type                 * pointer_type ;

  // The reference_type for an array is 'value_type *'
  // The reference_type for a single value is 'value_type &'

  typedef typename Impl::if_c< ! StaticValueSize , value_type *
                                                 , value_type & >::type  reference_type ;

  // Number of values if single value
  template< class F >
  KOKKOS_FORCEINLINE_FUNCTION static
  typename Impl::enable_if< Impl::is_same<F,FunctorType>::value && StaticValueSize , unsigned >::type
    value_count( const F & ) { return 1 ; }

  // Number of values if an array, protect via templating because 'f.value_count'
  // will only exist when the functor declares the value_type to be an array.
  template< class F >
  KOKKOS_FORCEINLINE_FUNCTION static
  typename Impl::enable_if< Impl::is_same<F,FunctorType>::value && ! StaticValueSize , unsigned >::type
    value_count( const F & f ) { return f.value_count ; }

  // Total size of the value
  KOKKOS_INLINE_FUNCTION static
  unsigned value_size( const FunctorType & f ) { return value_count( f ) * sizeof(value_type) ; }
};


#if defined( KOKKOS_HAVE_CXX11 )

template< class FunctorType , class ArgTag >
struct FunctorValueTraits< FunctorType
                         , ArgTag
                         , false  /* == exists FunctorType::value_type */
                         >
{
private:

  struct VOIDTAG {};   // Allow declaration of non-matching operator() with void argument tag.
  struct REJECTTAG {}; // Reject tagged operator() when using non-tagged execution policy.

  typedef typename
    Impl::if_c< Impl::is_same< ArgTag , void >::value , VOIDTAG , ArgTag >::type tag_type ;

  //----------------------------------------
  // parallel_for operator without a tag:

  template< class ArgMember >
  KOKKOS_INLINE_FUNCTION
  static VOIDTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( ArgMember ) const ) {}

  template< class ArgMember >
  KOKKOS_INLINE_FUNCTION
  static VOIDTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( const ArgMember & ) const ) {}

  template< class TagType , class ArgMember >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( TagType , ArgMember ) const ) {}

  template< class TagType , class ArgMember >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( TagType , const ArgMember & ) const ) {}

  template< class TagType , class ArgMember >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( const TagType & , ArgMember ) const ) {}

  template< class TagType , class ArgMember >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( const TagType & , const ArgMember & ) const ) {}

  //----------------------------------------
  // parallel_for operator with a tag:

  template< class ArgMember >
  KOKKOS_INLINE_FUNCTION
  static VOIDTAG deduce_reduce_type( tag_type , void (FunctorType::*)( tag_type , ArgMember ) const ) {}

  template< class ArgMember >
  KOKKOS_INLINE_FUNCTION
  static VOIDTAG deduce_reduce_type( tag_type , void (FunctorType::*)( const tag_type & , ArgMember ) const ) {}

  template< class ArgMember >
  KOKKOS_INLINE_FUNCTION
  static VOIDTAG deduce_reduce_type( tag_type , void (FunctorType::*)( tag_type , const ArgMember & ) const ) {}

  template< class ArgMember >
  KOKKOS_INLINE_FUNCTION
  static VOIDTAG deduce_reduce_type( tag_type , void (FunctorType::*)( const tag_type & , const ArgMember & ) const ) {}

  //----------------------------------------
  // parallel_reduce operator without a tag:

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( VOIDTAG , void (FunctorType::*)( ArgMember , T & ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( VOIDTAG , void (FunctorType::*)( const ArgMember & , T & ) const ) {}

  template< class TagType , class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( TagType , ArgMember , T & ) const ) {}

  template< class TagType , class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( TagType , const ArgMember & , T & ) const ) {}

  template< class TagType , class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( const TagType & , ArgMember , T & ) const ) {}

  template< class TagType , class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( const TagType & , const ArgMember & , T & ) const ) {}

  //----------------------------------------
  // parallel_reduce operator with a tag:

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( tag_type , void (FunctorType::*)( tag_type , ArgMember , T & ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( tag_type , void (FunctorType::*)( const tag_type & , ArgMember , T & ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( tag_type , void (FunctorType::*)( tag_type , const ArgMember & , T & ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( tag_type , void (FunctorType::*)( const tag_type & , const ArgMember & , T & ) const ) {}

  //----------------------------------------
  // parallel_scan operator without a tag:

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( VOIDTAG , void (FunctorType::*)( ArgMember , T & , bool ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( VOIDTAG , void (FunctorType::*)( const ArgMember & , T & , bool ) const ) {}

  template< class TagType , class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( TagType , ArgMember , T & , bool ) const ) {}

  template< class TagType , class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( TagType , const ArgMember & , T & , bool ) const ) {}

  template< class TagType , class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( const TagType & , ArgMember , T & , bool ) const ) {}

  template< class TagType , class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( const TagType & , const ArgMember & , T & , bool ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( VOIDTAG , void (FunctorType::*)( ArgMember , T & , const bool& ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( VOIDTAG , void (FunctorType::*)( const ArgMember & , T & , const bool& ) const ) {}

  template< class TagType , class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( TagType , ArgMember , T & , const bool& ) const ) {}

  template< class TagType , class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( TagType , const ArgMember & , T & , const bool& ) const ) {}

  template< class TagType , class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( const TagType & , ArgMember , T & , const bool& ) const ) {}

  template< class TagType , class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static REJECTTAG deduce_reduce_type( VOIDTAG , void (FunctorType::*)( const TagType & , const ArgMember & , T & , const bool& ) const ) {}
  //----------------------------------------
  // parallel_scan operator with a tag:

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( tag_type , void (FunctorType::*)( tag_type , ArgMember , T & , bool ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( tag_type , void (FunctorType::*)( const tag_type & , ArgMember , T & , bool ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( tag_type , void (FunctorType::*)( tag_type , const ArgMember& , T & , bool ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( tag_type , void (FunctorType::*)( const tag_type & , const ArgMember& , T & , bool ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( tag_type , void (FunctorType::*)( tag_type , ArgMember , T & , const bool& ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( tag_type , void (FunctorType::*)( const tag_type & , ArgMember , T & , const bool& ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( tag_type , void (FunctorType::*)( tag_type , const ArgMember& , T & , const bool& ) const ) {}

  template< class ArgMember , class T >
  KOKKOS_INLINE_FUNCTION
  static T deduce_reduce_type( tag_type , void (FunctorType::*)( const tag_type & , const ArgMember& , T & , const bool& ) const ) {}
  //----------------------------------------

  typedef decltype( deduce_reduce_type( tag_type() , & FunctorType::operator() ) ) ValueType ;

  enum { IS_VOID   = Impl::is_same<VOIDTAG  ,ValueType>::value };
  enum { IS_REJECT = Impl::is_same<REJECTTAG,ValueType>::value };

public:

  typedef typename Impl::if_c< IS_VOID || IS_REJECT , void , ValueType   >::type  value_type ;
  typedef typename Impl::if_c< IS_VOID || IS_REJECT , void , ValueType * >::type  pointer_type ;
  typedef typename Impl::if_c< IS_VOID || IS_REJECT , void , ValueType & >::type  reference_type ;

  enum { StaticValueSize = IS_VOID || IS_REJECT ? 0 : sizeof(ValueType) };

  KOKKOS_FORCEINLINE_FUNCTION static
  unsigned value_size( const FunctorType & ) { return StaticValueSize ; }

  KOKKOS_FORCEINLINE_FUNCTION static
  unsigned value_count( const FunctorType & ) { return IS_VOID || IS_REJECT ? 0 : 1 ; }
};

#endif /* #if defined( KOKKOS_HAVE_CXX11 ) */

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

// Function signatures for FunctorType::init function with a tag and not an array
template< class FunctorType , class ArgTag , bool IsArray = 0 == FunctorValueTraits<FunctorType,ArgTag>::StaticValueSize >
struct FunctorValueInitFunction {

  typedef typename FunctorValueTraits<FunctorType,ArgTag>::value_type value_type ;

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type & ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type & ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , value_type & ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , value_type & ) );

  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type volatile & ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type volatile & ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , value_type volatile & ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , value_type volatile & ) );
};

// Function signatures for FunctorType::init function with a tag and is an array
template< class FunctorType , class ArgTag >
struct FunctorValueInitFunction< FunctorType , ArgTag , true > {

  typedef typename FunctorValueTraits<FunctorType,ArgTag>::value_type value_type ;

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type * ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type * ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , value_type * ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , value_type * ) );

  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type volatile * ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type volatile * ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , value_type volatile * ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , value_type volatile * ) );
};

// Function signatures for FunctorType::init function without a tag and not an array
template< class FunctorType >
struct FunctorValueInitFunction< FunctorType , void , false > {

  typedef typename FunctorValueTraits<FunctorType,void>::reference_type value_type ;

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( value_type & ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( value_type & ) );

  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( value_type volatile & ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( value_type volatile & ) );
};

// Function signatures for FunctorType::init function without a tag and is an array
template< class FunctorType >
struct FunctorValueInitFunction< FunctorType , void , true > {

  typedef typename FunctorValueTraits<FunctorType,void>::reference_type value_type ;

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( value_type * ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( value_type * ) );

  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( value_type volatile * ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( value_type volatile * ) );
};

// Adapter for value initialization function.
// If a proper FunctorType::init is declared then use it,
// otherwise use default constructor.
template< class FunctorType , class ArgTag
        , class T = typename FunctorValueTraits<FunctorType,ArgTag>::reference_type
        , class Enable = void >
struct FunctorValueInit ;

/* No 'init' function provided for single value */
template< class FunctorType , class ArgTag , class T , class Enable >
struct FunctorValueInit< FunctorType , ArgTag , T & , Enable >
{
  template< typename U >
  KOKKOS_FORCEINLINE_FUNCTION static
  T & init( const FunctorType & f , U * p )
    {
      new((void*)p) T();
      return *((T*)p);
    };
};

/* No 'init' function provided for array value */
template< class FunctorType , class ArgTag , class T , class Enable >
struct FunctorValueInit< FunctorType , ArgTag , T * , Enable >
{
  template< typename U >
  KOKKOS_FORCEINLINE_FUNCTION static
  T * init( const FunctorType & f , U * p )
    {
      const int n = FunctorValueTraits< FunctorType , ArgTag >::value_count(f);
      for ( int i = 0 ; i < n ; ++i ) { new( (void*)(((T*)p) + i) ) T(); }
      return (T*)p ;
    }
};

/* 'init' function provided for single value */
template< class FunctorType , class T >
struct FunctorValueInit
  < FunctorType
  , void
  , T &
    // First  substitution failure when FunctorType::init does not exist.
#if defined( KOKKOS_HAVE_CXX11 )
    // Second substitution failure when FunctorType::init is not compatible.
  , decltype( FunctorValueInitFunction< FunctorType , void >::enable_if( & FunctorType::init ) )
#else
  , typename Impl::enable_if< 0 < sizeof( & FunctorType::init ) >::type
#endif
  >
{
  template< typename U >
  KOKKOS_FORCEINLINE_FUNCTION static
  T & init( const FunctorType & f , U * p )
    { f.init( *((T*)p) ); return *((T*)p) ; }
};

/* 'init' function provided for array value */
template< class FunctorType , class T >
struct FunctorValueInit
  < FunctorType
  , void
  , T *
    // First  substitution failure when FunctorType::init does not exist.
#if defined( KOKKOS_HAVE_CXX11 )
    // Second substitution failure when FunctorType::init is not compatible
  , decltype( FunctorValueInitFunction< FunctorType , void >::enable_if( & FunctorType::init ) )
#else
  , typename Impl::enable_if< 0 < sizeof( & FunctorType::init ) >::type
#endif
  >
{
  template< typename U >
  KOKKOS_FORCEINLINE_FUNCTION static
  T * init( const FunctorType & f , U * p )
    { f.init( (T*)p ); return (T*)p ; }
};

/* 'init' function provided for single value */
template< class FunctorType , class ArgTag , class T >
struct FunctorValueInit
  < FunctorType
  , ArgTag
  , T &
    // First  substitution failure when FunctorType::init does not exist.
#if defined( KOKKOS_HAVE_CXX11 )
    // Second substitution failure when FunctorType::init is not compatible.
  , decltype( FunctorValueInitFunction< FunctorType , ArgTag >::enable_if( & FunctorType::init ) )
#else
  , typename Impl::enable_if< 0 < sizeof( & FunctorType::init ) >::type
#endif
  >
{
  template< typename U >
  KOKKOS_FORCEINLINE_FUNCTION static
  T & init( const FunctorType & f , U * p )
    { f.init( ArgTag() , *((T*)p) ); return *((T*)p) ; }
};

/* 'init' function provided for array value */
template< class FunctorType , class ArgTag , class T >
struct FunctorValueInit
  < FunctorType
  , ArgTag
  , T *
    // First  substitution failure when FunctorType::init does not exist.
#if defined( KOKKOS_HAVE_CXX11 )
    // Second substitution failure when FunctorType::init is not compatible
  , decltype( FunctorValueInitFunction< FunctorType , ArgTag >::enable_if( & FunctorType::init ) )
#else
  , typename Impl::enable_if< 0 < sizeof( & FunctorType::init ) >::type
#endif
  >
{
  template< typename U >
  KOKKOS_FORCEINLINE_FUNCTION static
  T * init( const FunctorType & f , U * p )
    { f.init( ArgTag() , (T*)p ); return (T*)p ; }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

// Signatures for compatible FunctorType::join with tag and not an array
template< class FunctorType , class ArgTag , bool IsArray = 0 == FunctorValueTraits<FunctorType,ArgTag>::StaticValueSize >
struct FunctorValueJoinFunction {

  typedef typename FunctorValueTraits<FunctorType,ArgTag>::value_type value_type ;

  typedef       volatile value_type & vref_type ;
  typedef const volatile value_type & cvref_type ;

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , vref_type , cvref_type ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , vref_type , cvref_type ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , vref_type , cvref_type ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , vref_type , cvref_type ) );
};

// Signatures for compatible FunctorType::join with tag and is an array
template< class FunctorType , class ArgTag >
struct FunctorValueJoinFunction< FunctorType , ArgTag , true > {

  typedef typename FunctorValueTraits<FunctorType,ArgTag>::value_type value_type ;

  typedef       volatile value_type * vptr_type ;
  typedef const volatile value_type * cvptr_type ;

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , vptr_type , cvptr_type ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , vptr_type , cvptr_type ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , vptr_type , cvptr_type ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , vptr_type , cvptr_type ) );
};

// Signatures for compatible FunctorType::join without tag and not an array
template< class FunctorType >
struct FunctorValueJoinFunction< FunctorType , void , false > {

  typedef typename FunctorValueTraits<FunctorType,void>::value_type value_type ;

  typedef       volatile value_type & vref_type ;
  typedef const volatile value_type & cvref_type ;

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( vref_type , cvref_type ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( vref_type , cvref_type ) );
};

// Signatures for compatible FunctorType::join without tag and is an array
template< class FunctorType >
struct FunctorValueJoinFunction< FunctorType , void , true > {

  typedef typename FunctorValueTraits<FunctorType,void>::value_type value_type ;

  typedef       volatile value_type * vptr_type ;
  typedef const volatile value_type * cvptr_type ;

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( vptr_type , cvptr_type ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( vptr_type , cvptr_type ) );
};


template< class FunctorType , class ArgTag
        , class T = typename FunctorValueTraits<FunctorType,ArgTag>::reference_type
        , class Enable = void >
struct FunctorValueJoin ;

/* No 'join' function provided, single value */
template< class FunctorType , class ArgTag , class T , class Enable >
struct FunctorValueJoin< FunctorType , ArgTag , T & , Enable >
{
  template< typename U >
  KOKKOS_FORCEINLINE_FUNCTION static
  void join( const FunctorType & f , volatile U * const lhs , const volatile U * const rhs )
    {
      *((volatile T*)lhs) += *((const volatile T*)rhs);
    }
};

/* No 'join' function provided, array of values */
template< class FunctorType , class ArgTag , class T , class Enable >
struct FunctorValueJoin< FunctorType , ArgTag , T * , Enable >
{
  template< typename U >
  KOKKOS_FORCEINLINE_FUNCTION static
  void join( const FunctorType & f , volatile U * const lhs , const volatile U * const rhs )
    {
      const int n = FunctorValueTraits<FunctorType,ArgTag>::value_count(f);

      for ( int i = 0 ; i < n ; ++i ) { ((volatile T*)lhs)[i] += ((const volatile T*)rhs)[i]; }
    }
};

/* 'join' function provided, single value */
template< class FunctorType , class ArgTag , class T >
struct FunctorValueJoin
  < FunctorType
  , ArgTag
  , T &
    // First  substitution failure when FunctorType::join does not exist.
#if defined( KOKKOS_HAVE_CXX11 )
    // Second substitution failure when enable_if( & Functor::join ) does not exist
  , decltype( FunctorValueJoinFunction< FunctorType , ArgTag >::enable_if( & FunctorType::join ) )
#else
  , typename Impl::enable_if< 0 < sizeof( & FunctorType::join ) >::type
#endif
  >
{
  template< typename U >
  KOKKOS_FORCEINLINE_FUNCTION static
  void join( const FunctorType & f , volatile U * const lhs , const volatile U * const rhs )
    {
      f.join( ArgTag() , *((volatile T *)lhs) , *((const volatile T *)rhs) );
    }
};

/* 'join' function provided, no tag, single value */
template< class FunctorType , class T >
struct FunctorValueJoin
  < FunctorType
  , void
  , T &
    // First  substitution failure when FunctorType::join does not exist.
#if defined( KOKKOS_HAVE_CXX11 )
    // Second substitution failure when enable_if( & Functor::join ) does not exist
  , decltype( FunctorValueJoinFunction< FunctorType , void >::enable_if( & FunctorType::join ) )
#else
  , typename Impl::enable_if< 0 < sizeof( & FunctorType::join ) >::type
#endif
  >
{
  template< typename U >
  KOKKOS_FORCEINLINE_FUNCTION static
  void join( const FunctorType & f , volatile U * const lhs , const volatile U * const rhs )
    {
      f.join( *((volatile T *)lhs) , *((const volatile T *)rhs) );
    }
};

/* 'join' function provided for array value */
template< class FunctorType , class ArgTag , class T >
struct FunctorValueJoin
  < FunctorType
  , ArgTag
  , T *
    // First  substitution failure when FunctorType::join does not exist.
#if defined( KOKKOS_HAVE_CXX11 )
    // Second substitution failure when enable_if( & Functor::join ) does not exist
  , decltype( FunctorValueJoinFunction< FunctorType , ArgTag >::enable_if( & FunctorType::join ) )
#else
  , typename Impl::enable_if< 0 < sizeof( & FunctorType::join ) >::type
#endif
  >
{
  template< typename U >
  KOKKOS_FORCEINLINE_FUNCTION static
  void join( const FunctorType & f , volatile U * const lhs , const volatile U * const rhs )
    {
      f.join( ArgTag() , (volatile T *)lhs , (const volatile T *)rhs );
    }
};

/* 'join' function provided, no tag, array value */
template< class FunctorType , class T >
struct FunctorValueJoin
  < FunctorType
  , void
  , T *
    // First  substitution failure when FunctorType::join does not exist.
#if defined( KOKKOS_HAVE_CXX11 )
    // Second substitution failure when enable_if( & Functor::join ) does not exist
  , decltype( FunctorValueJoinFunction< FunctorType , void >::enable_if( & FunctorType::join ) )
#else
  , typename Impl::enable_if< 0 < sizeof( & FunctorType::join ) >::type
#endif
  >
{
  template< typename U >
  KOKKOS_FORCEINLINE_FUNCTION static
  void join( const FunctorType & f , volatile U * const lhs , const volatile U * const rhs )
    {
      f.join( (volatile T *)lhs , (const volatile T *)rhs );
    }
};

} // namespace Impl
} // namespace Kokkos

namespace Kokkos {

namespace Impl {

#if defined( KOKKOS_HAVE_CXX11 )

  template<typename ValueType, class JoinOp, class Enable = void>
  struct JoinLambdaAdapter {
    typedef ValueType value_type;
    const JoinOp& lambda;
    KOKKOS_INLINE_FUNCTION
    JoinLambdaAdapter(const JoinOp& lambda_):lambda(lambda_) {}

    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst, const volatile value_type& src) const {
      lambda(dst,src);
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type& dst, const value_type& src) const {
      lambda(dst,src);
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (volatile value_type& dst, const volatile value_type& src) const {
      lambda(dst,src);
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (value_type& dst, const value_type& src) const {
      lambda(dst,src);
    }
  };

  template<typename ValueType, class JoinOp>
  struct JoinLambdaAdapter<ValueType, JoinOp, decltype( FunctorValueJoinFunction< JoinOp , void >::enable_if( & JoinOp::join ) )> {
    typedef ValueType value_type;
    typedef StaticAssertSame<ValueType,typename JoinOp::value_type> assert_value_types_match;
    const JoinOp& lambda;
    KOKKOS_INLINE_FUNCTION
    JoinLambdaAdapter(const JoinOp& lambda_):lambda(lambda_) {}

    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst, const volatile value_type& src) const {
      lambda.join(dst,src);
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type& dst, const value_type& src) const {
      lambda.join(dst,src);
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (volatile value_type& dst, const volatile value_type& src) const {
      lambda.join(dst,src);
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (value_type& dst, const value_type& src) const {
      lambda.join(dst,src);
    }
  };

#endif

  template<typename ValueType>
  struct JoinAdd {
    typedef ValueType value_type;

    KOKKOS_INLINE_FUNCTION
    JoinAdd() {}

    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst, const volatile value_type& src) const {
      dst+=src;
    }
    KOKKOS_INLINE_FUNCTION
    void operator() (value_type& dst, const value_type& src) const {
      dst+=src;
    }
    KOKKOS_INLINE_FUNCTION
    void operator() (volatile value_type& dst, const volatile value_type& src) const {
      dst+=src;
    }
  };

}
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ArgTag
        , class T = typename FunctorValueTraits<FunctorType,ArgTag>::reference_type >
struct FunctorValueOps ;

template< class FunctorType , class ArgTag , class T >
struct FunctorValueOps< FunctorType , ArgTag , T & >
{
  KOKKOS_FORCEINLINE_FUNCTION static
  T * pointer( T & r ) { return & r ; }

  KOKKOS_FORCEINLINE_FUNCTION static
  T & reference( void * p ) { return *((T*)p); }

  KOKKOS_FORCEINLINE_FUNCTION static
  void copy( const FunctorType & , void * const lhs , const void * const rhs )
    { *((T*)lhs) = *((const T*)rhs); }
};

/* No 'join' function provided, array of values */
template< class FunctorType , class ArgTag , class T >
struct FunctorValueOps< FunctorType , ArgTag , T * >
{
  KOKKOS_FORCEINLINE_FUNCTION static
  T * pointer( T * p ) { return p ; }

  KOKKOS_FORCEINLINE_FUNCTION static
  T * reference( void * p ) { return ((T*)p); }

  KOKKOS_FORCEINLINE_FUNCTION static
  void copy( const FunctorType & f , void * const lhs , const void * const rhs )
    {
      const int n = FunctorValueTraits<FunctorType,ArgTag>::value_count(f);
      for ( int i = 0 ; i < n ; ++i ) { ((T*)lhs)[i] = ((const T*)rhs)[i]; }
    }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

// Compatible functions for 'final' function and value_type not an array
template< class FunctorType , class ArgTag , bool IsArray = 0 == FunctorValueTraits<FunctorType,ArgTag>::StaticValueSize >
struct FunctorFinalFunction {

  typedef typename FunctorValueTraits<FunctorType,ArgTag>::value_type value_type ;

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type & ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type & ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type & ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type & ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , value_type & ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , value_type & ) );

  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type volatile & ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type volatile & ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type volatile & ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type volatile & ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , value_type volatile & ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , value_type volatile & ) );

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type const & ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type const & ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type const & ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type const & ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , value_type const & ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , value_type const & ) );

  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type const volatile & ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type const volatile & ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type const volatile & ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type const volatile & ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , value_type const volatile & ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , value_type const volatile & ) );
};

// Compatible functions for 'final' function and value_type is an array
template< class FunctorType , class ArgTag >
struct FunctorFinalFunction< FunctorType , ArgTag , true > {

  typedef typename FunctorValueTraits<FunctorType,ArgTag>::value_type value_type ;

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type * ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type * ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type * ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type * ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , value_type * ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , value_type * ) );

  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type volatile * ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type volatile * ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type volatile * ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type volatile * ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , value_type volatile * ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , value_type volatile * ) );

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type const * ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type const * ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type const * ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type const * ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , value_type const * ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , value_type const * ) );

  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type const volatile * ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type const volatile * ) const );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , value_type const volatile * ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , value_type const volatile * ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , value_type const volatile * ) );
  // KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , value_type const volatile * ) );
};

template< class FunctorType >
struct FunctorFinalFunction< FunctorType , void , false > {

  typedef typename FunctorValueTraits<FunctorType,void>::value_type value_type ;

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( value_type & ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( value_type & ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( value_type & ) );

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( const value_type & ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( const value_type & ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( const value_type & ) );
};

template< class FunctorType >
struct FunctorFinalFunction< FunctorType , void , true > {

  typedef typename FunctorValueTraits<FunctorType,void>::value_type value_type ;

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( value_type * ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( value_type * ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( value_type * ) );

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( const value_type * ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( const value_type * ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( const value_type * ) );
};

/* No 'final' function provided */
template< class FunctorType , class ArgTag
        , class ResultType = typename FunctorValueTraits<FunctorType,ArgTag>::reference_type
        , class Enable = void >
struct FunctorFinal
{
  KOKKOS_FORCEINLINE_FUNCTION static
  void final( const FunctorType & , void * ) {}
};

/* 'final' function provided */
template< class FunctorType , class ArgTag , class T >
struct FunctorFinal
  < FunctorType
  , ArgTag
  , T &
    // First  substitution failure when FunctorType::final does not exist.
#if defined( KOKKOS_HAVE_CXX11 )
    // Second substitution failure when enable_if( & Functor::final ) does not exist
  , decltype( FunctorFinalFunction< FunctorType , ArgTag >::enable_if( & FunctorType::final ) )
#else
  , typename Impl::enable_if< 0 < sizeof( & FunctorType::final ) >::type
#endif
  >
{
  KOKKOS_FORCEINLINE_FUNCTION static
  void final( const FunctorType & f , void * p ) { f.final( *((T*)p) ); }

  KOKKOS_FORCEINLINE_FUNCTION static
  void final( FunctorType & f , void * p ) { f.final( *((T*)p) ); }
};

/* 'final' function provided for array value */
template< class FunctorType , class ArgTag , class T >
struct FunctorFinal
  < FunctorType
  , ArgTag
  , T *
    // First  substitution failure when FunctorType::final does not exist.
#if defined( KOKKOS_HAVE_CXX11 )
    // Second substitution failure when enable_if( & Functor::final ) does not exist
  , decltype( FunctorFinalFunction< FunctorType , ArgTag >::enable_if( & FunctorType::final ) )
#else
  , typename Impl::enable_if< 0 < sizeof( & FunctorType::final ) >::type
#endif
  >
{
  KOKKOS_FORCEINLINE_FUNCTION static
  void final( const FunctorType & f , void * p ) { f.final( (T*)p ); }

  KOKKOS_FORCEINLINE_FUNCTION static
  void final( FunctorType & f , void * p ) { f.final( (T*)p ); }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ArgTag
        , class ReferenceType = typename FunctorValueTraits<FunctorType,ArgTag>::reference_type >
struct FunctorApplyFunction {

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , ReferenceType ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , ReferenceType ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag         , ReferenceType ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ArgTag const & , ReferenceType ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag         , ReferenceType ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ArgTag const & , ReferenceType ) );
};

template< class FunctorType , class ReferenceType >
struct FunctorApplyFunction< FunctorType , void , ReferenceType > {

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ReferenceType ) const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)( ReferenceType ) );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (             *)( ReferenceType ) );
};

template< class FunctorType >
struct FunctorApplyFunction< FunctorType , void , void > {

  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)() const );
  KOKKOS_INLINE_FUNCTION static void enable_if( void (FunctorType::*)() );
};

template< class FunctorType , class ArgTag , class ReferenceType
        , class Enable = void >
struct FunctorApply
{
  KOKKOS_FORCEINLINE_FUNCTION static
  void apply( const FunctorType & , void * ) {}
};

/* 'apply' function provided for void value */
template< class FunctorType , class ArgTag >
struct FunctorApply
  < FunctorType
  , ArgTag
  , void
    // First  substitution failure when FunctorType::apply does not exist.
#if defined( KOKKOS_HAVE_CXX11 )
    // Second substitution failure when enable_if( & Functor::apply ) does not exist
  , decltype( FunctorApplyFunction< FunctorType , ArgTag , void >::enable_if( & FunctorType::apply ) )
#else
  , typename Impl::enable_if< 0 < sizeof( & FunctorType::apply ) >::type
#endif
  >
{
  KOKKOS_FORCEINLINE_FUNCTION static
  void apply( FunctorType & f ) { f.apply(); }

  KOKKOS_FORCEINLINE_FUNCTION static
  void apply( const FunctorType & f ) { f.apply(); }
};

/* 'apply' function provided for single value */
template< class FunctorType , class ArgTag , class T >
struct FunctorApply
  < FunctorType
  , ArgTag
  , T &
    // First  substitution failure when FunctorType::apply does not exist.
#if defined( KOKKOS_HAVE_CXX11 )
    // Second substitution failure when enable_if( & Functor::apply ) does not exist
  , decltype( FunctorApplyFunction< FunctorType , ArgTag >::enable_if( & FunctorType::apply ) )
#else
  , typename Impl::enable_if< 0 < sizeof( & FunctorType::apply ) >::type
#endif
  >
{
  KOKKOS_FORCEINLINE_FUNCTION static
  void apply( const FunctorType & f , void * p ) { f.apply( *((T*)p) ); }

  KOKKOS_FORCEINLINE_FUNCTION static
  void apply( FunctorType & f , void * p ) { f.apply( *((T*)p) ); }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* KOKKOS_FUNCTORADAPTER_HPP */

