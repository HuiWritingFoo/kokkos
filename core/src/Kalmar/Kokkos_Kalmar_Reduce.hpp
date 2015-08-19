/***************************************************************************
*   © 2012,2014 Advanced Micro Devices, Inc. All rights reserved.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/

///////////////////////////////////////////////////////////////////////////////
// AMP REDUCE
//////////////////////////////////////////////////////////////////////////////

#if !defined( KOKKOS_KALMAR_AMP_REDUCE_INL )
#define KOKKOS_KALMAR_AMP_REDUCE_INL


// Issue: taking the address of a 'tile_static' variable
// may not dereference properly ???
#define REDUCE_WAVEFRONT_SIZE 256 //64
#define _REDUCE_STEP(_LENGTH, _IDX, _W) \
if ((_IDX < _W) && ((_IDX + _W) < _LENGTH)) {\
      ValueJoin::join( functor , & scratch[_IDX] , & scratch[ _IDX + _W ] ); \
}\
    t_idx.barrier.wait();

#include <iostream>

#include <algorithm>
#include <type_traits>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Kokkos {
namespace Impl {

// This is the base implementation of reduction that is called by all of the convenience wrappers below.
// first and last must be iterators from a DeviceVector
template< class FunctorType , typename T >
void reduce_enqueue(
  const int           szElements ,
  const FunctorType & functor ,
  T * const output_result ,
  int const output_length )
{
  using namespace hc ;

  typedef Kokkos::Impl::FunctorValueTraits< FunctorType , void > ValueTraits ;
  typedef Kokkos::Impl::FunctorValueInit< FunctorType , void >   ValueInit ;
  typedef Kokkos::Impl::FunctorValueJoin< FunctorType , void >   ValueJoin ;
  typedef Kokkos::Impl::FunctorFinal< FunctorType , void >       ValueFinal ;

  typedef typename ValueTraits::pointer_type   pointer_type ;
  typedef typename ValueTraits::reference_type reference_type ;

  // Prepare to allocate 'sizeof(T) * output_length' reduction scratch
  // space for each thread within the tile.
  ts_allocator tsa ;
  tsa.setDynamicGroupSegmentSize( REDUCE_WAVEFRONT_SIZE * sizeof(T) * output_length );


  int max_ComputeUnits = 32;
  int numTiles = max_ComputeUnits*32;			/* Max no. of WG for Tahiti(32 compute Units) and 32 is the tuning factor that gives good performance*/

  int length = (REDUCE_WAVEFRONT_SIZE*numTiles);

  length = szElements < length ? szElements : length;
  unsigned int residual = length % REDUCE_WAVEFRONT_SIZE;
  length = residual ? (length + REDUCE_WAVEFRONT_SIZE - residual): length ;

  const int numTilesMax =( szElements + REDUCE_WAVEFRONT_SIZE - 1 ) / REDUCE_WAVEFRONT_SIZE ;

  if ( numTilesMax < numTiles ) numTiles = numTilesMax ;

  // For storing tiles' contributions:
  T * const result = new T[numTiles * output_length ];

  hc::extent< 1 > inputExtent(length);
  hc::tiled_extent< 1 >
    tiledExtentReduce = inputExtent.tile(REDUCE_WAVEFRONT_SIZE);

  // AMP doesn't have APIs to get CU capacity. Launchable size is great though.
#if 1
  printf("reduce_enqueue T = \"%s\" szElements %d length %d output_length %d numTiles %d\n"
        , typeid(T).name()
        , szElements
        , length
        , output_length
        , numTiles
        );
#endif
  try
  {
    hc::completion_future fut = hc::parallel_for_each
      ( tiledExtentReduce
      , tsa /* tile-static memory allocator */
      , [ = , & functor , & tsa ]
        ( hc::tiled_index<1> t_idx ) restrict(amp)
        {
          tsa.reset();

          typedef __attribute__((address_space(3))) T shared_T ;

          shared_T * const scratch =
            (shared_T *) tsa.alloc(REDUCE_WAVEFRONT_SIZE*sizeof(T)*output_length);

          int gx = t_idx.global[0];
          int gloId = gx;
          //  Initialize local data store
          //  Index of this member in its work group.
          unsigned int tileIndex = t_idx.local[0];

          // This thread accumulates into designated
          // portion of tile-static memory.
          reference_type accumulator =
            ValueInit::init(functor,scratch+tileIndex*output_length);

          for ( ; gx < szElements ; gx += length ) {
            functor(gx,accumulator);
          }

          t_idx.barrier.wait();

          // Reduce within this tile:
#if 1
          _REDUCE_STEP(REDUCE_WAVEFRONT_SIZE * output_length , tileIndex * output_length , 128 * output_length );
          _REDUCE_STEP(REDUCE_WAVEFRONT_SIZE * output_length , tileIndex * output_length , 64 * output_length );
          _REDUCE_STEP(REDUCE_WAVEFRONT_SIZE * output_length , tileIndex * output_length , 32 * output_length );
          _REDUCE_STEP(REDUCE_WAVEFRONT_SIZE * output_length , tileIndex * output_length , 16 * output_length );
          _REDUCE_STEP(REDUCE_WAVEFRONT_SIZE * output_length , tileIndex * output_length , 8 * output_length );
          _REDUCE_STEP(REDUCE_WAVEFRONT_SIZE * output_length , tileIndex * output_length , 4 * output_length );
          _REDUCE_STEP(REDUCE_WAVEFRONT_SIZE * output_length , tileIndex * output_length , 2 * output_length );
          _REDUCE_STEP(REDUCE_WAVEFRONT_SIZE * output_length , tileIndex * output_length , 1 * output_length );
#endif

          //  Abort threads that are passed the end of the input vector
          if (gloId >= szElements)
          	return;

          //  Write only the single reduced value for the entire workgroup
          if (tileIndex == 0)
          {
            const int beg = t_idx.tile[0] * output_length ;

            for ( int i = 0 ; i < output_length ; ++i ) {
              result[ beg + i ] = ((pointer_type)scratch)[i];
            }
          }

       });
       // End of hc::parallel_for_each
       fut.wait();

       std::cout << "result[0] = {" ;
       for ( int i = 0 ; i < output_length ; ++i ) {
         std::cout << " " ;
         std::cout << result[i] ;
         output_result[i] = result[i];
       }
       std::cout << " }" << std::endl ;

       for(int i = 1; i < numTiles; ++i)
         {
           std::cout << "join result[" << i << "] = {" ;
           for ( int j = 0 ; j < output_length ; ++j ) {
             std::cout << " " ;
             std::cout << result[i*output_length+j] ;
           }
           std::cout << " }" << std::endl ;
           ValueJoin::join( functor , output_result, result + i * output_length );
         }

       delete[] result ;

       ValueFinal::final( functor , output_result );
  }
  catch(std::exception &e)
  {
    throw e ;
  }


}

}} //end of namespace Kokkos::Impl

#endif /* #if !defined( KOKKOS_KALMAR_AMP_REDUCE_INL ) */

