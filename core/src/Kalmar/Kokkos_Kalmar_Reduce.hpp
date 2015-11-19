///////////////////////////////////////////////////////////////////////////////
// AMP REDUCE
//////////////////////////////////////////////////////////////////////////////

#if !defined( KOKKOS_KALMAR_AMP_REDUCE_INL )
#define KOKKOS_KALMAR_AMP_REDUCE_INL

#include <iostream>

#include <algorithm>
#include <numeric>
#include <cmath>
#include <type_traits>
#include <Kalmar/Kokkos_Kalmar_Tile.hpp>
#include <Kalmar/Kokkos_Kalmar_Invoke.hpp>
#include <Kalmar/Kokkos_Kalmar_Join.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Kokkos {
namespace Impl {


template<class T>
T* reduce_value(T* x, std::true_type) restrict(amp)
{
  return x;
}

template<class T>
T& reduce_value(T* x, std::false_type) restrict(amp)
{
  return *x;
}

template< class Tag, class F, class TransformIndex, class T >
void reduce_enqueue(
  const int szElements,
  const F & f,
  TransformIndex transform_index,
  T * const output_result,
  int const output_length)
{
  using namespace hc ;

  typedef Kokkos::Impl::FunctorValueTraits< F , Tag > ValueTraits ;
  typedef Kokkos::Impl::FunctorValueInit< F , Tag >   ValueInit ;
  typedef Kokkos::Impl::FunctorValueJoin< F , Tag >   ValueJoin ;
  typedef Kokkos::Impl::FunctorFinal< F , Tag >       ValueFinal ;

  typedef typename ValueTraits::pointer_type   pointer_type ;
  typedef typename ValueTraits::reference_type reference_type ;

  if (output_length < 1) return;

  assert(output_result != nullptr);
  const auto tile_size = get_tile_size<T>(output_length);
  const std::size_t tile_len = std::ceil(1.0 * szElements / tile_size);
  std::vector<T> result(tile_len*output_length);
  auto fut = tile_for<T[]>(tile_size * tile_len, output_length, [&](hc::tiled_index<1> t_idx, tile_buffer<T[]> buffer) restrict(amp) 
  {
      const auto local = t_idx.local[0];
      const auto global = t_idx.global[0];
      const auto tile = t_idx.tile[0];

      buffer.action_at(local, [&](T* state)
      {
          ValueInit::init(f, state);
          if (global < szElements)
          {
              kalmar_invoke<Tag>(f, transform_index(t_idx, tile_size, tile_len), reduce_value(state, std::is_pointer<reference_type>()));
          }
      });
      t_idx.barrier.wait();

      // Reduce within a tile using multiple threads.
      for(std::size_t s = 1; s < buffer.size(); s *= 2)
      {
          const std::size_t index = 2 * s * local;
          if (index < buffer.size())
          {
              buffer.action_at(index, index + s, [&](T* x, T* y)
              {
                  ValueJoin::join(f, x, y);
              });
          }
          t_idx.barrier.wait();
      }

      // Store the tile result in the global memory.
      if (local == 0)
      {
          // Workaround for assigning from LDS memory: std::copy should work
          // directly
          buffer.action_at(0, [&](T* x)
          {
              // Workaround: copy_if used to avoid memmove
              std::copy_if(x, x+output_length, result.data()+tile*output_length, std::bind([]{ return true; }));
          });
      }
      
  });
  ValueInit::init(f, output_result);
  fut.wait();
  for(int i=0;i<tile_len;i++)
    ValueJoin::join(f, output_result, result.data()+i*output_length);
  ValueFinal::final( f , output_result );
}

}} //end of namespace Kokkos::Impl

#endif /* #if !defined( KOKKOS_KALMAR_AMP_REDUCE_INL ) */

