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

#include <Kalmar/Kokkos_Kalmar_Invoke.hpp>
#include <Kalmar/Kokkos_Kalmar_Join.hpp>

#include "hc_am.hpp"
#include <Kalmar/Kokkos_Kalmar_Error.hpp>

namespace Kokkos {
namespace Impl {

template< class Tag, class F, class TransformIndex>
void scan_enqueue(
  const int len,
  const F & f,
  TransformIndex transform_index)
{
    typedef Kokkos::Impl::FunctorValueTraits< F, Tag>  ValueTraits;
    typedef Kokkos::Impl::FunctorValueInit<   F, Tag>  ValueInit;
    typedef Kokkos::Impl::FunctorValueJoin<   F, Tag>  ValueJoin;
    typedef Kokkos::Impl::FunctorValueOps<    F, Tag>  ValueOps;

    typedef typename ValueTraits::value_type    value_type;
    typedef typename ValueTraits::pointer_type    pointer_type;
    typedef typename ValueTraits::reference_type  reference_type;

    const auto tile_size = get_tile_size<value_type>();
    const std::size_t tile_len = std::ceil(1.0 * len / tile_size);
    std::vector<value_type> vecResult(tile_len);
    value_type* result = (value_type*)hc::am_alloc( sizeof(value_type)*tile_len, hc::accelerator(), 0 );
    if( tile_len )
      KALMAR_ASSERT( result );

    value_type* scratch = (value_type*)hc::am_alloc( sizeof(value_type)*len, hc::accelerator(), 0 );
    if( len )
      KALMAR_ASSERT( scratch );

    tile_for<value_type>(tile_len * tile_size, [=](hc::tiled_index<1> t_idx, tile_buffer<value_type> buffer) [[hc]]
    {
        const auto local = t_idx.local[0];
        const auto global = t_idx.global[0];
        const auto tile = t_idx.tile[0];

        // Join tile buffer elements
        const auto join = [&](std::size_t i, std::size_t j)
        {
            buffer.action_at(i, j, [&](value_type& x, const value_type& y)
            {
                ValueJoin::join(f, &x, &y);
            });
        };

        // Copy into tile
        buffer.action_at(local, [&](value_type& state)
        {
            ValueInit::init(f, &state);
            if (global < len) kalmar_invoke<Tag>(f, transform_index(t_idx, tile_size, tile_len), state, false);
        });
        t_idx.barrier.wait();
        // Up sweep phase
        for(std::size_t d=1;d<buffer.size();d*=2)
        {
            auto d2 = 2*d;
            auto i = local*d2;
            auto j = i + d - 1;
            auto k = i + d2 - 1;
            join(k, j);
            t_idx.barrier.wait();
        }

        result[tile] = buffer[buffer.size()-1];
        buffer[buffer.size()-1] = 0;
        // Down sweep phase
        for(std::size_t d=buffer.size()/2;d>0;d/=2)
        {
            auto d2 = 2*d;
            auto i = local*d2;
            auto j = i + d - 1;
            auto k = i + d2 - 1;
            auto t = buffer[k];
            join(k, j);
            buffer[j] = t;
            t_idx.barrier.wait();
        }
        // Copy tiles into global memory
        if (global < len) scratch[global] = buffer[local];

    }).wait();

    // Ugly codes
    // TODO: need HC-based prefix sum
    KALMAR_SAFE_CALL( hc::am_copy( (void*)vecResult.data(), (void*)result, vecResult.size() * sizeof(value_type) ) );
    // Compute prefix sum
    std::partial_sum(vecResult.begin(), vecResult.end(), vecResult.begin(), make_join_operator<ValueJoin>(f));
    KALMAR_SAFE_CALL( hc::am_copy( (void*)result, (void*)vecResult.data(), vecResult.size() * sizeof(value_type) ) );

    hc::parallel_for_each(hc::extent<1>(len).tile(tile_size), [=](hc::tiled_index<1> t_idx) [[hc]]
    {
        // const auto local = t_idx.local[0];
        const auto global = t_idx.global[0];
        const auto tile = t_idx.tile[0];

        if (global < len) 
        {
            auto final_state = scratch[global];
            if (tile != 0) ValueJoin::join(f, &final_state, &result[tile-1]);
            kalmar_invoke<Tag>(f, transform_index(t_idx, tile_size, tile_len), final_state, true);
        }
    }).wait();

    KALMAR_SAFE_CALL( hc::am_free( (void*)result ) );
    KALMAR_SAFE_CALL( hc::am_free( (void*)scratch ) );
}

} // namespace Impl
} // namespace Kokkos
