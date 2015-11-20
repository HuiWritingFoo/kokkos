
#include <Kalmar/Kokkos_Kalmar_Invoke.hpp>
#include <Kalmar/Kokkos_Kalmar_Join.hpp>

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
    std::vector<value_type> result(tile_len);
    std::vector<value_type> scratch(len);

    tile_for<value_type>(tile_len * tile_size, [&](hc::tiled_index<1> t_idx, tile_buffer<value_type> buffer) restrict(amp) 
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
        for(int d=1;d<buffer.size();d*=2)
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
        for(int d=buffer.size()/2;d>0;d/=2)
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

    // Compute prefix sum
    std::partial_sum(result.begin(), result.end(), result.begin(), make_join_operator<ValueJoin>(f));

    hc::parallel_for_each(hc::extent<1>(len).tile(tile_size), [&](hc::tiled_index<1> t_idx) restrict(amp) 
    {
        const auto local = t_idx.local[0];
        const auto global = t_idx.global[0];
        const auto tile = t_idx.tile[0];

        if (global < len) 
        {
            auto final_state = scratch[global];
            if (tile != 0) ValueJoin::join(f, &final_state, &result[tile-1]);
            kalmar_invoke<Tag>(f, transform_index(t_idx, tile_size, tile_len), final_state, true);
        }
    }).wait();
}

} // namespace Impl
} // namespace Kokkos
