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

#include <hc.hpp>
#include <type_traits>
#include <vector>
#include <memory>
#include <Kalmar/Kokkos_Kalmar_Config.hpp>

#include "hc_am.hpp"
#include <Kalmar/Kokkos_Kalmar_Error.hpp>

#if !defined( KOKKOS_KALMAR_TILE_H )
#define KOKKOS_KALMAR_TILE_H

// Macro to abstract out the enable_if craziness
#define KOKKOS_KALMAR_REQUIRES(...) \
    bool KokkosKalmarRequiresBool ## __LINE__ = true, typename std::enable_if<KokkosKalmarRequiresBool ## __LINE__ && (__VA_ARGS__), int>::type = 0


namespace Kokkos {
namespace Impl {

template<class T>
using lds_t = __attribute__((address_space(3))) T;

template<class T>
struct use_tile_memory
: std::conditional<(sizeof(T) < 1024),
    std::is_trivially_default_constructible<T>,
    std::false_type
>::type
{};

#if 1
#define KOKKOS_KALMAR_TILE_RESTRICT __HC__ __CPU__
#else
#define KOKKOS_KALMAR_TILE_RESTRICT
#endif

inline std::size_t get_max_tile_size() KOKKOS_KALMAR_TILE_RESTRICT
{
    return hc::accelerator().get_max_tile_static_size() - 1024;
}

inline std::size_t get_max_tile_thread() KOKKOS_KALMAR_TILE_RESTRICT
{
    return 256;
}

inline int next_pow_2(int x) KOKKOS_KALMAR_TILE_RESTRICT
{ 
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

template<class T>
inline std::size_t get_tile_size(std::size_t n = 1) KOKKOS_KALMAR_TILE_RESTRICT
{
    const auto size = sizeof(T) * n;
    const auto group_size = get_max_tile_size();
    if (size == 0 || size > group_size) return 0;
    // Assume that thread size is a power of 2
    auto thread_size = get_max_tile_thread();
    while(size > (group_size / thread_size) && thread_size > 2) thread_size /= 2;
    return thread_size;
}

template<class T>
struct array_view
{
    T* x;
    std::size_t n;

    array_view(T* xp, std::size_t np) KOKKOS_KALMAR_TILE_RESTRICT
    : x(xp), n(np)
    {}

    array_view(T* xp, T* yp) KOKKOS_KALMAR_TILE_RESTRICT
    : x(xp), n(yp-xp)
    {}

    T& operator[](std::size_t i) KOKKOS_KALMAR_TILE_RESTRICT
    {
        return x[i];
    }

    std::size_t size() const KOKKOS_KALMAR_TILE_RESTRICT
    {
        return this->n;
    }

    T* data() const KOKKOS_KALMAR_TILE_RESTRICT
    {
        return x;
    }

    T* begin() const KOKKOS_KALMAR_TILE_RESTRICT
    {
        return x;
    }

    T* end() const KOKKOS_KALMAR_TILE_RESTRICT
    {
        return x+this->size();
    }
};

template<class T>
struct kalmar_char
{ using type=char; };

template<class T>
struct kalmar_char<const T>
: std::add_const<typename kalmar_char<T>::type>
{};

template<class T>
struct kalmar_char<__attribute__((address_space(3))) T>
{ using type = __attribute__((address_space(3))) typename kalmar_char<T>::type; };

template<class T>
struct kalmar_char<const __attribute__((address_space(3))) T>
{ using type = const __attribute__((address_space(3))) typename kalmar_char<T>::type; };

template<class T, class Char=typename kalmar_char<T>::type>
Char* kalmar_byte_cast(T& x) KOKKOS_KALMAR_TILE_RESTRICT
{
    return reinterpret_cast<Char*>(&x);
}

template<class T, class U>
void kalmar_assign_impl(T& x, const U& y, std::true_type) KOKKOS_KALMAR_TILE_RESTRICT
{
    auto * src = kalmar_byte_cast(y);
    auto * dest = kalmar_byte_cast(x);
    std::copy(src, src+sizeof(T), dest);
}

template<class T, class U>
void kalmar_assign_impl(T& x, const U& y, std::false_type) KOKKOS_KALMAR_TILE_RESTRICT
{
    x = y;
}

// Workaround for assigning in and out of LDS memory
template<class T, class U>
void kalmar_assign(T& x, const U& y) KOKKOS_KALMAR_TILE_RESTRICT
{
    kalmar_assign_impl(x, y, std::integral_constant<bool, (
        not std::is_assignable<T, U>() and
        std::is_trivially_copyable<T>() and 
        std::is_trivially_copyable<U>() and 
        sizeof(T) == sizeof(U)
    )>());
}

// Compute the address space of tile
template<class T>
struct tile_type
: std::conditional<(use_tile_memory<T>()),
    __attribute__((address_space(3))) T,
    T
>
{};

template<class T, class Body>
void lds_for(__attribute__((address_space(3))) T& value, Body b) [[hc]]
{
#if KOKKOS_KALMAR_HAS_WORKAROUNDS
    T state = value;
    b(state);
    value = state;
#else
    b(value);
#endif
}


template<class T, class Body>
void lds_for(T& value, Body b) [[hc]]
{
    b(value);
}

constexpr std::size_t get_max_tile_array_size()
{
    return 24;
}

template<class Derived, class T>
struct single_action
{
    template<class Action, KOKKOS_KALMAR_REQUIRES(use_tile_memory<T>())>
    void action_at(std::size_t i, Action a) [[hc]]
    {
        auto& value = static_cast<Derived&>(*this)[i];
#if KOKKOS_KALMAR_HAS_WORKAROUNDS
        T state = value;
        a(state);
        value = state;
#else
        a(value);
#endif
    }

    template<class Action, KOKKOS_KALMAR_REQUIRES(!use_tile_memory<T>())>
    void action_at(std::size_t i, Action a) [[hc]]
    {
        a(static_cast<Derived&>(*this)[i]);
    }

    template<class Action>
    void action_at(std::size_t i, std::size_t j, Action a) [[hc]]
    {
        static_cast<Derived&>(*this).action_at(i, [&](T& x)
        {
            static_cast<Derived&>(*this).action_at(j, [&](T& y)
            {
                a(x, y);
            });
        });
    }
};

template<class T>
struct tile_buffer
: array_view<typename tile_type<T>::type>, single_action<tile_buffer<T>, T>
{
    typedef typename tile_type<T>::type element_type;
    typedef array_view<element_type> base;

    using base::base;

    tile_buffer(element_type* xp, std::size_t np, std::size_t) KOKKOS_KALMAR_TILE_RESTRICT
    : base(xp, np)
    {}

    tile_buffer(T* xp, T* yp, std::size_t) KOKKOS_KALMAR_TILE_RESTRICT
    : base(xp, yp)
    {}
};

template<class T>
struct tile_buffer<T[]>
{
    typedef typename tile_type<T>::type element_type;
    element_type* element_data;
    std::size_t n, m;

    tile_buffer(element_type* xp, std::size_t np, std::size_t mp) KOKKOS_KALMAR_TILE_RESTRICT
    : element_data(xp), n(np), m(mp)
    {}

    tile_buffer(element_type* xp, element_type* yp, std::size_t mp) KOKKOS_KALMAR_TILE_RESTRICT
    : element_data(xp), n(yp-xp), m(mp)
    {}

    element_type* operator[](std::size_t i) const KOKKOS_KALMAR_TILE_RESTRICT
    {
        return element_data+i*m;
    }

    template<class Action, KOKKOS_KALMAR_REQUIRES(use_tile_memory<T>())>
    void action_at(std::size_t i, Action a) [[hc]]
    {
        element_type* value = (*this)[i];
#if KOKKOS_KALMAR_HAS_WORKAROUNDS
        if (m > get_max_tile_array_size()) return;
        T state[get_max_tile_array_size()];
        // std::copy(value, value+m, state);
        // Workaround for assigning from LDS memory
        std::transform(value, value+m, state, [](element_type& x)
        {
          T result;
          kalmar_assign(result, x);
          return result;
        });
        a(state);
        std::copy(state, state+m, value);
#else
        a(value);
#endif
    }

    template<class Action, KOKKOS_KALMAR_REQUIRES(!use_tile_memory<T>())>
    void action_at(std::size_t i, Action a) [[hc]]
    {
        a((*this)[i]);
    }

    template<class Action>
    void action_at(std::size_t i, std::size_t j, Action a) [[hc]]
    {
        this->action_at(i, [&](T* x)
        {
            this->action_at(j, [&](T* y)
            {
                a(x, y);
            });
        });
    }

    std::size_t size() const KOKKOS_KALMAR_TILE_RESTRICT
    {
        return this->n;
    }

    element_type* data() const KOKKOS_KALMAR_TILE_RESTRICT
    {
        return element_data;
    }
};

// Zero initialize LDS memory
struct zero_init_f
{
    template<class T>
    void operator()(__attribute__((address_space(3))) T& x, std::size_t=1) const [[hc]]
    {
        auto * start = reinterpret_cast<__attribute__((address_space(3))) char*>(&x);
        std::fill(start, start+sizeof(T), 0);
    }

    template<class T>
    void operator()(__attribute__((address_space(3))) T* x, std::size_t size) const [[hc]]
    {
        std::for_each(x, x+size, *this);
    }
};

static constexpr zero_init_f zero_init = {};

template<class U, class F, class T=typename std::remove_extent<U>::type, KOKKOS_KALMAR_REQUIRES(use_tile_memory<T>())>
hc::completion_future tile_for_impl(std::size_t size, std::size_t array_size, const F& f)
{
    assert(array_size <= get_max_tile_array_size() && "Exceed max array size");
    const auto tile_size = get_tile_size<T>(array_size);
    assert(((size % tile_size) == 0) && "Tile size must be divisible by extent");
    auto grid = hc::extent<1>(size).tile(tile_size);
    grid.set_dynamic_group_segment_size(tile_size * sizeof(T) * array_size);
    return parallel_for_each(grid, [=](hc::tiled_index<1> t_idx) [[hc]]
    {
        typedef __attribute__((address_space(3))) T group_t;
        group_t * buffer = (group_t *)hc::get_group_segment_addr(hc::get_static_group_segment_size());
        tile_buffer<U> tb(buffer, tile_size, array_size);
        zero_init(tb[t_idx.local[0]], array_size);
        f(t_idx, tb);
    });
}

template<class U, class F, class T=typename std::remove_extent<U>::type, KOKKOS_KALMAR_REQUIRES(!use_tile_memory<T>())>
hc::completion_future tile_for_impl(std::size_t size, std::size_t array_size, const F& f)
{
    const auto tile_size = get_tile_size<T>(array_size);
    hc::extent<1> grid(size);
    T * buffer_data = (T*)hc::am_alloc(size*array_size*sizeof(T), hc::accelerator(), 0);
    if( size*array_size )
      KALMAR_ASSERT( buffer_data );

    auto fut = parallel_for_each(grid.tile(tile_size), [f, tile_size, array_size, buffer_data](hc::tiled_index<1> t_idx) [[hc]]
    {
        tile_buffer<U> tb(buffer_data + t_idx.tile[0]*tile_size*array_size, tile_size, array_size);
        f(t_idx, tb);
    });
#if KOKKOS_KALMAR_HAS_WORKAROUNDS
    // Workaround: extra thread here will prevent memory corruption
    std::thread([buffer_data, fut]
    {
        fut.wait();
        KALMAR_SAFE_CALL( hc::am_free( (void*)buffer_data ) );
    }).detach();
#endif

    return fut;
}

template<class T, class F>
hc::completion_future tile_for(std::size_t size, std::size_t array_size, const F& f)
{
    static_assert(std::rank<T>() > 0, "Array size only applies to array buffer");
    return tile_for_impl<T>(size, array_size, f);
}

template<class T, class F>
hc::completion_future tile_for(std::size_t size, const F& f)
{
    return tile_for_impl<T>(size, 1, f);
}

}}

#endif
