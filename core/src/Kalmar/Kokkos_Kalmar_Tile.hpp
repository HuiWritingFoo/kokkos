
#include <hc.hpp>
#include <type_traits>
#include <vector>
#include <memory>

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
#define KOKKOS_KALMAR_TILE_RESTRIC_CPU restrict(cpu, amp)
#else
#define KOKKOS_KALMAR_TILE_RESTRIC_CPU restrict(cpu)
#endif

inline std::size_t get_max_tile_size() KOKKOS_KALMAR_TILE_RESTRIC_CPU
{
    return hc::accelerator().get_max_tile_static_size() - 1024;
}

inline std::size_t get_max_tile_thread() KOKKOS_KALMAR_TILE_RESTRIC_CPU
{
    return 256;
}

inline int next_pow_2(int x) restrict(cpu, amp)
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
inline std::size_t get_tile_size(std::size_t n = 1) KOKKOS_KALMAR_TILE_RESTRIC_CPU
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

    array_view(T* x, std::size_t n) restrict(amp, cpu) 
    : x(x), n(n)
    {}

    array_view(T* x, T* y) restrict(amp, cpu) 
    : x(x), n(y-x)
    {}

    T& operator[](std::size_t i) const restrict(amp, cpu)
    {
        return x[i];
    }

    std::size_t size() const restrict(amp, cpu)
    {
        return this->n;
    }

    T* data() const restrict(amp, cpu)
    {
        return x;
    }

    T* begin() const restrict(amp, cpu)
    {
        return x;
    }

    T* end() const restrict(amp, cpu)
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
Char* kalmar_byte_cast(T& x) restrict(cpu, amp)
{
    return reinterpret_cast<Char*>(&x);
}

template<class T, class U>
void kalmar_assign_impl(T& x, const U& y, std::true_type) restrict(cpu, amp)
{
    auto * src = kalmar_byte_cast(y);
    auto * dest = kalmar_byte_cast(x);
    std::copy(src, src+sizeof(T), dest);
}

template<class T, class U>
void kalmar_assign_impl(T& x, const U& y, std::false_type) restrict(cpu, amp)
{
    x = y;
}

// Workaround for assigning in and out of LDS memory
template<class T, class U>
void kalmar_assign(T& x, const U& y) restrict(cpu, amp)
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
void lds_for(__attribute__((address_space(3))) T& value, Body b) restrict(amp)
{
    T state = value;
    b(state);
    value = state;
}


template<class T, class Body>
void lds_for(T& value, Body b) restrict(amp)
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
    void action_at(std::size_t i, Action a) restrict(amp)
    {
        auto& value = static_cast<Derived&>(*this)[i];
        T state = value;
        a(state);
        value = state;
    }

    template<class Action, KOKKOS_KALMAR_REQUIRES(!use_tile_memory<T>())>
    void action_at(std::size_t i, Action a) restrict(amp)
    {
        a(static_cast<Derived&>(*this)[i]);
    }

    template<class Action>
    void action_at(std::size_t i, std::size_t j, Action a) restrict(amp)
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

    tile_buffer(element_type* x, std::size_t n, std::size_t) restrict(amp, cpu) 
    : base(x, n)
    {}

    tile_buffer(T* x, T* y, std::size_t) restrict(amp, cpu) 
    : base(x, y)
    {}
};

template<class T>
struct tile_buffer<T[]>
{
    typedef typename tile_type<T>::type element_type;
    element_type* x;
    std::size_t n, m;

    tile_buffer(element_type* x, std::size_t n, std::size_t m) restrict(amp, cpu) 
    : x(x), n(n), m(m)
    {}

    tile_buffer(element_type* x, element_type* y, std::size_t m) restrict(amp, cpu) 
    : x(x), n(y-x), m(m)
    {}

    element_type* operator[](std::size_t i) const restrict(amp, cpu)
    {
        return x+i*m;
    }

    template<class Action, KOKKOS_KALMAR_REQUIRES(use_tile_memory<T>())>
    void action_at(std::size_t i, Action a) restrict(amp)
    {
        if (m > get_max_tile_array_size()) return;
        element_type* value = (*this)[i];
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
    }

    template<class Action, KOKKOS_KALMAR_REQUIRES(!use_tile_memory<T>())>
    void action_at(std::size_t i, Action a) restrict(amp)
    {
        a((*this)[i]);
    }

    template<class Action>
    void action_at(std::size_t i, std::size_t j, Action a) restrict(amp)
    {
        this->action_at(i, [&](T* x)
        {
            this->action_at(j, [&](T* y)
            {
                a(x, y);
            });
        });
    }

    std::size_t size() const restrict(amp, cpu)
    {
        return this->n;
    }

    element_type* data() const restrict(amp, cpu)
    {
        return x;
    }
};

// Zero initialize LDS memory
struct zero_init_f
{
    template<class T>
    void operator()(__attribute__((address_space(3))) T& x, std::size_t=1) const restrict(amp)
    {
        auto * start = reinterpret_cast<__attribute__((address_space(3))) char*>(&x);
        std::fill(start, start+sizeof(T), 0);
    }

    template<class T>
    void operator()(__attribute__((address_space(3))) T* x, std::size_t size) const restrict(amp)
    {
        std::for_each(x, x+size, *this);
    }
};

static constexpr zero_init_f zero_init = {};

template<class U, class F, class T=typename std::remove_extent<U>::type, KOKKOS_KALMAR_REQUIRES(use_tile_memory<T>())>
hc::completion_future tile_for_impl(std::size_t size, std::size_t array_size, F f) 
{
    assert(array_size <= get_max_tile_array_size() && "Exceed max array size");
    const auto tile_size = get_tile_size<T>(array_size);
    assert(((size % tile_size) == 0) && "Tile size must be divisible by extent");
    auto grid = hc::extent<1>(size).tile(tile_size);
    grid.set_dynamic_group_segment_size(tile_size * sizeof(T) * array_size);
    return parallel_for_each(grid, [=](hc::tiled_index<1> t_idx) restrict(amp) 
    {
        typedef __attribute__((address_space(3))) T group_t;
        group_t * buffer = (group_t *)hc::get_group_segment_addr(hc::get_static_group_segment_size());
        tile_buffer<U> tb(buffer, tile_size, array_size);
        zero_init(tb[t_idx.local[0]], array_size);
        f(t_idx, tb);
    });
}

template<class U, class F, class T=typename std::remove_extent<U>::type, KOKKOS_KALMAR_REQUIRES(!use_tile_memory<T>())>
hc::completion_future tile_for_impl(std::size_t size, std::size_t array_size, F f) 
{
    const auto tile_size = get_tile_size<T>(array_size);
    hc::extent<1> grid(size);
    auto buffer = std::make_shared<std::vector<T>>(size*array_size);
    auto * buffer_data = buffer->data();
    auto fut = parallel_for_each(grid.tile(tile_size), [f, tile_size, array_size, buffer_data](hc::tiled_index<1> t_idx) restrict(amp) 
    {
        tile_buffer<U> tb(buffer_data + t_idx.tile[0]*tile_size*array_size, tile_size, array_size);
        f(t_idx, tb);
    });
    // Workaround: extra thread here will prevent memory corruption
    std::thread([buffer, fut]
    {
        fut.wait();
        (void)buffer->size();
    }).detach();

    return fut;
}

template<class T, class F>
hc::completion_future tile_for(std::size_t size, std::size_t array_size, F f)
{
    static_assert(std::rank<T>() > 0, "Array size only applies to array buffer");
    return tile_for_impl<T>(size, array_size, f);
}

template<class T, class F>
hc::completion_future tile_for(std::size_t size, F f) 
{
    return tile_for_impl<T>(size, 1, f);
}

}}

#endif
