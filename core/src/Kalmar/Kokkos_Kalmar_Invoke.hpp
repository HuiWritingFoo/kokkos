
#include <type_traits>

#if !defined( KOKKOS_KALMAR_INVOKE_H )
#define KOKKOS_KALMAR_INVOKE_H

namespace Kokkos {
namespace Impl {


template<class Tag, class F, class... Ts, typename std::enable_if<(!std::is_void<Tag>()), int>::type = 0>
inline void kalmar_invoke(F&& f, Ts&&... xs) restrict(amp, cpu)
{
  f(Tag(), static_cast<Ts&&>(xs)...);
}

template<class Tag, class F, class... Ts, typename std::enable_if<(std::is_void<Tag>()), int>::type = 0>
inline void kalmar_invoke(F&& f, Ts&&... xs) restrict(amp, cpu)
{
  f(static_cast<Ts&&>(xs)...);
}

}}

#endif
