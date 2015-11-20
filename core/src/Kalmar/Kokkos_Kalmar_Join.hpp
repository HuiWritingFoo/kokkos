
#if !defined( KOKKOS_KALMAR_JOIN_H )
#define KOKKOS_KALMAR_JOIN_H

namespace Kokkos {
namespace Impl {


// Adaptor to use ValueJoin with standard algorithms
template<class Joiner, class F>
struct join_operator
{
  const F* fp;
  template<class T, class U>
  T operator()(T x, const U& y) const
  {
    Joiner::join(*fp, &x, &y);
    return x;
  }
};

template<class Joiner, class F>
join_operator<Joiner, F> make_join_operator(const F& f)
{
  return join_operator<Joiner, F>{&f};
}

}}

#endif
