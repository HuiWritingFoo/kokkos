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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#if ! defined(KOKKOS_HAVE_KALMAR)
#  error "It doesn't make sense to build this file unless the Kokkos::Kalmar device is enabled.  If you see this message, it probably means that there is an error in Kokkos' CMake build infrastructure."
#else

#include <Kokkos_Bitset.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Vector.hpp>

#include <TestBitset.hpp>
#include <TestUnorderedMap.hpp>
#include <TestStaticCrsGraph.hpp>
#include <TestVector.hpp>
#include <TestDualView.hpp>
#include <TestSegmentedView.hpp>
#include <TestComplex.hpp>

#include <iomanip>

namespace Test {

class kalmar : public ::testing::Test {
protected:
  static void SetUpTestCase () {
    std::cout << std::setprecision(5) << std::scientific;
    Kokkos::Kalmar::initialize ();
  }

  static void TearDownTestCase () {
    Kokkos::Kalmar::finalize ();
  }
};


TEST_F( kalmar , staticcrsgraph )
{
  TestStaticCrsGraph::run_test_graph< Kokkos::Kalmar >();
  TestStaticCrsGraph::run_test_graph2< Kokkos::Kalmar >();
}

TEST_F( kalmar, complex )
{
  testComplex<Kokkos::Kalmar> ();
}
#if 0
TEST_F( kalmar, bitset )
{
  test_bitset<Kokkos::Kalmar> ();
}
#endif

#define KALMAR_INSERT_TEST( name, num_nodes, num_inserts, num_duplicates, repeat, near ) \
  TEST_F( kalmar, UnorderedMap_insert_##name##_##num_nodes##_##num_inserts##_##num_duplicates##_##repeat##x) { \
    for (int i=0; i<repeat; ++i)                                        \
      test_insert<Kokkos::Kalmar> (num_nodes, num_inserts, num_duplicates, near); \
  }

#define KALMAR_FAILED_INSERT_TEST( num_nodes, repeat )                  \
  TEST_F( kalmar, UnorderedMap_failed_insert_##num_nodes##_##repeat##x) { \
    for (int i=0; i<repeat; ++i)                                        \
      test_failed_insert<Kokkos::Kalmar> (num_nodes);                   \
  }

#define KALMAR_ASSIGNEMENT_TEST( num_nodes, repeat )                    \
  TEST_F( kalmar, UnorderedMap_assignment_operators_##num_nodes##_##repeat##x) { \
    for (int i=0; i<repeat; ++i)                                        \
      test_assignement_operators<Kokkos::Kalmar> (num_nodes);           \
  }

#define KALMAR_DEEP_COPY( num_nodes, repeat )                           \
  TEST_F( kalmar, UnorderedMap_deep_copy##num_nodes##_##repeat##x) {    \
    for (int i=0; i<repeat; ++i)                                        \
      test_deep_copy<Kokkos::Kalmar> (num_nodes);                       \
  }

#define KALMAR_VECTOR_COMBINE_TEST( size )             \
  TEST_F( kalmar, vector_combination##size##x) {                        \
    test_vector_combinations<int,Kokkos::Kalmar>(size);                 \
  }

#define KALMAR_DUALVIEW_COMBINE_TEST( size )             \
  TEST_F( kalmar, dualview_combination##size##x) {                      \
    test_dualview_combinations<int,Kokkos::Kalmar>(size);               \
  }

#define KALMAR_SEGMENTEDVIEW_TEST( size )                               \
  TEST_F( kalmar, segmentedview_##size##x) {                            \
    test_segmented_view<double,Kokkos::Kalmar>(size);                   \
  }
#if 0
KALMAR_INSERT_TEST(close, 100000, 90000, 100, 500, true)
KALMAR_INSERT_TEST(far, 100000, 90000, 100, 500, false)
KALMAR_FAILED_INSERT_TEST( 10000, 1000 )
KALMAR_DEEP_COPY( 10000, 1 )
#endif

KALMAR_VECTOR_COMBINE_TEST( 10 )
KALMAR_VECTOR_COMBINE_TEST( 3057 )
KALMAR_DUALVIEW_COMBINE_TEST( 10 )
#if 0
KALMAR_SEGMENTEDVIEW_TEST( 10000 )
#endif

#undef KALMAR_INSERT_TEST
#undef KALMAR_FAILED_INSERT_TEST
#undef KALMAR_ASSIGNEMENT_TEST
#undef KALMAR_DEEP_COPY
#undef KALMAR_VECTOR_COMBINE_TEST
#undef KALMAR_DUALVIEW_COMBINE_TEST
#undef KALMAR_SEGMENTEDVIEW_TEST

} // namespace test

#endif // KOKKOS_HAVE_KALMAR


