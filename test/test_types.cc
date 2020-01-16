#include <tuple>
#include <vector>
#include "gtest/gtest.h"
#include "libdistributed_types.h"

using namespace distributed::types;

TEST(type_to_dtype, array) {
  std::array<int,5> a;
  auto array_type = type_to_datatype<decltype(a)>::dtype();

  type_info array_info(array_type);
  EXPECT_EQ(array_info.get_combiner(), MPI_COMBINER_CONTIGUOUS);
  EXPECT_EQ(array_info.get_dtypes().size(), 1);
  EXPECT_EQ(array_info.get_dtypes()[0], MPI_INT);
  EXPECT_EQ(array_info.get_integers().size(), 1);
  EXPECT_EQ(array_info.get_integers()[0], 5);
}

TEST(type_to_dtype, tuple) {
  std::tuple<int,double> t;
  auto tuple_type = type_to_datatype<decltype(t)>::dtype();

  type_info tuple_info(tuple_type);
  EXPECT_EQ(tuple_info.get_combiner(), MPI_COMBINER_STRUCT);
  EXPECT_EQ(tuple_info.get_dtypes().size(), 2);
  EXPECT_EQ(tuple_info.get_dtypes()[0], MPI_INT);
  EXPECT_EQ(tuple_info.get_dtypes()[1], MPI_DOUBLE);
}

TEST(type_to_dtype, simple) {
  int num_integers, num_addresses, num_datatypes, combiner;
  auto int_type = type_to_datatype<int>::dtype();
  type_info int_info{int_type};
  EXPECT_EQ(int_info.get_combiner(), MPI_COMBINER_NAMED);
  EXPECT_EQ(int_type, MPI_INT) << "expected to be MPI_INT";

  auto double_type = type_to_datatype<double>::dtype();
  type_info double_info{double_type};
  EXPECT_EQ(double_info.get_combiner(), MPI_COMBINER_NAMED);
  EXPECT_EQ(double_type, MPI_DOUBLE) << "exected to be MPI_DOUBLE";
}
