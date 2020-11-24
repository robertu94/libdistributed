#include <mpi.h>
#include <tuple>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <std_compat/optional.h>
#include <std_compat/variant.h>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "libdistributed_comm.h"

namespace comm = distributed::comm;
using namespace std::literals::chrono_literals;

/**
 * ugly hack for pretty printers for standard library objects
 */
namespace std{
  template <class T>
  void PrintTo(const compat::optional<T>& optional, std::ostream* os) {
    if(optional)
    {
      *os << *optional;
    } else {
      *os << "{}";
    }
  }

  struct PrintToIfImpl {
    template <size_t N, class Tuple>
    void operator()(const Tuple& v, std::ostream* os, size_t index) {
      if(N == index) {
        *os << "{ " << index  << " " << compat::get<N>(v) << " }";
      }
    }
  };
  template <class... T, size_t... Is>
  void PrintToImpl(const compat::variant<T...>& v, std::ostream* os, size_t index,  std::index_sequence<Is...>) {
    comm::serializer::fold_comma(v, PrintToIfImpl{}, os, index);
  }
  template <class... T>
  void PrintTo(const compat::variant<T...>& v, std::ostream* os) {
    size_t index = compat::index(v);
    PrintToImpl(v, os, index, std::index_sequence_for<T...>{});
  }
}

class test_comm : public ::testing::Test {
  protected:
  void SetUp() override {
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(size < 2) GTEST_SKIP();
    MPI_Comm_split(MPI_COMM_WORLD, size < 2, rank, &bcast_comm);
  }
  void TearDown() override {
    distributed::comm::serializer::get_type_registry().clear();
    if(!(size < 2)) MPI_Comm_free(&bcast_comm);
  }

  int size, rank;
  MPI_Comm bcast_comm;
};

TEST_F(test_comm, primatives) {
  int i{};
  double d{};
  if(rank == 0) {
    i = 3;
    d = 2.5;
    comm::send(i, 1);
    comm::send(d, 1);
  } else if (rank == 1) {
    comm::recv(i, 0);
    comm::recv(d, 0);
  }
  if(rank < 2) {
    EXPECT_EQ(i, 3);
    EXPECT_EQ(d, 2.5);
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 0);
  }

}


TEST_F(test_comm, primatives_bcast) {
  int i;
  if(rank == 0) {
    i = 3;
  }
  comm::bcast(i, 0);
  EXPECT_EQ(i, 3);
}

TEST_F(test_comm, vector) {
  std::vector<int> v;
  std::vector<int> expected = {1, 2, 3, 4, 5};
  if(rank == 0) {
    v = expected;
    comm::send(v, 1);
  } else if (rank == 1) {
    comm::recv(v, 0);
  }
  if(rank < 2) {
    EXPECT_THAT(v, ::testing::ContainerEq(expected));
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 0);
  }
}

TEST_F(test_comm, vector_bcast) {
  std::vector<int> v;
  std::vector<int> expected = {1, 2, 3, 4, 5};
  if(rank == 0) {
    v = expected;
  } else if (rank == 1) {
  }
  comm::bcast(v, 0);
  if(rank < 2) {
    EXPECT_THAT(v, ::testing::ContainerEq(expected));
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 0);
  }
}


TEST_F(test_comm, vector_tuple) {
  std::vector<std::tuple<int,double>> v;
  std::vector<std::tuple<int,double>> expected = {{1,2.3}, {2,3.7}};
  if(rank == 0) {
    v = expected;
    comm::send(v, 1);
  } else if (rank == 1) {
    comm::recv(v, 0);
  }
  if(rank < 2) {
    EXPECT_THAT(v, ::testing::ContainerEq(expected));
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 1);
  }
}

TEST_F(test_comm, vector_tuple_bcast) {
  std::vector<std::tuple<int,double>> v;
  std::vector<std::tuple<int,double>> expected = {{1,2.3}, {2,3.7}};
  if(rank == 0) {
    v = expected;
  } else if (rank == 1) {
  }
  comm::bcast(v, 0);
  if(rank < 2) {
    EXPECT_THAT(v, ::testing::ContainerEq(expected));
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 1);
  }
}

TEST_F(test_comm, tuple) {
  std::tuple<int, double> v;
  std::tuple<int, double> expected = {1, 2.4};
  if(rank == 0) {
    v = expected;
    comm::send(v, 1);
  } else if (rank == 1) {
    comm::recv(v, 0);
  }
  if(rank < 2) {
    EXPECT_EQ(v, expected);
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 1);
  }
}

TEST_F(test_comm, tuple_bcast) {
  std::tuple<int, double> v;
  std::tuple<int, double> expected = {1, 2.4};
  if(rank == 0) {
    v = expected;
  }
  comm::bcast(v, 0);
  if(rank < 2) {
    EXPECT_EQ(v, expected);
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 1);
  }
}



TEST_F(test_comm, optional) {
  compat::optional<double> v;
  compat::optional<double> expected = 2.4;
  if(rank == 0) {
    v = expected;
    comm::send(v, 1);
  } else if (rank == 1) {
    comm::recv(v, 0);
  }
  if(rank < 2) {
    EXPECT_EQ(v, expected);
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 0);
  }
}

TEST_F(test_comm, optional_bcast) {
  compat::optional<double> v;
  compat::optional<double> expected = 2.3;
  if(rank == 0) {
    v = expected;
  }
  comm::bcast(v, 0);
  if(rank < 2) {
    EXPECT_EQ(v, expected);
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 0);
  }
}


TEST_F(test_comm, array) {
  std::array<double, 3> v;
  std::array<double, 3> expected = {2.4, 3.1, 2.7};
  if(rank == 0) {
    v = expected;
    comm::send(v, 1);
  } else if (rank == 1) {
    comm::recv(v, 0);
  }
  if(rank < 2) {
    EXPECT_EQ(v, expected);
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 1);
  }
}

TEST_F(test_comm, array_bcast) {
  std::array<double, 3> v;
  std::array<double, 3> expected = {2.4, 3.1, 2.7};
  if(rank == 0) {
    v = expected;
  }
  comm::bcast(v, 0);
  if(rank < 2) {
    EXPECT_EQ(v, expected);
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 1);
  }
}

TEST_F(test_comm, variant) {
  compat::variant<int, double> v;
  compat::variant<int, double> expected = 2.4;
  if(rank == 0) {
    v = expected;
    comm::send(v, 1);
  } else if (rank == 1) {
    comm::recv(v, 0);
  }
  if(rank < 2) {
    EXPECT_EQ(v, expected);
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 0);
  }
}

TEST_F(test_comm, variant_bcast) {
  compat::variant<int, double> v;
  compat::variant<int, double> expected = 2.4;
  if(rank == 0) {
    v = expected;
  }
  comm::bcast(v, 0);
  if(rank < 2) {
    EXPECT_EQ(v, expected);
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 0);
  }
}

TEST_F(test_comm, tuple_vector_int) {
  using dtype = std::tuple<std::vector<int>, double>;
  dtype v;
  dtype expected = {{1,2,3}, 4.7};
  if(rank == 0) {
    v = expected;
    comm::send(v, 1);
  } else if (rank == 1) {
    comm::recv(v, 0);
  }
  if(rank < 2) {
    EXPECT_EQ(v, expected);
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 0);
  }

}

TEST_F(test_comm, tuple_vector_int_bcast) {
  using dtype = std::tuple<std::vector<int>, double>;
  dtype v;
  dtype expected = {{1,2,3}, 4.7};
  if(rank == 0) {
    v = expected;
  }
  comm::bcast(v, 0);
  if(rank < 2) {
    EXPECT_EQ(v, expected);
    EXPECT_EQ(comm::serializer::get_type_registry().size(), 0);
  }
}

TEST_F(test_comm, test_size_t) {
  std::vector<size_t> v;
  std::vector<size_t> v_expected {1,2,3};
  if(rank == 0) {
    v = v_expected;
  }
  comm::bcast(v, 0);
  if(rank < 2) {
    EXPECT_EQ(v, v_expected);
  }
}
