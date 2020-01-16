#include <mpi.h>
#include <tuple>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include "gtest/gtest.h"

#include "libdistributed_work_queue.h"

using namespace distributed::queue;
using namespace std::literals::chrono_literals;
TEST(test_work_queue, single_no_stop) {
  using request = std::tuple<int>;
  using response = std::tuple<int, double>;
  std::vector<request> tasks;
  for (int i = 0; i < 5; ++i) {
    tasks.emplace_back(i);
  }
  std::vector<response> results;

  work_queue(
      MPI_COMM_WORLD,
      std::begin(tasks),
      std::end(tasks),
      [](request req) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto req_value = std::get<0>(req);
        return std::make_tuple(req_value, std::pow(req_value, 2));
      },
      [&](response res) {
        auto [i,d] = res;
        results.push_back(res);
      }
      );

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank == 0) {
    EXPECT_EQ(results.size(), 5);
  } else {
    EXPECT_EQ(results.size(), 0);
  }
}


TEST(test_work_queue, multi_no_stop) {
  using request = std::tuple<int>;
  using response = std::tuple<int, double>;
  std::vector<request> tasks;
  tasks.reserve(5);
  for (int i = 0; i < 5; ++i) {
    tasks.emplace_back(i);
  }
  std::vector<response> results;

  work_queue(
      MPI_COMM_WORLD,
      std::begin(tasks),
      std::end(tasks),
      [](request req) {
        std::vector<response> responses;
        responses.reserve(5);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto req_value = std::get<0>(req);
        for (int i = 0; i < 5; ++i) {
          responses.emplace_back(req_value, std::pow(req_value, 2));
        }
        return responses;
      },
      [&](response res) {
        auto [i,d] = res;
        results.push_back(res);
      }
      );

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank == 0) {
    EXPECT_EQ(results.size(), 25);
  } else {
    EXPECT_EQ(results.size(), 0);
  }
}
namespace std{
inline void PrintTo(const std::chrono::milliseconds& duration, ::std::ostream * os) {
  *os << duration.count() << "ms";
}
}

TEST(test_work_queue, single_withstop) {
  using request = std::tuple<int>;
  using response = std::tuple<int, double>;
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<request> tasks(size*2);
  for (int i = 0; i < 2*size; ++i) {
    tasks[i] = {i};
  }
  std::vector<response> results;

  auto start_time = std::chrono::high_resolution_clock::now();

  work_queue(
      MPI_COMM_WORLD,
      std::begin(tasks),
      std::end(tasks),
      [](request req, StopToken& token) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto req_value = std::get<0>(req);
        if(req_value != 0) {
          for (int i = 0; i < 3 && !token.stop_requested(); ++i) {
            std::this_thread::sleep_for(50ms);
          }
        } else {
          token.request_stop();
        }
        return std::make_tuple(req_value, std::pow(req_value, 2));
      },
      [&](response res) {
        auto [i,d] = res;
        results.push_back(res);
      }
      );

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::milliseconds const duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
  std::chrono::milliseconds const time_limit = 170ms;
  EXPECT_LE(duration, time_limit);
}
