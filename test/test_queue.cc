#include <mpi.h>
#include <tuple>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include "gtest/gtest.h"

#include "libdistributed_work_queue.h"
#include "libdistributed_work_queue_options.h"

using namespace distributed::queue;
using namespace std::literals::chrono_literals;
TEST(test_work_queue, single_no_stop) {
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
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto req_value = std::get<0>(req);
        return std::make_tuple(req_value, std::pow(req_value, 2));
      },
      [&](response res) {
        int i; double d;
        i = std::get<int>(res);
        i = std::get<double>(res);
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
      [&](std::vector<response>const& res) {
        for (auto const& element : res) {
          results.push_back(element);
        }
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
  std::vector<request> tasks(static_cast<size_t>(size)*2);
  for (int i = 0; i < 2*size; ++i) {
    tasks[i] = {i};
  }
  std::vector<response> results;

  auto start_time = std::chrono::high_resolution_clock::now();

  work_queue(
      MPI_COMM_WORLD,
      std::begin(tasks),
      std::end(tasks),
      [](request req, TaskManager<request, MPI_Comm>& token) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto req_value = std::get<0>(req);
        if(req_value != 0) {
          for (int i = 0; i < 3 && !token.stop_requested(); ++i) {
            std::cout << "worker " << rank  << " sleeps" << std::endl;
            std::this_thread::sleep_for(50ms);
          }
        } else {
          std::cout << "worker " << rank << " requests stop" << std::endl;
          token.request_stop();
        }
        return std::make_tuple(req_value, std::pow(req_value, 2));
      },
      [&](response res) {
        int i; double d;
        i = std::get<int>(res);
        d = std::get<double>(res);
        std::cout << "master recv " << i << " " << d << std::endl;
        results.push_back(res);
      }
      );

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::milliseconds const duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
  std::chrono::milliseconds const time_limit = 170ms;
  EXPECT_LE(duration, time_limit);
}

TEST(test_work_queue, single_dynamic_worker) {
  using request = std::tuple<int>;
  using response = std::tuple<int, double>;
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<request> tasks(size, 1);
  std::vector<response> results;

  int executions = 0;
  int total_executions = 0;

  work_queue(
      MPI_COMM_WORLD,
      std::begin(tasks),
      std::end(tasks),
      [&executions](request req, TaskManager<request, MPI_Comm>& token) {
        auto req_value = std::get<0>(req);
        ++executions;
        if(req_value == 1) {
          token.push(2);
        }
        return std::make_tuple(req_value, std::pow(req_value, 2));
      },
      [&](response res) {
        int i; double d;
        i = std::get<int>(res);
        i = std::get<double>(res);
        results.push_back(res);
      }
      );

  MPI_Allreduce(&executions, &total_executions, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_EQ(total_executions, size*2);
}

TEST(test_work_queue, single_dynamic_master) {
  using request = std::tuple<int>;
  using response = std::tuple<int, double>;
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<request> tasks(size, 1);
  std::vector<response> results;


  int executions = 0;
  int total_executions = 0;

  work_queue(
      MPI_COMM_WORLD,
      std::begin(tasks),
      std::end(tasks),
      [&executions](request req) {
        auto req_value = std::get<0>(req);
        ++executions;
        return std::make_tuple(req_value, std::pow(req_value, 2));
      },
      [&](response res, TaskManager<request, MPI_Comm>& token) {
        int i; double d;
        i = std::get<int>(res);
        i = std::get<double>(res);
        if(i == 1) {
          token.push(2);
        }
        results.push_back(res);
      }
      );

  MPI_Allreduce(&executions, &total_executions, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_EQ(total_executions, size*2);
}

TEST(test_work_queue, fallback_1process) {
  using request = std::tuple<int>;
  using response = std::tuple<int, double>;
  int size;
  MPI_Comm_size(MPI_COMM_SELF, &size);
  std::vector<request> tasks(size, 1);
  std::vector<response> results;


  int executions = 0;
  int total_executions = 0;

  work_queue(
      MPI_COMM_SELF,
      std::begin(tasks),
      std::end(tasks),
      [&executions](request req) {
        auto req_value = std::get<0>(req);
        ++executions;
        return std::make_tuple(req_value, std::pow(req_value, 2));
      },
      [&](response res, TaskManager<request, MPI_Comm>& token) {
        int i; double d;
        i = std::get<int>(res);
        i = std::get<double>(res);
        if(i == 1) {
          token.push(2);
        }
        results.push_back(res);
      }
      );

  MPI_Allreduce(&executions, &total_executions, 1, MPI_INT, MPI_SUM, MPI_COMM_SELF);
  EXPECT_EQ(total_executions, 2);
}

TEST(test_work_queue, worker_masters_groups) {
  using request = std::tuple<int>;
  using response = std::tuple<int, double>;

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(size < 4 || size % 2 == 1) GTEST_SKIP() << "this test needs an even number of processes at least 4";

  std::vector<request> tasks(size, 1);
  std::vector<response> results;
  int executions = 0;
  int total_executions = 0;

  work_queue_options<request> options;
  options.set_groups(
      [&]{
        std::vector<size_t> new_groups(size);
        for (size_t i = 0; i < size; ++i) {
          new_groups[i] = i / 2;
        }
        return new_groups;
      }()
  );


  work_queue(
      options,
      std::begin(tasks), std::end(tasks),
      [&executions](request req, TaskManager<request, MPI_Comm>& token) {
        auto req_value = std::get<0>(req);
        MPI_Comm* subcomm = token.get_subcommunicator();
        int subcomm_size = 0, subcomm_rank = 0;
        MPI_Comm_size(*subcomm, &subcomm_size);
        MPI_Comm_rank(*subcomm, &subcomm_rank);
        EXPECT_EQ(subcomm_size, 2);
        if(subcomm_rank == 0) {
          ++executions;
        }
        return std::make_tuple(req_value, std::pow(req_value, 2));
      },
      [&](response res, TaskManager<request, MPI_Comm>& token) {
        int i; double d;
        i = std::get<int>(res);
        i = std::get<double>(res);
        MPI_Comm* subcomm = token.get_subcommunicator();
        int subcomm_size = 0;
        MPI_Comm_size(*subcomm, &subcomm_size);
        EXPECT_EQ(subcomm_size, 2);

        if(i == 1) {
          token.push(2);
        }
        results.push_back(res);
      }
      );
  MPI_Allreduce(&executions, &total_executions, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_EQ(total_executions, size*3);
}


TEST(test_work_queue, worker_masters_groups_with_cancelation) {
  using request = std::tuple<int>;
  using response = std::tuple<int, double>;

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(size < 4 || size % 2 == 1) GTEST_SKIP() << "this test needs an even number of processes at least 4";

  std::vector<request> tasks(size, 1);
  std::vector<response> results;
  int executions = 0;
  int total_executions = 0;

  work_queue_options<request> options;
  options.set_groups(
      [&]{
        std::vector<size_t> new_groups(size);
        for (size_t i = 0; i < size; ++i) {
          new_groups[i] = i / 2;
        }
        return new_groups;
      }()
  );


  work_queue(
      options,
      std::begin(tasks), std::end(tasks),
      [&executions](request req, TaskManager<request, MPI_Comm>& token) {
        auto req_value = std::get<0>(req);
        MPI_Comm* subcomm = token.get_subcommunicator();
        int subcomm_size = 0, subcomm_rank = 0;
        MPI_Comm_size(*subcomm, &subcomm_size);
        MPI_Comm_rank(*subcomm, &subcomm_rank);
        EXPECT_EQ(subcomm_size, 2);
        if(subcomm_rank == 0) {
          ++executions;
        }
        return std::make_tuple(req_value, std::pow(req_value, 2));
      },
      [&](response res, TaskManager<request, MPI_Comm>& token) {
        int i; double d;
        i = std::get<int>(res);
        i = std::get<double>(res);
        MPI_Comm* subcomm = token.get_subcommunicator();
        int subcomm_size = 0, subcomm_rank = 0;
        MPI_Comm_size(*subcomm, &subcomm_size);
        MPI_Comm_size(*subcomm, &subcomm_rank);
        EXPECT_EQ(subcomm_size, 2);

        if(i == 1) {
          token.push(2);
          if(results.size() > size/2 && subcomm_rank == 1) {
            token.request_stop();
          }
        }
        results.push_back(res);
      }
      );
  MPI_Allreduce(&executions, &total_executions, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_LE(total_executions, size*3);
}
