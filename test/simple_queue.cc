#include <iostream>
#include <tuple>
#include <chrono>
#include <thread>
#include <cmath>

#include <mpi.h>
#include <work_queue.h>

using namespace std::literals::chrono_literals;
namespace queue = distributed::queue;

int main(int argc, char *argv[])
{
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  using request = std::tuple<int>;
  using response = std::tuple<int, double>;
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<request> tasks(size*2);
  for (int i = 0; i < 2*size; ++i) {
    tasks[i] = {i};
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  queue::work_queue(
    MPI_COMM_WORLD, std::begin(tasks), std::end(tasks),
    [](request req, queue::StopToken& token) {
      //code in this lambda expression gets run once for each task
      auto [i] = req;
      std::cout << "worker got i=" << i << std::endl;

      // if the request is request 0, request termination
      // otherwise sleep for 150ms in 50ms increments
      if (i != 0) {
        for (int j = 0; j < 3 && !token.stop_requested(); ++j) {
          std::this_thread::sleep_for(50ms);
        }
      } else {
        token.request_stop();
      }

      return std::make_tuple(i, std::pow(i, 2));
    },
    [&](response res) {
      //code in this lambda gets run once for each element returned
      //by the worker threads
      auto [i, d] = res;
      std::cout << "master got i=" << i << " d=" << d << std::endl;
    });

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::milliseconds const duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);

  if(rank == 0) {
    std::cout << "work took " << duration.count() << "ms" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
