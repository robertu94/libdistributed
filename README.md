# LibDistributed

LibDistributed provides a collection of facilities for MPI that create for higher level facilities for programming in C++.

## Using LibDistributed

Here is a minimal example with error handling of how to use LibDistributed.

```cpp
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
```

## Getting Started

After skimming the example, LibDistributed has a few major types that you will need to use:

Type                     | Use 
-------------------------|----------------------------------------------------------------------
`work_queue.h`           | A distributed work queue with cancellation support
`types.h`                | Uses templates to create `MPI_Datatype`s

## Dependencies

+ `cmake` version `3.13` or later
+ either:
  + `gcc-8.3.0` or later
  + `clang-9.0.0` or later
+ An MPI implementation supporting MPI-3 or later.  Tested on OpenMPI 4.0.2


## Building and Installing LibDistributed

LibDistributed uses CMake to configure build options.  See CMake documentation to see how to configure options

+ `CMAKE_INSTALL_PREFIX` - install the library to a local directory prefix
+ `BUILD_DOCS` - build the project documentation
+ `BUILD_TESTING` - build the test cases

```bash
BUILD_DIR=build
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake ..
make
make test
make install
```

To build the documentation:


```bash
BUILD_DIR=build
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake .. -DBUILD_DOCS=ON
make docs
# the html docs can be found in $BUILD_DIR/html/index.html
# the man pages can be found in $BUILD_DIR/man/
```


## Stability

As of version 1.0.0, LibDistributed will follow the following API stability guidelines:

+ The functions defined in files in `./include` are to considered stable
+ The functions defined in files in ending in `_impl.h` considered unstable

Stable means:

+ New APIs may be introduced with the increase of the minor version number.
+ APIs may gain additional overloads for C++ compatible interfaces with an increase in the minor version number.
+ An API may change the number or type of parameters with an increase in the major version number.
+ An API may be removed with the change of the major version number

Unstable means:

+ The API may change for any reason with the increase of the minor version number

Additionally, the performance of functions, memory usage patterns may change for both stable and unstable code with the increase of the patch version.


## Bug Reports

Please files bugs to the Github Issues page on the robertu94 github repository.

Please read this post on [how to file a good bug report](https://codingnest.com/how-to-file-a-good-bug-report/).  After reading this post, please provide the following information specific to LibDistributed:

+ Your OS version and distribution information, usually this can be found in `/etc/os-release`
+ the output of `cmake -L $BUILD_DIR`
+ the version of each of LibDistributed's dependencies listed in the README that you have installed. Where possible, please provide the commit hashes.

