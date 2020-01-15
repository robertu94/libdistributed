#include <iostream>
#include <string>
#include "mpi.h"
#include <unistd.h>
#include "gtest/gtest.h"

#include <execinfo.h>

void failing_error_handler(MPI_Comm* comm, int* ec, ...) {
  
  int rank, size;
  MPI_Comm_size(*comm, &size);
  MPI_Comm_rank(*comm, &rank);
  int nptrs;
  void* backtrace_buffer[100];
  char** backtrace_strings;
  nptrs = backtrace(backtrace_buffer, 100);
  backtrace_strings = backtrace_symbols(backtrace_buffer, nptrs);

  for (int i = 0; i < nptrs; ++i) {
    printf("BT %d/%d: [%d] %s\n", rank, size, i, backtrace_strings[i]);
  }
  free(backtrace_strings);


  int length;
  std::string s(MPI_MAX_ERROR_STRING, '\0');
  MPI_Error_string(*ec, &s[0], &length);
  ADD_FAILURE() << s.c_str();
}

int main(int argc, char *argv[])
{
  int rank, size, disable_printers=1;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::testing::InitGoogleTest(&argc, argv);

  if(rank == 0){
    int opt;
    while((opt = getopt(argc, argv, "p")) != -1) {
        switch(opt) {
          case 'p':
          disable_printers = 0;
          break;
        default:
          break;
        }
    }
  }
  MPI_Bcast(&disable_printers, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //disable printers for non-root process
  if(rank != 0 and disable_printers) {
    auto&& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
  }

  int result = RUN_ALL_TESTS();

  int all_result=0;
  MPI_Allreduce(&result, &all_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(rank == 0 && all_result) std::cerr << "one or more tests failed on another process, please check them" << std::endl;
  MPI_Finalize();

  return all_result;
}
