#include "libdistributed_comm.h"


namespace distributed {
namespace comm {
namespace serializer {

MPI_Datatype mpi_size_t() {
  return (std::is_same<uint8_t, size_t>::value) ? MPI_UINT8_T :
         (std::is_same<uint16_t, size_t>::value) ? MPI_UINT16_T :
         (std::is_same<uint32_t, size_t>::value) ? MPI_UINT32_T :
         (std::is_same<uint64_t, size_t>::value) ? MPI_UINT64_T :
         MPI_INT;
}
}
}
}
