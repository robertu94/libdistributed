#ifndef LIBDISTRIBUTED_WORK_QUEUE_OPTIONS_H
#define LIBDISTRIBUTED_WORK_QUEUE_OPTIONS_H



#include <mpi.h>
#include <vector>
#include <numeric>

/**
 *  \file
 *  \brief customization points for work_queue
 */

namespace distributed {
namespace queue {
template<class Task>
class work_queue_options {
  public:
    /**
     * Construct a work queue options for a MPI_COMM_WORLD
     *
     */
    work_queue_options()
  {
    MPI_Comm_dup(MPI_COMM_WORLD, &queue_comm);
  }
    /**
     * Construct a work queue options for a given communicator
     *
     * \param[in] queue_comm the communicator to use for the queue
     */ 
    work_queue_options(MPI_Comm comm)
  {
    MPI_Comm_dup(comm, &queue_comm);
  }

    ~work_queue_options() {
      if(queue_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&queue_comm);
      }
    }

    work_queue_options(work_queue_options&)=delete;
    work_queue_options& operator=(work_queue_options&)=delete;
    work_queue_options(work_queue_options&& rhs) noexcept:
      queue_comm(rhs.queue_comm)
    {
      rhs.queue_comm = MPI_COMM_NULL;
    }
    work_queue_options& operator=(work_queue_options&& rhs) noexcept {
      queue_comm = rhs.queue_comm;
      rhs.queue_comm = MPI_COMM_NULL;
    }

    /**
     * \returns the rank of the current process on queue_comm
     */
    int get_queue_rank() const {
      int rank;
      MPI_Comm_rank(queue_comm, &rank);
      return rank;
    }

    /**
     * \returns the rank of the current process on queue_comm
     */
    int get_queue_size() const {
      int size;
      MPI_Comm_size(queue_comm, &size);
      return size;
    }

    bool is_master() const {
      const auto groups = get_groups();
      const auto rank = get_queue_rank();
      return rank == get_root() || groups[rank] == groups[get_root()];
        
    }

    /**
     * \returns the rank in queue_comm that corresponds to the lead master process
     */
    size_t get_root() const {
      return root;
    }
    /**
     * \param[in] comm the communicator that will be used by the work queue
     * \returns a vector of length MPI_Comm_size that contains non-negative
     *   values indicating how the ranks should be grouped for use in the work
     *   queue the root value must be present at least once.
     *
     */
    std::vector<size_t> get_groups() const {
      int size, rank;
      MPI_Comm_size(queue_comm, &size);
      MPI_Comm_rank(queue_comm, &rank);
      if (size != groups.size()) {
        std::vector<size_t> g(size);
        std::iota(std::begin(g), std::end(g), 0);
        return g;
      } else {
        return groups;
      }
    }

    /**
     * \returns a native handle to the underlying communicator
     */
    MPI_Comm get_native_queue_comm() const {
      return queue_comm;
    }

    /**
     * \param[in] new_groups sets the groups to new_groups
     */
    void set_groups(std::vector<size_t>const& new_groups) {
      groups = new_groups;
    }
    /**
     * \param[in] new_root sets the root to new_root
     */
    void set_root(size_t new_root) {
      root = new_root;
    }


  private:
    MPI_Comm queue_comm;
    std::vector<size_t> groups{0};
    size_t root = 0;
};


}
}

#endif /* end of include guard: LIBDISTRIBUTED_WORK_QUEUE_OPTIONS_H_ZCPVMQGB */
