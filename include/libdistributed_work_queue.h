#ifndef LIBDISTRIBUTED_WORK_QUEUE_H
#define LIBDISTRIBUTED_WORK_QUEUE_H
#include <mpi.h>
#include <utility>
#include "libdistributed_types.h"
#include "libdistributed_task_manager.h"
#include "libdistributed_work_queue_impl.h"

/**
 * \file
 * \brief a distributed work queue that supports cancellation
 */

namespace distributed {
namespace queue {

  /**
   * \param[in] comm the communicator to duplicate to use for communication
   * \param[in] tasks_begin an iterator to the beginning of the task list
   * \param[in] tasks_end an iterator to the end of the task list
   * \param[in] worker_fn function called by the worker ranks.  
   * \param[in] master_fn function called by the master ranks.
   *
   * `worker_fn` can have one of several possible signatures:
   *
   * 1. `ResponseType worker_fn(RequestType)`
   * 3. `ResponseType worker_fn(RequestType, TaskManager<RequestType>&)`
   *
   * The versions that take a `TaskManager<RequestType>&`, pass a subclass of
   * TaskManager which allows the caller to request that the remaining tasks in
   * the queue be canceled and that other tasks that are currently running be
   * cooperatively notified that they can stop.
   *
   *
   * `master_fn`  can have one of two possible signatures:
   *
   * 1. void master_fn(ResponseType)
   * 2. void master_fn(ResponseType, TaskManager<RequestType>&)
   *
   * The version that takes a `TaskManager<RequestType>&`, passes a subclass of
   * `TaskManager<RequestType>&` which allows the caller to request that the
   * remaining tasks in the queue be canceled and that other tasks that are
   * currently running be cooperatively notified that they can stop.
   *
   * \see distributed::queue::TaskManager<RequestType> for details on the semantics about cancellation
   */
template <class TaskForwardIt, class WorkerFunction, class MasterFunction>
void work_queue (
    MPI_Comm comm,
    TaskForwardIt tasks_begin,
    TaskForwardIt tasks_end,
    WorkerFunction worker_fn,
    MasterFunction master_fn
    ) {
  //setup communicator
  int rank, size;
  MPI_Comm queue_comm;
  MPI_Comm_dup(comm, &queue_comm);
  MPI_Comm_rank(queue_comm, &rank);
  MPI_Comm_size(queue_comm, &size);

  using RequestType = typename impl::iterator_to_value_type<TaskForwardIt>::type;
  using ResponseType = decltype( impl::maybe_stop_token( worker_fn,
        std::declval<RequestType>(),
        std::declval<TaskManager<RequestType>&>()
        )
      );

  if(size > 1) {
    //determine the request and response types from the input
    if(rank == 0) {
      impl::master<RequestType, ResponseType>(queue_comm, tasks_begin, tasks_end, master_fn);
    } else {
      impl::worker<RequestType, ResponseType>(queue_comm, worker_fn);
    }

  } else {
    impl::no_workers<RequestType, ResponseType>(tasks_begin, tasks_end, master_fn, worker_fn);
  }
  comm::serializer::get_type_registry().clear();
  MPI_Comm_free(&queue_comm);
}

}
}
#endif
