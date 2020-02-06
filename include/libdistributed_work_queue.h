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
   * 2. `Container<ResponseType> worker_fn(RequestType)`
   * 3. `ResponseType worker_fn(RequestType, TaskManager<RequestType>&)`
   * 4. `Container<ResponseType> worker_fn(RequestType, TaskManager<RequestType>&)`
   *
   * The versions that take a `TaskManager<RequestType>&`, pass a subclass of
   * TaskManager which allows the caller to request that the remaining tasks in
   * the queue be canceled and that other tasks that are currently running be
   * cooperatively notified that they can stop.
   *
   * The versions that return a `Container<ResponseType>` -- a type that
   * conforms to the Container named requirement with element type ResponseType.
   * -- with call master_fn once for each element in the container
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
   * \see TaskManager<RequestType> for details on the semantics about cancellation
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
  int rank;
  MPI_Comm queue_comm;
  MPI_Comm_dup(comm, &queue_comm);
  MPI_Comm_rank(queue_comm, &rank);

  //determine the request and response types from the input
  using RequestType = typename impl::iterator_to_value_type<TaskForwardIt>::type;
  using ResponseType = typename impl::type_or_value_type<decltype(
      impl::maybe_stop_token(
        worker_fn,
        std::declval<RequestType>(),
        std::declval<TaskManager<RequestType>&>()
        )
      )>::type;

  //register types
  MPI_Datatype request_type = types::type_to_datatype<RequestType>::dtype();
  MPI_Datatype response_type = types::type_to_datatype<ResponseType>::dtype();

  if(rank == 0) {
    impl::master<RequestType, ResponseType>(queue_comm, request_type, response_type, tasks_begin, tasks_end, master_fn);
  } else {
    impl::worker<RequestType, ResponseType>(queue_comm, request_type, response_type, worker_fn);
  }

  MPI_Type_free(&request_type);
  MPI_Type_free(&response_type);
  MPI_Comm_free(&queue_comm);
}

}
}
#endif
