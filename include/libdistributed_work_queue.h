#ifndef LIBDISTRIBUTED_WORK_QUEUE_H
#define LIBDISTRIBUTED_WORK_QUEUE_H
#include <mpi.h>
#include <utility>
#include "libdistributed_types.h"
#include "libdistributed_task_manager.h"
#include "libdistributed_work_queue_options.h"
#include "libdistributed_work_queue_impl.h"

/**
 * \file
 * \brief a distributed work queue that supports cancellation
 */

namespace distributed {
namespace queue {

  /**
   * type trait to determine task type from an iterator type
   *
   */
  template <class Type>
  struct iterator_to_request_type {
    /**
     * the contained within the iterator
     */
    using type = typename impl::iterator_to_value_type<Type>::type;
  };

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
template <class TaskRandomIt, class WorkerFunction, class MasterFunction>
void work_queue (
    work_queue_options<typename impl::iterator_to_value_type<TaskRandomIt>::type> const& options,
    TaskRandomIt tasks_begin,
    TaskRandomIt tasks_end,
    WorkerFunction worker_fn,
    MasterFunction master_fn
    ) {
  //setup communicator

  using RequestType = typename impl::iterator_to_value_type<TaskRandomIt>::type;
  using ResponseType = decltype( impl::maybe_stop_token( worker_fn,
        std::move(std::declval<RequestType>()),
        std::declval<TaskManager<RequestType, MPI_Comm>&>()
        )
      );
  

  if(options.get_queue_size() > 1) {
    //create sub-communicators
    const int rank = options.get_queue_rank();
    auto groups = options.get_groups();
    MPI_Comm subcomm;
    MPI_Comm_split(
        options.get_native_queue_comm(),
        groups[rank],
        rank,
        &subcomm
        );
    int subrank;
    MPI_Comm_rank(subcomm, &subrank);

    //determine the request and response types from the input
    if(options.is_master()) {
      if(subrank == 0) {
        impl::master_main<RequestType, ResponseType>(subcomm, tasks_begin, tasks_end, master_fn, options);
      } else {
        impl::master_aux<RequestType, ResponseType>(subcomm, master_fn, options);
      }
    } else {
      if(subrank == 0) {
        impl::worker_main<RequestType, ResponseType>(subcomm, worker_fn, options);
      } else {
        impl::worker_aux<RequestType, ResponseType>(subcomm, worker_fn, options);
      }
    }
    MPI_Comm_free(&subcomm);

  } else {
    impl::no_workers<RequestType, ResponseType>(tasks_begin, tasks_end, master_fn, worker_fn);
  }
  comm::serializer::get_type_registry().clear();
}

}
}
#endif
