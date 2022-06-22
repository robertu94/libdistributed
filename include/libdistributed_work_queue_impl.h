#include <iterator>
#include <queue>
#include <set>
#include <type_traits>
#include <algorithm>
#include <unordered_set>
#include <mpi.h>
#include <iostream>

#include "libdistributed_task_manager.h"
#include "libdistributed_work_queue_options.h"
#include "libdistributed_comm.h"
#include "libdistributed_version.h"

namespace distributed {
namespace queue {
namespace impl {
  
template <class Container>
size_t count_unique(Container const& c) {
  std::unordered_set<typename Container::value_type> seen;
  return std::count_if(
      std::begin(c),
      std::end(c),
      [&seen](typename Container::const_reference v) {
        return seen.insert(v).second;
      });
}

template <class T>
struct iterator_to_value_type {
  typedef typename std::iterator_traits<T>::value_type type;
};

template <class V>
struct iterator_to_value_type<std::back_insert_iterator<V>> {
  typedef typename std::back_insert_iterator<V>::container_type::value_type type;
};

template <typename T, typename = void>
struct is_iterable : std::false_type {};
template <typename T>
struct is_iterable<T, std::void_t<decltype(std::declval<T>().begin()),
                                  decltype(std::declval<T>().end())>>
    : std::true_type {};

enum class worker_status: int {
  done = 1,
  more = 2,
  cancel = 3,
  new_task = 4
};


template <class RequestType, class ResponseType>
class WorkerTaskManager : public TaskManager<RequestType, MPI_Comm>
{
public:
  WorkerTaskManager(MPI_Comm subcomm, work_queue_options<RequestType> const& options)
    : TaskManager<RequestType, MPI_Comm>()
    , queue_comm(options.get_native_queue_comm())
    , subcomm(subcomm)
    , ROOT(options.get_root())
    , num_workers_v(count_unique(options.get_groups()) - 1)
  {
    MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &flag);
    *flag = 0;
    MPI_Win_create(flag, sizeof(int), sizeof(int), MPI_INFO_NULL, queue_comm, &win);
  }

  ~WorkerTaskManager() {
    MPI_Barrier(queue_comm);
    MPI_Win_free(&win);
    MPI_Free_mem(flag);
  }

  bool stop_requested() override {
    int result = 0;
    int zero = 0;
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, /*target_rank*/ROOT, /*assert*/0, win);
    MPI_Fetch_and_op(&zero, &result, MPI_INT, /*target_rank*/ROOT, /*target_disp*/0, MPI_NO_OP, win);
    MPI_Win_unlock(ROOT, win);
#if LIBDISTRIBUTED_DEBUG_WORK_QUEUE_TERMINATION
    int rank;
    MPI_Comm_rank(queue_comm, &rank);
    std::cout << "impl worker " << rank <<  " checks stop " << std::boolalpha << static_cast<bool>(result) << std::endl;
#endif
    return result;
  }


  void request_stop() override {
#if LIBDISTRIBUTED_DEBUG_WORK_QUEUE_TERMINATION
    int rank;
    MPI_Comm_rank(queue_comm, &rank);
    std::cout << "impl worker " << rank <<  " issues stop " << std::endl;
#endif
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, /*target_rank*/ROOT, /*assert*/0, win);
    int one = 1, ignore=0;
    MPI_Fetch_and_op(&one, &ignore, MPI_INT, /*target_rank*/ROOT, /*target_disp*/0, MPI_REPLACE, win);
    MPI_Win_unlock(ROOT, win);
  }

  void push(RequestType const& request) override {
    ResponseType response;
    //let master know a new task is coming
    comm::send(response, 0, (int)worker_status::new_task, queue_comm);

    //send the new request to the master
    comm::send(request, 0, (int)worker_status::new_task, queue_comm);
  }

  MPI_Comm* get_subcommunicator() override {
    return &subcomm;
  }

  size_t num_workers() const override {
    return num_workers_v;
  }

  private:
  MPI_Comm queue_comm, subcomm;
  MPI_Win win;
  int* flag;
  const int ROOT;
  size_t num_workers_v;
};

template <class RequestType>
class MasterTaskManager : public TaskManager<RequestType, MPI_Comm>
{
public:
  template <class TaskIt>
  MasterTaskManager(MPI_Comm comm, MPI_Comm subcomm, TaskIt begin, TaskIt end, work_queue_options<RequestType> const& options, size_t num_workers_v)
    : TaskManager<RequestType, MPI_Comm>(),
      comm(comm),
      subcomm(subcomm),
      ROOT(options.get_root()),
      num_workers_v(count_unique(options.get_groups())-1)
  {
      MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &flag);
      *flag = 0;
      MPI_Win_create(flag, sizeof(int), sizeof(int), MPI_INFO_NULL, comm, &win);
       while(begin != end) {
        requests.emplace(*begin);
         ++begin;
       }
  }

  ~MasterTaskManager() {
    MPI_Barrier(comm);
    MPI_Win_free(&win);
    MPI_Free_mem(flag);
  }

  bool stop_requested() override {
    int result = 0;
    int zero = 0;
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, /*target_rank*/ROOT, /*assert*/0, win);
    MPI_Fetch_and_op(&zero, &result, MPI_INT, /*target_rank*/ROOT, /*target_disp*/0, MPI_NO_OP, win);
    MPI_Win_unlock(ROOT, win);
#if LIBDISTRIBUTED_DEBUG_WORK_QUEUE_TERMINATION
    int rank;
    MPI_Comm_rank(comm, &rank);
    std::cout << "impl master " << rank <<  " checks stop " << std::boolalpha << static_cast<bool>(result) << std::endl;
#endif
    return result;
  }

  void request_stop() override {
#if LIBDISTRIBUTED_DEBUG_WORK_QUEUE_TERMINATION
    int rank;
    MPI_Comm_rank(comm, &rank);
    std::cout << "impl worker " << rank <<  " issues stop " << std::endl;
#endif
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, /*target_rank*/ROOT, /*assert*/0, win);
    int one = 1, ignore=0;
    MPI_Fetch_and_op(&one, &ignore, MPI_INT, /*target_rank*/ROOT, /*target_disp*/0, MPI_REPLACE, win);
    MPI_Win_unlock(ROOT, win);
  }

  void push(RequestType const& request) override {
    requests.emplace(request);
  }

  void pop() {
    requests.pop();
  }

  RequestType const& front() const {
    return requests.front();
  }

  RequestType& front() {
    return requests.front();
  }

  bool empty() const {
    return requests.empty();
  }

  MPI_Comm* get_subcommunicator() override {
    return &subcomm;
  }

  size_t num_workers() const override{
    return num_workers_v;
  }

  void recv_tasks() {
    int num_has_tasks = 0;
    int stop_requested_flag = 0;
    int ignored = 0;
    MPI_Reduce(&ignored, &stop_requested_flag, 1, MPI_INT, MPI_MAX, 0, subcomm);
    if(stop_requested_flag) {
      request_stop();
    }
    MPI_Reduce(&ignored, &num_has_tasks, 1, MPI_INT, MPI_SUM, 0, subcomm);
    for (int i = 0; i < num_has_tasks; ++i) {
      std::vector<RequestType> new_requests;
      comm::recv(new_requests, MPI_ANY_SOURCE, MPI_ANY_TAG, subcomm, MPI_STATUS_IGNORE);
      for (auto const& task : new_requests) {
        requests.push(task);
      }
    }
  }

  private:
  MPI_Comm comm, subcomm;
  std::queue<RequestType> requests;
  const int ROOT;
  size_t num_workers_v;
  int* flag;
  MPI_Win win;
};

template <class RequestType, class ResponseType, class TaskForwardIt, class Function>
void master_main(MPI_Comm subcom, TaskForwardIt tasks_begin, TaskForwardIt tasks_end, Function master_fn, work_queue_options<RequestType> const& options)
{
  MPI_Comm comm = options.get_native_queue_comm();
    //create worker queue
    std::queue<int> workers;
    {
      auto const& groups = options.get_groups();
      std::set<int> group_ids;
      for (size_t i = 0; i < groups.size(); ++i) {
        //if inserted, and not a master process 
        if(group_ids.insert(groups[i]).second && (not(i == options.get_root() || groups[options.get_root()] == groups[i]))) {
          workers.push(static_cast<int>(i));
        }
      }
    }

    //create task queue

    MasterTaskManager<RequestType> task_manager(comm, subcom, tasks_begin, tasks_end, options, workers.size());

    int outstanding = 0;
    while((!task_manager.empty() and !task_manager.stop_requested()) or outstanding > 0) {

      while(!task_manager.empty() and !task_manager.stop_requested() and !workers.empty()) {
        
        int worker_id = workers.front();
        ++outstanding;
        workers.pop();

        RequestType request = std::move(task_manager.front());
        task_manager.pop();

        comm::send(request, worker_id, (int)worker_status::more, comm);
      }

      MPI_Status response_status;
      ResponseType response;

      comm::recv(response, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &response_status);
      switch(worker_status(response_status.MPI_TAG)) {
        case worker_status::more:
          MPI_Abort(comm, 3);
          break;
        case worker_status::done:
          {
          int not_done = false;
          comm::bcast(not_done, 0, subcom);
          comm::bcast(response, 0, subcom);
          maybe_stop_token(master_fn, std::move(response), task_manager);
          task_manager.recv_tasks();
          workers.push(response_status.MPI_SOURCE);
          outstanding--;
          }
          break;
        case worker_status::cancel:
#if LIBDISTRIBUTED_DEBUG_WORK_QUEUE_TERMINATION
          std::cout << "master recv stop from worker " << response_status.MPI_SOURCE << std::endl;
#endif
          task_manager.request_stop();
          break;
        case worker_status::new_task:
          RequestType request;
          comm::recv(request, response_status.MPI_SOURCE, (int)worker_status::new_task, comm);
          task_manager.push(request);
          break;
      }
    }
    
    if(not task_manager.stop_requested()) task_manager.request_stop();

    while(not workers.empty()) {
      int worker_id = workers.front();
      workers.pop();

      RequestType request;
      comm::send(request, worker_id, (int)worker_status::done, comm);
    }
    {
      int done = true;
      comm::bcast(done, 0, subcom);
    }

}

template <class RequestType, class ResponseType>
class MasterAuxTaskManager final : public TaskManager<RequestType, MPI_Comm>
{
  public:
  MasterAuxTaskManager(MPI_Comm subcomm, work_queue_options<RequestType> const& options):
    queue_comm(options.get_native_queue_comm()),
    subcomm(subcomm),
    request_done(0),
    ROOT(options.get_root()),
    num_workers_v(count_unique(options.get_groups()) -1)
  {
      MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &flag);
      *flag = 0;
      MPI_Win_create(flag, sizeof(int), sizeof(int), MPI_INFO_NULL, queue_comm, &win);
  }

  ~MasterAuxTaskManager() {
    MPI_Barrier(queue_comm);
    MPI_Win_free(&win);
    MPI_Free_mem(flag);
  }

  void request_stop() final {
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, /*target_rank*/ROOT, /*assert*/0, win);
    int one = 1, ignore=0;
    MPI_Fetch_and_op(&one, &ignore, MPI_INT, /*target_rank*/ROOT, /*target_disp*/0, MPI_REPLACE, win);
    MPI_Win_unlock(0, win);
  }

  bool stop_requested() final {
    int rank;
    MPI_Comm_rank(queue_comm, &rank);
    int result = 0;
    int zero = 0;
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, /*target_rank*/ROOT, /*assert*/0, win);
    MPI_Fetch_and_op(&zero, &result, MPI_INT, /*target_rank*/ROOT, /*target_disp*/0, MPI_NO_OP, win);
    MPI_Win_unlock(ROOT, win);
    return result;
  }

  void push(RequestType const& request) final {
    requests.push_back(request);
  }

  void send_tasks() {
    int has_tasks = (requests.empty()) ? 0: 1;
    int ignore=0;
    MPI_Reduce(&request_done, &ignore, 1, MPI_INT, MPI_MAX, 0, subcomm);
    MPI_Reduce(&has_tasks, &ignore, 1, MPI_INT, MPI_SUM, 0, subcomm);
    if(has_tasks) {
      comm::send(requests, 0, 0, subcomm);
    }
    requests.clear();
  }

  MPI_Comm* get_subcommunicator() final {
    return &subcomm;
  }

  size_t num_workers() const final {
    return num_workers_v;
  }

  private:
  MPI_Comm queue_comm, subcomm;
  MPI_Win win;
  std::vector<RequestType> requests;
  int* flag, done, request_done;
  const int ROOT;
  size_t num_workers_v;
};

template <class RequestType, class ResponseType, class Function>
void master_aux(MPI_Comm subcomm, Function master_fn, work_queue_options<RequestType> const& options) {
  MasterAuxTaskManager<RequestType, ResponseType> task_manager(subcomm, options);

  int master_done = false;
  while(!task_manager.stop_requested() && !master_done) {
    ResponseType response;
    comm::bcast(master_done, 0, subcomm);
    if(!master_done) {
      comm::bcast(response, 0, subcomm);
      maybe_stop_token(master_fn, std::move(response), task_manager);
      task_manager.send_tasks();
    }
  }
}

template <typename Function, class Message, class RequestType,
          typename = void>
struct takes_stop_token : std::false_type
{};

template <typename Function, class Message, class RequestType>
struct takes_stop_token<
  Function, Message, RequestType,
  std::void_t<decltype(std::declval<Function>()(
    std::move(std::declval<Message>()), std::declval<TaskManager<RequestType, MPI_Comm>&>()))>> : std::true_type
{};

template <class Function, class Message, class RequestType, class Enable = void>
struct maybe_stop_token_impl {
    static auto call(Function f, Message m, TaskManager<RequestType, MPI_Comm>&) {
      return f(m);
    }
};


template <class Function, class Message, class RequestType>
struct maybe_stop_token_impl<Function, Message, RequestType,
  typename std::enable_if_t<takes_stop_token<Function,Message, RequestType>::value>> {
    static auto call(Function f, Message m, TaskManager<RequestType, MPI_Comm>& s) {
      return f(m,s);
    }
};

template <class Function, class Message, class RequestType>
auto maybe_stop_token(Function f, Message&& m, TaskManager<RequestType, MPI_Comm>& s)
{
  return maybe_stop_token_impl<Function, Message, RequestType>::call(f, std::forward<Message>(m), s);
}

template <class RequestType, class ResponseType, class Function>
void worker_main(MPI_Comm subcomm, Function worker_fn, work_queue_options<RequestType> const& options)
{
  WorkerTaskManager<RequestType, ResponseType> stop_token(subcomm, options);
  MPI_Comm queue_comm = options.get_native_queue_comm();

  int worker_done = false;
  while(!worker_done) {
    MPI_Status task_status;
    RequestType request;
    comm::recv(request, options.get_root(), MPI_ANY_TAG, queue_comm, &task_status);

    switch(worker_status(task_status.MPI_TAG)) {
      case worker_status::done:
        worker_done = true;
        comm::bcast(worker_done, 0, subcomm);
        break;
      case worker_status::more:
        {
          comm::bcast(worker_done, 0, subcomm);
          comm::bcast(request, 0, subcomm);
          auto response = maybe_stop_token(worker_fn, std::move(request), stop_token);
          comm::send(response, options.get_root(), (int)worker_status::done, queue_comm);
        }
        break;
      case worker_status::cancel:
      case worker_status::new_task:
        MPI_Abort(queue_comm, 2);
        break;
    }
  }

}

template <class RequestType, class ResponseType, class Function>
void worker_aux(MPI_Comm subcomm, Function worker_fn, work_queue_options<RequestType> const& options) {
  WorkerTaskManager<RequestType, ResponseType> task_manager(subcomm, options);
  int worker_done = false;
  while(!task_manager.stop_requested() && !worker_done) {
    comm::bcast(worker_done, 0, subcomm);
    if(!worker_done) {
      RequestType request;
      comm::bcast(request, 0, subcomm);
      maybe_stop_token(worker_fn, std::move(request), task_manager);
    }
  }

}

template <class RequestType>
class NoWorkersTaskManager: public TaskManager<RequestType, MPI_Comm> {
  public:

  template<class TaskForwardIt>
  NoWorkersTaskManager(TaskForwardIt tasks_begin, TaskForwardIt tasks_end)
  {
    for(; tasks_begin != tasks_end; ++tasks_begin) {
      requests.emplace(*tasks_begin);
    }
  }

  bool stop_requested() override {
    return is_stop_requested;
  }

  void request_stop() override {
#if LIBDISTRIBUTED_DEBUG_WORK_QUEUE_TERMINATION
    std::cout << "no workers says stop" << std::endl;
#endif
    is_stop_requested = true;
  }

  void push(RequestType const& request) override {
    requests.push(request);
  }

  void pop() {
    requests.pop();
  }

  RequestType const& front() const {
    return requests.front();
  }

  RequestType& front() {
    return requests.front();
  }

  bool empty() const {
    return requests.empty();
  }

  MPI_Comm* get_subcommunicator() override {
    return &subcomm;
  }

  size_t num_workers() const override {
    return 1;
  }

  private:
  MPI_Comm subcomm = MPI_COMM_SELF;
  bool is_stop_requested = false;
  std::queue<RequestType> requests{};
};

template <class RequestType, class ResponseType, class TaskForwardIt, class WorkerFn, class MasterFn>
void no_workers(TaskForwardIt tasks_begin, TaskForwardIt tasks_end, MasterFn master_fn, WorkerFn worker_fn) {
  NoWorkersTaskManager<RequestType> task_manager(tasks_begin, tasks_end);

  while(!task_manager.empty() && !task_manager.stop_requested()) {
    RequestType task = std::move(task_manager.front());
    task_manager.pop();

    auto response = maybe_stop_token(worker_fn, std::move(task), task_manager);
    maybe_stop_token(master_fn, response, task_manager);
  }
}

}
}
}
