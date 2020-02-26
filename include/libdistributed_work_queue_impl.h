#include <iterator>
#include <queue>
#include <type_traits>
#include <algorithm>
#include <mpi.h>

#include "libdistributed_task_manager.h"
#include "libdistributed_comm.h"

namespace distributed {
namespace queue {
namespace impl {

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

constexpr int ROOT = 0;
enum class worker_status: int {
  done = 1,
  more = 2,
  cancel = 3,
  new_task = 4
};


template <class RequestType, class ResponseType>
class WorkerTaskManager : public TaskManager<RequestType>
{
public:
  WorkerTaskManager(MPI_Comm comm)
    : TaskManager<RequestType>()
    , comm(comm)
    , stop_request()
    , flag(0)
  {
  MPI_Ibcast(&done, 1, MPI_INT, ROOT, comm, &stop_request);
  }

  bool stop_requested() override {
    if(!flag) {
      MPI_Test(&stop_request, &flag, MPI_STATUS_IGNORE);
    }
    return flag;
  }

  void wait_stopped() {
    if(!flag) {
      MPI_Wait(&stop_request, MPI_STATUS_IGNORE);
      flag = true;
    }
  }

  void request_stop() override {
    int done = 1;
    comm::send(done, ROOT, (int)worker_status::cancel, comm);
  }

  void push(RequestType const& request) override {
    ResponseType response;
    MPI_Request mpi_request;
    //let master know a new task is coming
    comm::send(response, 0, (int)worker_status::new_task, comm);

    //send the new request to the master
    comm::send(request, 0, (int)worker_status::new_task, comm);
  }

  private:
  MPI_Comm comm;
  MPI_Request stop_request;
  int flag;
  int done;//used by MPI_Ibcast for syncronization, do not read unless flag==true
};

template <class RequestType>
class MasterTaskManager : public TaskManager<RequestType>
{
public:
  template <class TaskIt>
  MasterTaskManager(MPI_Comm comm, TaskIt begin, TaskIt end)
    : TaskManager<RequestType>(),
      comm(comm),
      is_stop_requested(0)
  {
       while(begin != end) {
        requests.emplace(*begin);
         ++begin;
       }
  }

  bool stop_requested() override {
    return is_stop_requested == 1;
  }

  void request_stop() override {
    MPI_Request request;
    is_stop_requested = 1;
    MPI_Ibcast(&is_stop_requested, 1, MPI_INT, ROOT, comm, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
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

  private:
  MPI_Comm comm;
  int is_stop_requested;
  std::queue<RequestType> requests;
};

template <class RequestType, class ResponseType, class TaskForwardIt, class Function>
void master(MPI_Comm comm, TaskForwardIt tasks_begin, TaskForwardIt tasks_end, Function master_fn)
{
    //create worker queue
    int size;
    std::queue<int> workers;
    MPI_Comm_size(comm, &size);
    for (int i = 1; i < size; ++i) {
      workers.push(i);
    }

    //create task queue

    MasterTaskManager<RequestType> task_manager(comm, tasks_begin, tasks_end);

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
          maybe_stop_token(master_fn, response, task_manager);
          workers.push(response_status.MPI_SOURCE);
          outstanding--;
          break;
        case worker_status::cancel:
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

}

template <typename Function, class Message, class RequestType,
          typename = void>
struct takes_stop_token : std::false_type
{};

template <typename Function, class Message, class RequestType>
struct takes_stop_token<
  Function, Message, RequestType,
  std::void_t<decltype(std::declval<Function>()(
    std::declval<Message>(), std::declval<TaskManager<RequestType>&>()))>> : std::true_type
{};

template <class Function, class Message, class RequestType, class Enable = void>
struct maybe_stop_token_impl {
    static auto call(Function f, Message m, TaskManager<RequestType>&) {
      return f(m);
    }
};


template <class Function, class Message, class RequestType>
struct maybe_stop_token_impl<Function, Message, RequestType,
  typename std::enable_if_t<takes_stop_token<Function,Message, RequestType>::value>> {
    static auto call(Function f, Message m, TaskManager<RequestType>& s) {
      return f(m,s);
    }
};

template <class Function, class Message, class RequestType>
auto maybe_stop_token(Function f, Message m, TaskManager<RequestType>& s)
{
  return maybe_stop_token_impl<Function, Message, RequestType>::call(f, m, s);
}

template <class RequestType, class ResponseType, class Function>
void worker(MPI_Comm comm, Function worker_fn)
{

  int rank;
  MPI_Comm_rank(comm, &rank);
  WorkerTaskManager<RequestType, ResponseType> stop_token(comm);

  bool worker_done = false;
  while(!worker_done) {
    MPI_Status task_status;
    RequestType request;
    comm::recv(request, ROOT, MPI_ANY_TAG, comm, &task_status);

    switch(worker_status(task_status.MPI_TAG)) {
      case worker_status::done:
        worker_done = true;
        break;
      case worker_status::more:
        {
          auto response = maybe_stop_token(worker_fn, request, stop_token);
          comm::send(response, ROOT, (int)worker_status::done, comm);
        }
        break;
      case worker_status::cancel:
      case worker_status::new_task:
        MPI_Abort(comm, 2);
        break;
    }
  }

  stop_token.wait_stopped();
}

template <class RequestType>
class NoWorkersTaskManager: public TaskManager<RequestType> {
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


  private:
  bool is_stop_requested = false;
  std::queue<RequestType> requests{};
};

template <class RequestType, class ResponseType, class TaskForwardIt, class WorkerFn, class MasterFn>
void no_workers(TaskForwardIt tasks_begin, TaskForwardIt tasks_end, MasterFn master_fn, WorkerFn worker_fn) {
  NoWorkersTaskManager<RequestType> task_manager(tasks_begin, tasks_end);

  while(!task_manager.empty() && !task_manager.stop_requested()) {
    RequestType task = task_manager.front();
    task_manager.pop();

    auto response = maybe_stop_token(worker_fn, task, task_manager);
    maybe_stop_token(master_fn, response, task_manager);
  }
}

}
}
}
