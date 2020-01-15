#include <iterator>
#include <queue>
#include <type_traits>
#include <mpi.h>

#include "stop_token.h"

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

template <class T, class Enable = void>
struct type_or_value_type {
  using type = T;
};

template <class T>
struct type_or_value_type<T, std::enable_if_t<is_iterable<T>::value>> {
  using type = typename T::value_type;
};


constexpr int ROOT = 0;
enum class worker_status: int {
  done = 1,
  more = 2,
  cancel = 3
};


class WorkerStopToken : public StopToken
{
public:
  WorkerStopToken(MPI_Comm comm, MPI_Request request)
    : comm(comm)
    , stop_request(request)
    , flag(0)
  {}

  bool stop_requested() override {
    if(flag) {
      MPI_Test(&stop_request, &flag, MPI_STATUS_IGNORE);
    }
    return flag;
  }

  void request_stop() override {
    int done = 1;
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Request request;
    MPI_Isend(&done, 1, MPI_INT, 0, (int)worker_status::cancel, comm, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  private:
  MPI_Comm comm;
  MPI_Request stop_request;
  int flag;
};

class MasterStopToken : public StopToken
{
public:
  MasterStopToken(MPI_Comm comm)
    : StopToken(),
      comm(comm),
      is_stop_requested(0)
  {}

  bool stop_requested() override {
    return is_stop_requested == 1;
  }

  void request_stop() override {
    MPI_Request request;
    is_stop_requested = 1;
    MPI_Ibcast(&is_stop_requested, 1, MPI_INT, ROOT, comm, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  private:
  MPI_Comm comm;
  int is_stop_requested;
};

template <class RequestType, class ResponseType, class TaskForwardIt, class Function>
void master(MPI_Comm comm, MPI_Datatype request_dtype, MPI_Datatype response_dtype, TaskForwardIt tasks_begin, TaskForwardIt tasks_end, Function master_fn)
{
    //create worker queue
    int size;
    std::queue<int> workers;
    MPI_Comm_size(comm, &size);
    for (int i = 1; i < size; ++i) {
      workers.push(i);
    }

    MasterStopToken stop_token(comm);

    int outstanding = 0;
    while((tasks_begin != tasks_end and !stop_token.stop_requested()) or outstanding > 0) {

      while(tasks_begin != tasks_end and !stop_token.stop_requested() and !workers.empty()) {
        int worker_id = workers.front();
        ++outstanding;
        workers.pop();

        RequestType request = *tasks_begin;
        ++tasks_begin;

        MPI_Request mpi_request;
        MPI_Isend(&request, 1, request_dtype, worker_id, (int)worker_status::more, comm, &mpi_request);
        MPI_Wait(&mpi_request, MPI_STATUS_IGNORE);
      }

      MPI_Status response_status;
      MPI_Request mpi_response;
      ResponseType response;

      MPI_Irecv(&response, 1, response_dtype, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &mpi_response);
      MPI_Wait(&mpi_response, &response_status);
      switch(worker_status(response_status.MPI_TAG)) {
        case worker_status::more:
          maybe_stop_token(master_fn, response, stop_token);
          break;
        case worker_status::done:
          workers.push(response_status.MPI_SOURCE);
          outstanding--;
          break;
        case worker_status::cancel:
          stop_token.request_stop();
          break;
      }
    }
    
    if(not stop_token.stop_requested()) stop_token.request_stop();

    while(not workers.empty()) {
      int worker_id = workers.front();
      workers.pop();

      RequestType request;
      MPI_Request mpi_request;
      MPI_Isend(&request, 1, request_dtype, worker_id, (int)worker_status::done, comm, &mpi_request);
      MPI_Wait(&mpi_request, MPI_STATUS_IGNORE);
    }

}

template <class IterableType>
typename std::enable_if<is_iterable<IterableType>::value, void>::type 
worker_send(MPI_Comm comm, MPI_Datatype response_dtype, IterableType iterable)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  auto begin = std::begin(iterable);
  auto end = std::end(iterable);
  auto value = *begin;
  MPI_Request mpi_request;
  for (; begin != end; ++begin) {
    MPI_Isend(&value, 1, response_dtype, ROOT, (int)worker_status::more, comm,
              &mpi_request);
    MPI_Wait(&mpi_request, MPI_STATUS_IGNORE);
  }
  MPI_Isend(value, 1, response_dtype, ROOT, (int)worker_status::done, comm, &mpi_request);
  MPI_Wait(&mpi_request, MPI_STATUS_IGNORE);
}

template <class ValueType>
typename std::enable_if<!is_iterable<ValueType>::value, void>::type 
worker_send(MPI_Comm comm,  MPI_Datatype response_dtype, ValueType value)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Request request;
    MPI_Isend(&value, 1, response_dtype, ROOT, (int)worker_status::more, comm, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    MPI_Isend(&value, 1, response_dtype, ROOT, (int)worker_status::done, comm, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

template <typename Function, class Message,
          typename = void>
struct takes_stop_token : std::false_type
{};

template <typename Function, class Message>
struct takes_stop_token<
  Function, Message,
  std::void_t<decltype(std::declval<Function>()(
    std::declval<Message>(), std::declval<StopToken&>()))>> : std::true_type
{};

template <class Function, class Message, class Enable = void>
struct maybe_stop_token_impl {
    static auto call(Function f, Message m, StopToken&) {
      return f(m);
    }
};


template <class Function, class Message>
struct maybe_stop_token_impl<Function, Message, 
  typename std::enable_if_t<takes_stop_token<Function,Message>::value>> {
    static auto call(Function f, Message m, StopToken& s) {
      return f(m,s);
    }
};

template <class Function, class Message>
auto maybe_stop_token(Function f, Message m, StopToken& s)
{
  return maybe_stop_token_impl<Function, Message>::call(f, m, s);
}

template <class RequestType, class ResponseType, class Function>
void worker(MPI_Comm comm, MPI_Datatype request_dtype, MPI_Datatype response_dtype, Function worker_fn)
{

  int rank;
  MPI_Comm_rank(comm, &rank);
  int done = 0;
  MPI_Request stop_request;
  MPI_Ibcast(&done, 1, MPI_INT, ROOT, comm, &stop_request);
  WorkerStopToken stop_token(comm, stop_request);

  bool worker_done = false;
  while(!worker_done) {
    MPI_Request task_request;
    MPI_Status task_status;
    RequestType request;
    MPI_Irecv(&request, 1, request_dtype, ROOT, MPI_ANY_TAG, comm, &task_request);
    MPI_Wait(&task_request, &task_status);

    switch(worker_status(task_status.MPI_TAG)) {
      case worker_status::done:
        worker_done = true;
        break;
      case worker_status::more:
        {
          auto response = maybe_stop_token(worker_fn, request, stop_token);
          worker_send(comm, response_dtype, response);
        }
        break;
      case worker_status::cancel:
        MPI_Abort(comm, 2);
        break;
    }
  }
  MPI_Wait(&stop_request, MPI_STATUS_IGNORE);
}

}
}
}
