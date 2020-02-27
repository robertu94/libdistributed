#ifndef LIBDISTRIBUTED_STOP_TOKEN_H
#define LIBDISTRIBUTED_STOP_TOKEN_H

/**
 * \file
 * \brief  Manage a running `work_queue`
 */

namespace distributed {
namespace queue {

class StopToken {
  public:
  virtual ~StopToken()=default;

  /**
   * \returns true if a stop has been requested and recieved by this rank
   */
  virtual bool stop_requested()=0;
  /**
   * Request that the current work_queue be stopped.
   *
   * After a stop request is received, no additional requests will be
   * processed; however, running tasks will continue running until the return.
   * Running tasks SHOULD periodically check stop_requested's return value to
   * see if they should attempt to terminate early.
   */
  virtual void request_stop()=0;

};
/**
 * A distributed token which can be used to request stopping computation on a
 * work_queue
 */
template <class RequestType>
class TaskManager: public StopToken {
  public:
  virtual ~TaskManager()=default;
  /**
   * Ask the manager to queue a new request
   *
   * \param[in] request the request that you would like the master to enqueue
   */
  virtual void push(RequestType const& request)=0;
};

}
}

#endif
