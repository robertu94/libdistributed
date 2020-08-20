#include <libdistributed_comm.h>

distributed::comm::serializer::type_registry&
distributed::comm::serializer::get_type_registry()
{
  static type_registry registery;
  return registery;
}
  
