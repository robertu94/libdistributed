#ifndef LIBDISTRIBUTED_COMM_H
#define LIBDISTRIBUTED_COMM_H



#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <ostream>
#include <sstream>
#include <tuple>
#include <std_compat/type_traits.h>
#include <utility>
#include <vector>
#include <map>
#include <string>
#include <std_compat/optional.h>
#include <std_compat/variant.h>
#include <mpi.h>
#include "libdistributed_version.h"

/**
 * \file
 * \brief send, recv, and bcast operations for c++ types
 *
 * Users can serialize their own types by specializing distributed::comm::serializer::serializer for their types.
 */

namespace distributed {
namespace comm {
namespace serializer {

namespace {
  MPI_Datatype mpi_size_t() {
    return (std::is_same<uint8_t, size_t>::value) ? MPI_UINT8_T :
           (std::is_same<uint16_t, size_t>::value) ? MPI_UINT16_T :
           (std::is_same<uint32_t, size_t>::value) ? MPI_UINT32_T :
           (std::is_same<uint64_t, size_t>::value) ? MPI_UINT64_T :
           MPI_INT;
  }


  template <template <class...> class Container, size_t N, class IndexedPredicate, class... T>
  struct fold_or_impl {
    template <class... Args>
    static 
    auto
    fold(Container<T...>& v, IndexedPredicate f, Args&... args)
    -> decltype(f.template operator()<N>(v, args...))
    {
      return f.template operator()<N>(v, args...) ||
        fold_or_impl<Container, N-1, IndexedPredicate, T...>::fold(v, f, args...);

    }
    template <class... Args>
    static 
    auto
    fold(Container<T...>const& v, IndexedPredicate f, Args&... args)
    -> decltype(f.template operator()<N>(v, args...))
    {
      return f.template operator()<N>(v, args...) ||
        fold_or_impl<Container, N-1, IndexedPredicate, T...>::fold(v, f, args...);

    }
  };

  template <template <class...> class Container, class IndexedPredicate, class... T>
  struct fold_or_impl<Container, 0, IndexedPredicate, T...> {
    template <class... Args>
    static auto fold(Container<T...>& v, IndexedPredicate f, Args&... args)
    -> decltype(f.template operator()<0>(v, args...))
    {
      return f.template operator()<0>(v, args...);
    }
    template <class... Args>
    static auto fold(Container<T...>const& v, IndexedPredicate f, Args&... args)
    -> decltype(f.template operator()<0>(v, args...))
    {
      return f.template operator()<0>(v, args...);
    }
  };

  template <template <class...> class Container, class... T, class IndexedPredicates, class... Args>
  auto fold_or(Container<T...>& v, IndexedPredicates f, Args&&... args)
    -> decltype(f.template operator()<0>(v, args...))
  {
    
    return fold_or_impl<Container, sizeof...(T)-1, IndexedPredicates, T...>::fold(v,f, std::forward<Args>(args)...);
  }

  template <template <class...> class Container, class... T, class IndexedPredicates, class... Args>
  auto fold_or(Container<T...>const& v, IndexedPredicates f, Args&&... args)
    -> decltype(f.template operator()<0>(v, args...))
  {
    
    return fold_or_impl<Container, sizeof...(T)-1, IndexedPredicates, T...>::fold(v,f, std::forward<Args>(args)...);
  }

  template <template <class...> class Container, size_t N, class IndexedPredicate, class... T>
  struct fold_comma_impl {
    template <class... Args>
    static 
    auto
    fold(Container<T...>& v, IndexedPredicate f, Args&... args)
    -> decltype(f.template operator()<N>(v, args...))
    {
      fold_comma_impl<Container, N-1, IndexedPredicate, T...>::fold(v, f, args...);
      return f.template operator()<N>(v, args...);
    }

    template <class... Args>
    static 
    auto
    fold(Container<T...>const& v, IndexedPredicate f, Args&... args)
    -> decltype(f.template operator()<N>(v, args...))
    {
      fold_comma_impl<Container, N-1, IndexedPredicate, T...>::fold(v, f, args...);
      return f.template operator()<N>(v, args...);
    }
  };

  template <template <class...> class Container, class IndexedPredicate, class... T>
  struct fold_comma_impl<Container, 0, IndexedPredicate, T...> {
    template <class... Args>
    static auto fold(Container<T...>& v, IndexedPredicate f, Args&... args)
    -> decltype(f.template operator()<0>(v, args...))
    {
      return f.template operator()<0>(v, args...);
    }
    template <class... Args>
    static auto fold(Container<T...>const & v, IndexedPredicate f, Args&... args)
    -> decltype(f.template operator()<0>(v, args...))
    {
      return f.template operator()<0>(v, args...);
    }
  };

  template <template <class...> class Container, class... T, class IndexedPredicates, class... Args>
  auto fold_comma(Container<T...>& v, IndexedPredicates f, Args&&... args)
    -> decltype(f.template operator()<0>(v, args...))
  {
    
    return fold_comma_impl<Container, sizeof...(T)-1, IndexedPredicates, T...>::fold(v,f, std::forward<Args>(args)...);
  }


  template <template <class...> class Container, class... T, class IndexedPredicates, class... Args>
  auto fold_comma(Container<T...> const& v, IndexedPredicates f, Args&&... args)
    -> decltype(f.template operator()<0>(v, args...))
  {
    
    return fold_comma_impl<Container, sizeof...(T)-1, IndexedPredicates, T...>::fold(v,f, std::forward<Args>(args)...);
  }

}

struct datatype_manager
{
  datatype_manager(MPI_Datatype type)
    : type(type)
  {}
  ~datatype_manager() { MPI_Type_free(&type); }

  MPI_Datatype type;
};


/** type_registry register MPI_Datatypes generated by serialization */
using type_registry = std::map<std::string, datatype_manager>;

/** get a handle to the type registry */
type_registry& get_type_registry();

/** base serialization template works for all iterable types*/
template <class T>
struct serializer
{
  /** is the type serializable using MPI_Datatypes for both the sender and
   * receiver at compile time?*/
  using mpi_type = std::false_type;
};

/**
 *  defines the core MPI_Datatypes
 * \param[in] data_type
 * \param[in] mpi_dtype
 */
#define define_basic_type(data_type, mpi_dtype)                                \
  /** specialization of the serialization trait for data_type */               \
  template <>                                                                  \
  struct serializer<data_type>                                                 \
  {                                                                            \
    /** true or false, the type can be represented at compile time as a MPI_Datatype*/ \
    using mpi_type = std::true_type;                                           \
    /** \returns a MPI_Datatype to represent the type if mpi_type is true, else MPI_INT */ \
    static MPI_Datatype dtype() { return mpi_dtype; }                                 \
    /** \returns a string representing the name of the type */ \
    static std::string name() { return #mpi_dtype; }                           \
                                                                               \
    /** 
     Sends a data type from one location to another
      \param[in] t the data to send
      \param[in] dest the MPI rank to send to
      \param[in] tag the MPI tag to send to
      \param[in] comm the MPI_Comm to send to
      \returns an error code from the underlying send */ \
    static int send(data_type const& t, int dest, int tag, MPI_Comm comm)      \
    {                                                                          \
      return MPI_Send(&t, 1, mpi_dtype, dest, tag, comm);                      \
    }                                                                          \
    /** 
     Recv a data type from another location
      \param[in] t the data to recv from
      \param[in] source the MPI rank to recv from
      \param[in] tag the MPI tag to recv from
      \param[in] comm the MPI_Comm to recv from
      \param[in] status the MPI_Status to recv from
      \returns an error code from the underlying recv */ \
    static int recv(data_type& t, int source, int tag, MPI_Comm comm,          \
                    MPI_Status* status)                                        \
    {                                                                          \
      return MPI_Recv(&t, 1, mpi_dtype, source, tag, comm, status);            \
    }                                                                          \
                                                                               \
    /** 
     Broadcast a data type from another location
      \param[in] t the data to broadcast from
      \param[in] root the MPI rank to broadcast from
      \param[in] comm the MPI_Comm to broadcast from
      \returns an error code from the underlying MPI_Bcast(s) */ \
    static int bcast(data_type& t, int root, MPI_Comm comm)           \
    {                                                                          \
      return MPI_Bcast(&t, 1, mpi_dtype, root, comm);                          \
    }                                                                          \
  };

define_basic_type(int8_t, MPI_INT8_T);
define_basic_type(int16_t, MPI_INT16_T);
define_basic_type(int32_t, MPI_INT32_T);
define_basic_type(int64_t, MPI_INT64_T);
define_basic_type(uint8_t, MPI_UINT8_T);
define_basic_type(uint16_t, MPI_UINT16_T);
define_basic_type(uint32_t, MPI_UINT32_T);
define_basic_type(uint64_t, MPI_UINT64_T);
define_basic_type(float, MPI_FLOAT);
define_basic_type(double, MPI_DOUBLE);
define_basic_type(char, MPI_CHAR);
#if !LIBDISTRIBUTED_COMPAT_HAS_SIZE_T_IS_UINTXX_T 
define_basic_type(size_t, mpi_size_t());
#endif

/**
 * serializer for arrays of initialized pointers
 */
template <class T> 
struct serializer<T*>
{
  /** is the type serializable using MPI_Datatypes for both the sender and
   * receiver at compile time?*/
  using mpi_type = typename serializer<T>::mpi_type;
  /** \returns a MPI_Datatype to represent the type if mpi_type is true, else MPI_INT */
  static MPI_Datatype dtype() {
    return serializer<T>::dtype();
  }
  /** \returns a string representing the name of the type */
  static std::string name() {
    return serializer<T>::name() + "*";
  }

  /** 
   * Sends a data type from one location to another
   * \param[in] t the data to send
   * \param[in] dest the MPI rank to send to
   * \param[in] tag the MPI tag to send to
   * \param[in] comm the MPI_Comm to send to
   * \returns an error code from the underlying send */
  static int send(T* const& t, int dest, int tag, MPI_Comm comm) {
    return serializer<T>::send(*t, dest, tag, comm);
  }
  /** 
   Recv a data type from another location
    \param[in] t the data to recv from
    \param[in] source the MPI rank to recv from
    \param[in] tag the MPI tag to recv from
    \param[in] comm the MPI_Comm to recv from
    \param[in] status the MPI_Status to recv from
    \returns an error code from the underlying recv */
  static int recv(T*& t, int source, int tag, MPI_Comm comm,
                  MPI_Status* status) {
    return serializer<T>::recv(*t, source, tag, comm, status);
  }
  
  /** 
   Broadcast a data type from another location
    \param[in] t the data to broadcast from
    \param[in] root the MPI rank to broadcast from
    \param[in] comm the MPI_Comm to broadcast from
    \returns an error code from the underlying MPI_Bcast(s) */
  static int bcast(T*& t, int root, MPI_Comm comm) {
    return serializer<T>::bcast(*t, root, comm);
  }


};

/** specialization of serializion for pair */
template <class T, class V>
struct serializer<std::pair<T, V>>
{
  /** is the type serializable using MPI_Datatypes for both the sender and
   * receiver at compile time?*/
  using mpi_type = compat::conjunction<typename serializer<T>::mpi_type,
                                    typename serializer<V>::mpi_type>;

  /** \returns a MPI_Datatype to represent the type if mpi_type is true, else MPI_INT */
  static MPI_Datatype dtype()
  {
    MPI_Datatype ret;
    type_registry& registery = get_type_registry();
    auto name = serializer<std::pair<T, V>>::name();
    auto it = registery.find(name);
    if (it == registery.end()) {
      std::pair<T, V> exemplar;
      int blocklen[] = { 1, 1 };
      MPI_Aint displacements[2];
      MPI_Get_address(&exemplar.first, &displacements[0]);
      MPI_Get_address(&exemplar.second, &displacements[1]);
      MPI_Aint min_address = std::min(displacements[0], displacements[1]);
      displacements[0] -= min_address;
      displacements[1] -= min_address;
      MPI_Datatype types[] = { serializer<T>::dtype(), serializer<V>::dtype() };
      int err = MPI_Type_create_struct(2, blocklen, displacements, types, &ret);

      MPI_Type_commit(&ret);
      registery.emplace(name, ret);
    } else {
      ret = it->second.type;
    }
    return ret;
  }

  /** \returns a string representing the name of the type */
  static std::string name()
  {
    std::stringstream n;
    n << "std::pair<" << serializer<T>::name() << ", " << serializer<V>::name()
      << ">";
    return n.str();
  }

  /** 
   * Sends a data type from one location to another
   * \param[in] t the data to send
   * \param[in] dest the MPI rank to send to
   * \param[in] tag the MPI tag to send to
   * \param[in] comm the MPI_Comm to send to
   * \returns an error code from the underlying send */
  static int send(std::pair<T, V> const& t, int dest, int tag, MPI_Comm comm)
  {
    if (serializer<std::pair<T,V>>::mpi_type::value) {
      return MPI_Send(&t, 1, dtype(), dest, tag, comm);
    } else {
      return serializer<T>::send(t.first, dest, tag, comm);
      return serializer<V>::send(t.second, dest, tag, comm);
    }
  }

  /** 
   * Recv a data type from another location
   * \param[in] t the data to recv from
   * \param[in] source the MPI rank to recv from
   * \param[in] tag the MPI tag to recv from
   * \param[in] comm the MPI_Comm to recv from
   * \param[in] status the MPI_Status to recv from
   * \returns an error code from the underlying recv */
  static int recv(std::pair<T, V>& t, int source, int tag, MPI_Comm comm,
                  MPI_Status* status)
  {
    MPI_Status alt_status;
    if(status == MPI_STATUS_IGNORE) status = &alt_status;
    status->MPI_SOURCE = source;
    status->MPI_TAG = tag;

    if (serializer<std::pair<T,V>>::mpi_type::value) {
      return MPI_Recv(&t, 1, dtype(), source, tag, comm, status);
    } else {
      return serializer<T>::recv(t.first, source, tag, comm, status);
      return serializer<V>::recv(t.second, source, tag, comm, status);
    }
  }
  /** 
   * Broadcast a data type from another location
   * \param[in] t the data to broadcast from
   * \param[in] root the MPI rank to broadcast from
   * \param[in] comm the MPI_Comm to broadcast from
   * \returns an error code from the underlying MPI_Bcast(s) */
  static int bcast(std::pair<T, V>& t, int root, MPI_Comm comm)
  {
    if (serializer<std::pair<T,V>>::mpi_type::value) {
      return MPI_Bcast(&t, 1, dtype(), root, comm);
    } else {
      return serializer<T>::bcast(t.first, root, comm);
      return serializer<V>::bcast(t.second, root, comm);
    }
  }
};

template <class T, class... Rest>
struct printer {
  static std::ostream& print(std::ostream& out) {
    out << serializer<T>::name() << ',';
    return printer<Rest...>::print(out);
  }
};

template <class T>
struct printer<T> {
  static std::ostream& print(std::ostream& out) {
    return out << serializer<T>::name() << ',';
  }
};
template <class... T>
std::ostream& make_name(std::ostream& out) {
  return printer<T...>::print(out);
}

/** specialization of serializion for tuple */
template <class... T>
struct serializer<std::tuple<T...>>
{
  /** is the type serializable using MPI_Datatypes for both the sender and
   * receiver at compile time?*/
  using mpi_type = compat::conjunction<typename serializer<T>::mpi_type...>;

  /** \returns a string representing the name of the type */
  static std::string name() { 
    std::stringstream n;
    n << "std::tuple<";
    make_name<T...>(n);
    n << ">";
    return n.str();
  }
  /** \returns a MPI_Datatype to represent the type if mpi_type is true, else MPI_INT */
  static MPI_Datatype dtype() { 
    if (mpi_type::value) {
      type_registry& registery = get_type_registry();
      auto type_name = name();
      auto it = registery.find(type_name);
      if(it == registery.end()) {
          std::tuple<T...> arg;
          MPI_Datatype type;
          constexpr int length = sizeof...(T);
          std::array<int, length> blocklengths;
          blocklengths.fill(1);
          std::array<MPI_Aint, length> displacements = make_displacements(arg);
          std::array<MPI_Datatype, length> dtypes = make_dtypes(arg);

          MPI_Type_create_struct(
              length,
              blocklengths.data(),
              displacements.data(),
              dtypes.data(),
              &type
              );
          MPI_Type_commit(&type);
          registery.emplace(type_name, type);
          return type;
      } else {
        return it->second.type;
      }
    } else {
      return MPI_INT;
    }
  }
  /** 
   * Sends a data type from one location to another
   * \param[in] t the data to send
   * \param[in] dest the MPI rank to send to
   * \param[in] tag the MPI tag to send to
   * \param[in] comm the MPI_Comm to send to
   * \returns an error code from the underlying send */
  static int send(std::tuple<T...> const& t, int dest, int tag, MPI_Comm comm)
  {
    if(mpi_type::value) {
      return MPI_Send(&t, 1, dtype(), dest, tag, comm);
    } else {
      send_each(t, dest, tag, comm);
    }
    return 0;
  }

  /** 
   * Recv a data type from another location
   * \param[in] t the data to recv from
   * \param[in] source the MPI rank to recv from
   * \param[in] tag the MPI tag to recv from
   * \param[in] comm the MPI_Comm to recv from
   * \param[in] status the MPI_Status to recv from
   * \returns an error code from the underlying recv */
  static int recv(std::tuple<T...>& t, int source, int tag, MPI_Comm comm,
                  MPI_Status* status)
  {
    MPI_Status alt_status;
    if(mpi_type::value) {
      return MPI_Recv(&t, 1, dtype(), source, tag, comm, status);
    } else {
      if(status == MPI_STATUS_IGNORE) status = &alt_status;
      status->MPI_SOURCE = source;
      status->MPI_TAG = tag;
      recv_each(t, comm, status);
    }
    return 0;
  }

  /** 
   * Broadcast a data type from another location
   * \param[in] t the data to broadcast from
   * \param[in] root the MPI rank to broadcast from
   * \param[in] comm the MPI_Comm to broadcast from
   * \returns an error code from the underlying MPI_Bcast(s) */
  static int bcast(std::tuple<T...>& t, int root, MPI_Comm comm)
  {
    if(mpi_type::value) {
      return MPI_Bcast(&t, 1, dtype(), root, comm);
    } else {
      return bcast_each(t, root, comm);
    }
    return 0;
  }

  private:
      static MPI_Aint address_helper(void* location)
      {
        MPI_Aint address;
        MPI_Get_address(location, &address);
        return address;
      }

      template <class Tuple, size_t... Is>
      static std::array<MPI_Aint, sizeof...(Is)> make_displacements_impl(Tuple arg, std::index_sequence<Is...>)
      {
        std::array<MPI_Aint, sizeof...(Is)> displacements =  {
          address_helper((void*)&std::get<Is>(arg))...,
        };
        MPI_Aint min = *std::min_element(std::begin(displacements), std::end(displacements));
        for (auto& displace : displacements) {
          displace -= min;
        }
        
        return displacements;
      }

      static std::array<MPI_Aint, sizeof...(T)> make_displacements(std::tuple<T...>& arg)
      {
        return make_displacements_impl(arg, std::index_sequence_for<T...>{});
      }


      template <class Tuple, size_t... Is>
      static std::array<MPI_Datatype, sizeof...(Is)> make_dtypes_impl(Tuple& arg, std::index_sequence<Is...>)
      {
        return { 
          serializer<typename std::tuple_element<Is,Tuple>::type>::dtype() ...,
        };
      }

      static std::array<MPI_Datatype, sizeof...(T)> make_dtypes(std::tuple<T...>& arg)
      {
        return make_dtypes_impl(arg, std::index_sequence_for<T...>{});
      }


      struct recv_impl{
        template<size_t N, class Tuple>
        void operator()(Tuple& arg, MPI_Comm comm, MPI_Status* status) {
          serializer<typename std::tuple_element<N,Tuple>::type>::recv(std::get<N>(arg), status->MPI_SOURCE, status->MPI_TAG, comm, status);
        }
      };
      template <class Tuple>
      static int recv_each_impl(Tuple& arg, MPI_Comm comm, MPI_Status* status) {
        fold_comma(arg, recv_impl{}, comm, status);
        return 0;
      }

      static int recv_each(std::tuple<T...>& arg, MPI_Comm comm, MPI_Status* status) {
        return recv_each_impl(arg, comm, status);
      }


      struct send_impl{
        template<size_t N, class Tuple>
        void operator()(Tuple const& arg, int dest, int tag, MPI_Comm comm) {
          serializer<typename std::tuple_element<N,Tuple>::type>::send(std::get<N>(arg), dest, tag, comm);
        }
      };
      template <class Tuple>
      static int send_each_impl(Tuple const& arg, int dest, int tag, MPI_Comm comm) {
        fold_comma(arg, send_impl{}, dest, tag, comm);
        return 0;
      }

      static int send_each(std::tuple<T...> const& arg, int dest, int tag, MPI_Comm comm) {
        return send_each_impl(arg, dest, tag, comm);
      }

      struct bcast_impl{
        template <size_t N, class Tuple>
        void operator()(Tuple& arg, int root, MPI_Comm comm) {
          serializer<typename std::tuple_element<N,Tuple>::type>::bcast(std::get<N>(arg), root, comm);
        }
      };
      template <class Tuple>
      static int bcast_each_impl(Tuple& arg, int root, MPI_Comm comm) {
        fold_comma(arg, bcast_impl{}, root, comm);
        return 0;
      }

      static int bcast_each(std::tuple<T...>& arg, int root, MPI_Comm comm) {
        return bcast_each_impl(arg, root, comm);
      }
};

/** specialization of serializion for variant */
template <class... T>
struct serializer<compat::variant<T...>>
{
  /** is the type serializable using MPI_Datatypes for both the sender and
   * receiver at compile time?*/
  using mpi_type = std::false_type;
  /** \returns a string representing the name of the type */
  static std::string name() {
    std::stringstream ss;
    ss << "compat::variant<"; 
    make_name<T...>(ss);
    //(ss << ... << (serializer<T>::name() + ","));
    ss << '>';
    return ss.str();
  }
  /** \returns a MPI_Datatype to represent the type if mpi_type is true, else MPI_INT */
  static MPI_Datatype dtype()  {
    return MPI_INT;
  }

  /** 
   * Sends a data type from one location to another
   * \param[in] t the data to send
   * \param[in] dest the MPI rank to send to
   * \param[in] tag the MPI tag to send to
   * \param[in] comm the MPI_Comm to send to
   * \returns an error code from the underlying send */
  static int send(compat::variant<T...> const& t, int dest, int tag, MPI_Comm comm)
  {
    size_t index = compat::index(t);
    MPI_Send(&index, 1, mpi_size_t(), dest, tag, comm);
    if(index != compat::variant_npos) {
      send_index(t, index, dest, tag, comm);
    }

    return 0;
  }
  /** 
   * Recv a data type from another location
   * \param[in] t the data to recv from
   * \param[in] source the MPI rank to recv from
   * \param[in] tag the MPI tag to recv from
   * \param[in] comm the MPI_Comm to recv from
   * \param[in] status the MPI_Status to recv from
   * \returns an error code from the underlying recv */
  static int recv(compat::variant<T...>& t, int source, int tag, MPI_Comm comm,
                  MPI_Status* status) {
    MPI_Status alt_status;
    if(status == MPI_STATUS_IGNORE) status = &alt_status;
    size_t index;
    MPI_Recv(&index, 1, mpi_size_t(), source, tag, comm, status);
    if(index != compat::variant_npos) {
      recv_index(t, index, status->MPI_SOURCE, status->MPI_TAG, comm, status);
    }
    return 0;
  }
  /** 
   * Broadcast a data type from another location
   * \param[in] t the data to broadcast from
   * \param[in] root the MPI rank to broadcast from
   * \param[in] comm the MPI_Comm to broadcast from
   * \returns an error code from the underlying MPI_Bcast(s) */
  static int bcast(compat::variant<T...>& t, int root, MPI_Comm comm) {
    size_t index = compat::index(t);
    MPI_Bcast(&index, 1, mpi_size_t(), root, comm);
    if(index != compat::variant_npos) {
      bcast_index(t, index, root, comm);
    }
    return 0;
  }
  private:


  struct send_if {
  template <size_t N, class Tuple>
    int operator()(Tuple const& t, size_t index, int dest, int tag, MPI_Comm comm) {
      if(N == index) {
        using dtype = typename compat::variant_alternative<N,Tuple>::type;
        auto const& value = compat::get<N>(t);
        if(serializer<dtype>::mpi_type::value) {
          return MPI_Send(&value, 1, serializer<dtype>::dtype(), dest, tag, comm);
        } else {
          return serializer<dtype>::send(value, dest, tag, comm);
        }
      }
      return 0;
    }
  };

  static int send_index_impl(compat::variant<T...> const&t, size_t index, int dest, int tag, MPI_Comm comm) {
    return fold_or(t, send_if{}, index, dest, tag, comm);
  }
  static int send_index(compat::variant<T...> const&t, size_t index, int dest, int tag, MPI_Comm comm) {
    return send_index_impl(t, index, dest, tag, comm);
  }


  struct recv_if {
    template <size_t N, class Tuple>
    int operator()(Tuple& t, size_t index, int source, int tag, MPI_Comm comm, MPI_Status* status) {
      int ret = 0;
      if(N == index) {
        using dtype = typename compat::variant_alternative<N,Tuple>::type;
        dtype value;
        if(serializer<dtype>::mpi_type::value) {
          ret = MPI_Recv(&value, 1, serializer<dtype>::dtype(), source, tag, comm, status);
        } else {
          ret = serializer<dtype>::recv(value, source, tag, comm, status);
        }
        t = std::move(value);
      }
      return ret;
    }
  };

  static int recv_index_impl(compat::variant<T...>&t, size_t index, int source, int tag, MPI_Comm comm, MPI_Status* status) {
    return fold_or(t, recv_if{}, index, source, tag, comm, status);
  }
  static int recv_index(compat::variant<T...>&t, size_t index, int source, int tag, MPI_Comm comm, MPI_Status* status) {
    return recv_index_impl(t, index, source, tag, comm, status);
  }

  struct bcast_if{
    template<size_t N, class Tuple>
    int operator()(Tuple& t, size_t index, int root, MPI_Comm comm) {
      int ret = 0;
      if(index == N) {
        using dtype = typename compat::variant_alternative<N,Tuple>::type;
        int rank;
        MPI_Comm_rank(comm, &rank);
        if(rank == root){ 
          dtype value = compat::get<N>(t);
          if(serializer<dtype>::mpi_type::value) {
            MPI_Bcast(&value, 1, serializer<dtype>::dtype(), root, comm);
          } else {
            serializer<dtype>::bcast(value, root, comm);
          }
        } else {
          dtype value;
          if(serializer<dtype>::mpi_type::value) {
            MPI_Bcast(&value, 1, serializer<dtype>::dtype(), root, comm);
          } else {
            serializer<dtype>::bcast(value, root, comm);
          }
          t = value;
        }
      }
      return ret;
    }
  };

  template<size_t... Is>
  static int bcast_index_impl(compat::variant<T...>&t, size_t index, int root, MPI_Comm comm, std::index_sequence<Is...>) {
    return fold_or(t, bcast_if{}, index, root, comm);
  }
  static int bcast_index(compat::variant<T...>&t, size_t index, int root, MPI_Comm comm) {
    return bcast_index_impl(t, index, root, comm, std::index_sequence_for<T...>{});
  }
};

/** specialization of serializion for optional */
template <class T>
struct serializer<compat::optional<T>>
{
  /** is the type serializable using MPI_Datatypes for both the sender and
   * receiver at compile time?*/
  using mpi_type = std::false_type;
  /** \returns a MPI_Datatype to represent the type if mpi_type is true, else MPI_INT */
  static MPI_Datatype dtype() { return MPI_INT; }
  /** \returns a string representing the name of the type */
  static std::string name() {
    std::stringstream ss;
    ss << "compat::optional<" << serializer<T>::name() << '>';
    return ss.str();
  }
  /** 
   * Sends a data type from one location to another
   * \param[in] t the data to send
   * \param[in] dest the MPI rank to send to
   * \param[in] tag the MPI tag to send to
   * \param[in] comm the MPI_Comm to send to
   * \returns an error code from the underlying send */
  static int send(compat::optional<T> const& t, int dest, int tag, MPI_Comm comm)
  {
    int occupied = t.has_value();
    int ret = MPI_Send(&occupied, 1, MPI_INT, dest, tag, comm);
    if(occupied) {
      return serializer<T>::send(t.value(), dest, tag, comm);
    }
    return ret;
  }

  /** 
   * Recv a data type from another location
   * \param[in] t the data to recv from
   * \param[in] source the MPI rank to recv from
   * \param[in] tag the MPI tag to recv from
   * \param[in] comm the MPI_Comm to recv from
   * \param[in] status the MPI_Status to recv from
   * \returns an error code from the underlying recv */
  static int recv(compat::optional<T>& t, int source, int tag, MPI_Comm comm,
                  MPI_Status* status) {
    MPI_Status alt_status;
    if(status == MPI_STATUS_IGNORE) status = &alt_status;
    int occupied;
    int ret = MPI_Recv(&occupied, 1, MPI_INT, source, tag, comm, status);
    if(occupied) {
      T value;
      ret |= serializer<T>::recv(value, status->MPI_SOURCE, status->MPI_TAG, comm, status);
      t = std::move(value);
    }
    return ret;
  }
  /** 
   * Broadcast a data type from another location
   * \param[in] t the data to broadcast from
   * \param[in] root the MPI rank to broadcast from
   * \param[in] comm the MPI_Comm to broadcast from
   * \returns an error code from the underlying MPI_Bcast(s) */
  static int bcast(compat::optional<T>& t, int root, MPI_Comm comm) {
    //has_value may be called if the process has a value or not
    int occupied = t.has_value();
    int ret = MPI_Bcast(&occupied, 1, MPI_INT, root, comm);
    //if the optional does not have a value on the root process, we are done
    if(occupied) {
      int rank;
      MPI_Comm_rank(comm, &rank);
      if(rank == root) {
        //the root process is guaranteed to have memory to send
        ret |= serializer<T>::bcast(*t, root, comm);
      } else {
        //other processes may have no memory, allocate a tempoary
        T value;
        ret |= serializer<T>::bcast(value, root, comm);
        t = std::move(value);
      }
    }
    return ret;
  }
};

/** specialization for serialization for vector types */
template <class T>
struct serializer<std::vector<T>>
{
  /** is the type serializable using MPI_Datatypes for both the sender and
   * receiver at compile time?*/
  using mpi_type = std::false_type;
  /** \returns a string representing the name of the type */
  static std::string name() { 
    std::stringstream n;
    n << "std::vector<" << serializer<T>::name() << ">";
    return n.str();
  }
  /** \returns a MPI_Datatype to represent the type if mpi_type is true, else MPI_INT */
  static MPI_Datatype dtype() { return MPI_INT; }
  /** 
   * Sends a data type from one location to another
   * \param[in] t the data to send
   * \param[in] dest the MPI rank to send to
   * \param[in] tag the MPI tag to send to
   * \param[in] comm the MPI_Comm to send to
   * \returns an error code from the underlying send */
  static int send(std::vector<T> const& t, int dest, int tag, MPI_Comm comm)
  {
    int ret = 0;
    size_t size = t.size();
    ret = MPI_Send(&size, 1, mpi_size_t(), dest, tag, comm);
    if(serializer<T>::mpi_type::value) {
      return MPI_Send(t.data(), t.size(), serializer<T>::dtype(), dest, tag, comm);
    } else { 
      for (auto const& item : t) {
        ret |= serializer<T>::send(item, dest, tag, comm);
      }
    }
    return ret;
  }
  /** 
   * Recv a data type from another location
   * \param[in] t the data to recv from
   * \param[in] source the MPI rank to recv from
   * \param[in] tag the MPI tag to recv from
   * \param[in] comm the MPI_Comm to recv from
   * \param[in] status the MPI_Status to recv from
   * \returns an error code from the underlying recv */
  static int recv(std::vector<T>& t, int source, int tag, MPI_Comm comm,
                  MPI_Status* status)
  {
    MPI_Status alt_status;
    if(status == MPI_STATUS_IGNORE) status = &alt_status;
    size_t size, ret=0;
    ret = MPI_Recv(&size, 1, mpi_size_t(), source, tag, comm, status);
    t.resize(size);

    if (serializer<T>::mpi_type::value) {
      return MPI_Recv(t.data(), t.size(), serializer<T>::dtype(), status->MPI_SOURCE, status->MPI_TAG, comm, status);
    } else {
      for (auto& item : t) {
        ret |= serializer<T>::recv(item, status->MPI_SOURCE, status->MPI_TAG, comm, status);
      }
    }
    return ret;
  }

  /** 
   * Broadcast a data type from another location
   * \param[in] t the data to broadcast from
   * \param[in] root the MPI rank to broadcast from
   * \param[in] comm the MPI_Comm to broadcast from
   * \returns an error code from the underlying MPI_Bcast(s) */
  static int bcast(std::vector<T>& t, int root, MPI_Comm comm)
  {
    size_t size = t.size(), ret=0;
    ret = MPI_Bcast(&size, 1, mpi_size_t(), root, comm);
    t.resize(size);
    if (serializer<T>::mpi_type::value) {
      return MPI_Bcast(t.data(), t.size(), serializer<T>::dtype(), root, comm);
    } else {
      for (auto& item : t) {
        ret |= serializer<T>::bcast(item, root, comm);
      }
    }
    return ret;
  }
};

/** specialization for serialization for vector types */
template <class CharT>
struct serializer<std::basic_string<CharT>>
{
  /** is the type serializable using MPI_Datatypes for both the sender and
   * receiver at compile time?*/
  using mpi_type = std::false_type;
  /** \returns a string representing the name of the type */
  static std::string name() { 
    std::stringstream n;
    n << "std::string<" << serializer<CharT>::name() << ">";
    return n.str();
  }
  /** \returns a MPI_Datatype to represent the type if mpi_type is true, else MPI_INT */
  static MPI_Datatype dtype() { return MPI_INT; }
  /** 
   * Sends a data type from one location to another
   * \param[in] t the data to send
   * \param[in] dest the MPI rank to send to
   * \param[in] tag the MPI tag to send to
   * \param[in] comm the MPI_Comm to send to
   * \returns an error code from the underlying send */
  static int send(std::basic_string<CharT> const& t, int dest, int tag, MPI_Comm comm)
  {
    int ret = 0;
    size_t size = t.size();
    ret = MPI_Send(&size, 1, mpi_size_t(), dest, tag, comm);
    if(serializer<CharT>::mpi_type::value) {
      return MPI_Send(&t.front(), t.size(), serializer<CharT>::dtype(), dest, tag, comm);
    } else { 
      for (auto const& item : t) {
        ret |= serializer<CharT>::send(item, dest, tag, comm);
      }
    }
    return ret;
  }
  /** 
   * Recv a data type from another location
   * \param[in] t the data to recv from
   * \param[in] source the MPI rank to recv from
   * \param[in] tag the MPI tag to recv from
   * \param[in] comm the MPI_Comm to recv from
   * \param[in] status the MPI_Status to recv from
   * \returns an error code from the underlying recv */
  static int recv(std::basic_string<CharT>& t, int source, int tag, MPI_Comm comm,
                  MPI_Status* status)
  {
    MPI_Status alt_status;
    if(status == MPI_STATUS_IGNORE) status = &alt_status;
    size_t size, ret=0;
    ret = MPI_Recv(&size, 1, mpi_size_t(), source, tag, comm, status);
    t.resize(size);

    if (serializer<CharT>::mpi_type::value) {
      return MPI_Recv(&t.front(), t.size(), serializer<CharT>::dtype(), status->MPI_SOURCE, status->MPI_TAG, comm, status);
    } else {
      for (auto& item : t) {
        ret |= serializer<CharT>::recv(item, status->MPI_SOURCE, status->MPI_TAG, comm, status);
      }
    }
    return ret;
  }

  /** 
   * Broadcast a data type from another location
   * \param[in] t the data to broadcast from
   * \param[in] root the MPI rank to broadcast from
   * \param[in] comm the MPI_Comm to broadcast from
   * \returns an error code from the underlying MPI_Bcast(s) */
  static int bcast(std::basic_string<CharT>& t, int root, MPI_Comm comm)
  {
    size_t size = t.size(), ret=0;
    ret = MPI_Bcast(&size, 1, mpi_size_t(), root, comm);
    t.resize(size);
    if(size > 0) {
      if (serializer<CharT>::mpi_type::value) {
        return MPI_Bcast(&t.front(), t.size(), serializer<CharT>::dtype(), root, comm);
      } else {
        for (auto& item : t) {
          ret |= serializer<CharT>::bcast(item, root, comm);
        }
      }
    }
    return ret;
  }
};

template <class T, size_t N>
struct serializer<std::array<T, N>>
{
  /** is the type serializable using MPI_Datatypes for both the sender and
   * receiver at compile time?*/
  using mpi_type = typename serializer<T>::mpi_type;
  /** \returns a string representing the name of the type */
  static std::string name() {
    std::stringstream ss;
    ss << "std::array<" << serializer<T>::name() << '>';
    return ss.str();
  }
  /** \returns a MPI_Datatype to represent the type if mpi_type is true, else MPI_INT */
  static MPI_Datatype dtype() {
    if(mpi_type::value) {
      type_registry& registery = get_type_registry();
      auto type_name = name();
      auto it = registery.find(type_name);
      if(it == registery.end()) {
      MPI_Datatype array_type;
      MPI_Type_contiguous(N, serializer<T>::dtype(),  &array_type);
      MPI_Type_commit(&array_type);
      registery.emplace(name(), array_type);
      return array_type;
      } else  {
        return it->second.type;
      }
        
    } else {
      return MPI_INT;
    }
  }

  /** 
   * Sends a data type from one location to another
   * \param[in] t the data to send
   * \param[in] dest the MPI rank to send to
   * \param[in] tag the MPI tag to send to
   * \param[in] comm the MPI_Comm to send to
   * \returns an error code from the underlying send */
  static int send(std::array<T,N> const& t, int dest, int tag, MPI_Comm comm)
  {
    if(mpi_type::value) {
      return MPI_Send(t.data(), 1, dtype(), dest, tag, comm);
    } else {
      int ret = 0;
      for (size_t i = 0; i < N; ++i) {
        ret |= serializer<T>::send(t[i], dest, tag, comm);
      }
      return ret;
    }
  }

  /** 
   * Recv a data type from another location
   * \param[in] t the data to recv from
   * \param[in] source the MPI rank to recv from
   * \param[in] tag the MPI tag to recv from
   * \param[in] comm the MPI_Comm to recv from
   * \param[in] status the MPI_Status to recv from
   * \returns an error code from the underlying recv */
  static int recv(std::array<T,N>& t, int source, int tag, MPI_Comm comm,
                  MPI_Status* status)
  {
    MPI_Status alt_status;
    if(status == MPI_STATUS_IGNORE) status = &alt_status;
    status->MPI_SOURCE = source;
    status->MPI_TAG = tag;

    if(mpi_type::value) {
      return MPI_Recv(t.data(), 1, dtype(), source, tag, comm, status);
    } else {
      int ret = 0;
      for (size_t i = 0; i < N; ++i) {
        ret |= serializer<T>::recv(t[i], status->MPI_SOURCE, status->MPI_TAG, comm, status);
      }
      return ret;
    }
  }

  /** 
   * Broadcast a data type from another location
   * \param[in] t the data to broadcast from
   * \param[in] root the MPI rank to broadcast from
   * \param[in] comm the MPI_Comm to broadcast from
   * \returns an error code from the underlying MPI_Bcast(s) */
  static int bcast(std::array<T,N>& t, int root, MPI_Comm comm)
  {
    if(mpi_type::value) {
      return MPI_Bcast(t.data(), 1, dtype(), root, comm);
    } else {
      int ret;
      for (size_t i = 0; i < N; ++i) {
        ret |= serializer<T>::bcast(t[i], root, comm);
      }
      return ret;
    }
  }

};

} // namespace serializer

/**
 * helper function to initiate sends
 * \param[in] value which value to send
 * \param[in] dest which destination to use
 * \param[in] tag which tag to use
 * \param[in] comm which communicator to use
 */
template <class T>
int
send(T const& value, int dest, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD)
{
  return serializer::serializer<T>::send(value, dest, tag, comm);
}

/**
 * helper function to initiate recv
 * \param[in] value to recv into
 * \param[in] source which source to use
 * \param[in] tag which tag to use
 * \param[in] comm which communicator to use
 * \param[in] status which status object to use if any
 */
template <class T>
int
recv(T& values, int source, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD,
     MPI_Status* status = MPI_STATUS_IGNORE)
{
  return serializer::serializer<T>::recv(values, source, tag, comm, status);
}

/**
 * helper function to initiate recv
 * \param[in] value to recv into
 * \param[in] root which process to broadcast from
 * \param[in] comm which communicator to use
 */
template <class T>
int
bcast(T& values, int root, MPI_Comm comm = MPI_COMM_WORLD)
{
  return serializer::serializer<T>::bcast(values, root, comm);
}

} // namespace comm
} // namespace distributed
#endif /* end of include guard: LIBDISTRIBUTED_COMM_H */
