#include <array>
#include <vector>
#include <algorithm>
#include <mpi.h>
/** \file 
 *  \brief utilities for working with types
 */

namespace distributed {
  namespace types {

    /**
     * a class that provides introspective information on a MPI_Datatype
     */
    class type_info {
      public:
      /**
       * \param[in] dtype the datatype to inspect
       */
      type_info(MPI_Datatype dtype) {
        int num_integers, num_addresses, num_datatypes;
        MPI_Type_get_envelope(dtype, &num_integers, &num_addresses, &num_datatypes, &combiner);
        integers.resize(num_integers);
        addresses.resize(num_addresses);
        dtypes.resize(num_datatypes);
        if(combiner != MPI_COMBINER_NAMED) {
          MPI_Type_get_contents(dtype, num_integers, num_addresses, num_datatypes, integers.data(), addresses.data(), dtypes.data());
        }
      }

      /**
       * \returns the combiner type for this MPI_Datatype
       * \see MPI_Type_get_envelope
       */
      int get_combiner() const {
        return combiner;
      }

      /**
       * \returns the integers passed to create this type, (think strides, array sizes, etc...)
       * \see MPI_Type_get_envelope
       */
      std::vector<int> const& get_integers() const {
        return integers;
      }

      /**
       * \returns the addresses passed to create this type (think offsets in a struct)
       * \see MPI_Type_get_envelope
       */
      std::vector<MPI_Aint> const& get_addresses() const {
        return addresses;
      }

      /**
       * \returns the MPI_Datatypes that make up this type
       * \see MPI_Type_get_envelope
       */
      std::vector<MPI_Datatype> const& get_dtypes() const {
        return dtypes;
      }

      private:
      int combiner;
      std::vector<int> integers;
      std::vector<MPI_Aint> addresses;
      std::vector<MPI_Datatype> dtypes;
    };

    /**
     * a struct that converts from a type to a MPI_Datatype
     */
    template <class T>
    struct type_to_datatype;

    namespace {

      MPI_Aint address_helper(void* location)
      {
        MPI_Aint address;
        MPI_Get_address(location, &address);
        return address;
      }

      template <class Tuple, size_t... Is>
      std::array<MPI_Aint, sizeof...(Is)> make_displacements_impl(Tuple arg, std::index_sequence<Is...>)
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

      template <class... T>
      std::array<MPI_Aint, sizeof...(T)> make_displacements(std::tuple<T...>& arg)
      {
        return make_displacements_impl(arg, std::index_sequence_for<T...>{});
      }


      template <class Tuple, size_t... Is>
      std::array<MPI_Datatype, sizeof...(Is)> make_dtypes_impl(Tuple& arg, std::index_sequence<Is...>)
      {
        return { 
          type_to_datatype<typename std::tuple_element<Is,Tuple>::type>::dtype() ...,
        };
      }

      template <class... T>
      std::array<MPI_Datatype, sizeof...(T)> make_dtypes(std::tuple<T...>& arg)
      {
        return make_dtypes_impl(arg, std::index_sequence_for<T...>{});
      }
    }

      /** converts a type to MPI_Datatype */
      template <>
      struct type_to_datatype<float>
      {
        /** \returns MPI_FLOAT */
        static MPI_Datatype dtype() { return MPI_FLOAT; }
      };

      /** converts a type to MPI_Datatype */
      template <>
      struct type_to_datatype<int>
      {
        /** \returns MPI_INT */
        static MPI_Datatype dtype() { return MPI_INT; }
      };

      /** converts a type to MPI_Datatype */
      template <>
      struct type_to_datatype<double>
      {
        /** \returns MPI_DOUBLE */
        static MPI_Datatype dtype() { return MPI_DOUBLE; } 
      };

      /** converts a type to MPI_Datatype */
      template <class T, size_t N>
      struct type_to_datatype<std::array<T,N>>
      {
        /** \returns a MPI_Type_contigious datatype */
        static MPI_Datatype dtype() {
          MPI_Datatype type;
          MPI_Datatype element_type = type_to_datatype<T>::dtype();
          MPI_Type_contiguous(N, type_to_datatype<T>::dtype(), &type);
          MPI_Type_commit(&type);
          return type;
        }
      };

      /** converts a type to MPI_Datatype */
      template <class... T>
      struct type_to_datatype<std::tuple<T...>>
      {
        /** \returns a MPI_Type_struct datatype */
        static MPI_Datatype dtype() {
          std::tuple<T...> arg;
          MPI_Datatype type;
          constexpr int length = sizeof...(T);
          std::array<int, length> blocklengths;
          blocklengths.fill(1);
          std::array<MPI_Aint, length> displacements = make_displacements<T...>(arg);
          std::array<MPI_Datatype, length> dtypes = make_dtypes<T...>(arg);

          MPI_Type_create_struct(
              length,
              blocklengths.data(),
              displacements.data(),
              dtypes.data(),
              &type
              );
          MPI_Type_commit(&type);
          return type;
        }
      };

  }
}
