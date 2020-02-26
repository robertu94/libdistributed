#ifndef LIBDISTRIBUTED_TYPES_H
#define LIBDISTRIBUTED_TYPES_H
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

  }
}

#endif
