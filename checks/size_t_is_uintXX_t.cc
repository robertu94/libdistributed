#include <cstdint>
#include <cstddef>
#include <type_traits>

int main(int argc, char *argv[])
{
  static_assert(( std::is_same<size_t, uint8_t>::value ||
      std::is_same<size_t, uint16_t>::value ||
      std::is_same<size_t, uint32_t>::value ||
      std::is_same<size_t, uint64_t>::value )
      ,"size_t is not uintXX_t");
  return 0;
}
