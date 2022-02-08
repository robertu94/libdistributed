cmake_minimum_required(VERSION 3.12)
project(libdistributed VERSION "0.3.0" LANGUAGES CXX)

#correct was to set a default build type
# https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type was set. Setting build type to ${default_build_type}.")
  set(CMAKE_BUILD_TYPE ${default_build_type} CACHE 
    STRING "Choose the type to build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

include(CTest)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(check_cpp_standard api)
  string(TOUPPER "${api}" HEADER_SUFFIX)
  try_compile(
    COMPILED
    ${CMAKE_BINARY_DIR}
    SOURCES "${CMAKE_SOURCE_DIR}/checks/${api}.cc" 
  )
  message(STATUS "Checking for ${api}: ${COMPILED}")
  set(LIBDISTRIBUTED_COMPAT_HAS_${HEADER_SUFFIX} ${COMPILED} PARENT_SCOPE)
endfunction()

check_cpp_standard(size_t_is_uintXX_t)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libdistributed_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/libdistributed_version.h
  )

include(GNUInstallDirs)

find_package(std_compat REQUIRED)
find_package(MPI COMPONENTS CXX REQUIRED)

option(BUILD_SHARED_LIBS "BUILD SHARED LIBRARIES" ON)
option(LIBDISTRIBUTED_DEBUG_WORK_QUEUE_TERMINATION "Print Debug information in the work_queue" OFF)
add_library(libdistributed
  #core features
    src/work_queue.cc
    src/comm.cc

  #public headers
        include/libdistributed_task_manager.h
    include/libdistributed_types.h
    include/libdistributed_work_queue.h

  #private headers
    include/libdistributed_work_queue_impl.h
  )
target_include_directories(
  libdistributed
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libdistributed>
  )
target_link_libraries( libdistributed PUBLIC MPI::MPI_CXX std_compat::std_compat)

option(USE_CLANG_TIDY "include clang-tidy warnings in the build log" OFF)
if(USE_CLANG_TIDY)
  find_program(CLANG_TIDY clang-tidy)
  set_target_properties(libdistributed PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY}")
  set_target_properties(libdistributed PROPERTIES C_CLANG_TIDY "${CLANG_TIDY}")
endif()

install(TARGETS libdistributed EXPORT LibDistributedTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
install(EXPORT LibDistributedTargets NAMESPACE LibDistributed:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibDistributed)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/LibDistributedConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/LibDistributedConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibDistributed
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/LibDistributedConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/LibDistributedConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/LibDistributedConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibDistributed
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libdistributed)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/libdistributed_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libdistributed)

option(BUILD_DOCS "build the documetation" OFF)
if(BUILD_DOCS)
  find_package(Doxygen REQUIRED dot)
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_MAN_LINKS YES)
  set(DOXYGEN_GENERATE_MAN YES)
  set(DOXYGEN_EXTRACT_LOCAL_METHODS YES)
  set(DOXYGEN_EXTRACT_STATIC YES)
  set(DOXYGEN_MACRO_EXPANSION YES)
  set(DOXYGEN_EXCLUDE_PATTERNS "*_impl.h")
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
  doxygen_add_docs(
    docs
    ${PROJECT_SOURCE_DIR}/README.md
    ${PROJECT_SOURCE_DIR}/include
    COMMENT "Generate Documenation"
  )
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

