cmake_minimum_required(VERSION 3.12)
project(libdistributed VERSION "0.0.10" LANGUAGES CXX)

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

set(default_cxx_standard 17)
if(NOT LIBDISTRIBUTED_CXX_VERSION)
  set(LIBDISTRIBUTED_CXX_VERSION ${default_cxx_standard} CACHE 
    STRING "Which version of the cxx standard to expect" FORCE)
  set_property(CACHE LIBDISTRIBUTED_CXX_VERSION PROPERTY STRINGS 17)
endif()

include(GNUInstallDirs)

find_package(MPI COMPONENTS CXX)

option(BUILD_SHARED_LIBS "BUILD SHARED LIBRARIES" ON)
add_library(libdistributed
  #core features
    src/work_queue.cc

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
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libdistributed>
  )
target_link_libraries( libdistributed PUBLIC MPI::MPI_CXX)
target_compile_features(libdistributed PUBLIC cxx_std_${LIBDISTRIBUTED_CXX_VERSION})

option(USE_CLANG_TIDY "include clang-tidy warnings in the build log" OFF)
if(USE_CLANG_TIDY)
  find_program(CLANG_TIDY clang-tidy)
  set_target_properties(libdistributed PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY}")
  set_target_properties(libdistributed PROPERTIES C_CLANG_TIDY "${CLANG_TIDY}")
endif()

export(TARGETS libdistributed NAMESPACE LibDistributed:: FILE
  LibDistributedConfig.cmake)
install(TARGETS libdistributed EXPORT LibDistributedConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
install(EXPORT LibDistributedConfig NAMESPACE LibDistributed:: DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/LibDistributed/cmake)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libdistributed)

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

