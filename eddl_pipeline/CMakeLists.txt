cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)

if(DEFINED CMAKE_BUILD_TYPE)
  # User defined build type
  set(CMAKE_BUILD_TYPE
      ${CMAKE_BUILD_TYPE}
      CACHE
        STRING
        "Choose the type of build: Release Debug tsan asan lsan ubsan"
        FORCE)
else()
  # Default build type
  set(CMAKE_BUILD_TYPE
      Release
      CACHE
        STRING
        "Choose the type of build: Release Debug tsan asan lsan ubsan"
        FORCE)
endif()

set(PROJECT_NAME "uc15_bimcv_covid19")
project(${PROJECT_NAME})

# ##############################################################################
# Sanitizers
# ##############################################################################
# ThreadSanitizer
set(CMAKE_CXX_FLAGS_TSAN "-fsanitize=thread -g -O1")

# AddressSanitize
set(CMAKE_CXX_FLAGS_ASAN
    "-fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -O1"
)

# LeakSanitizer
set(CMAKE_CXX_FLAGS_LSAN "-fsanitize=leak -fno-omit-frame-pointer -g -O1")

# UndefinedBehaviour
set(CMAKE_CXX_FLAGS_UBSAN "-fsanitize=undefined")

# ##############################################################################
# Sources
# ##############################################################################
add_subdirectory(src)
