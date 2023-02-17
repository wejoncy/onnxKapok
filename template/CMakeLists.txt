cmake_minimum_required(VERSION 3.20)
project(aot_lib)

option(WITH_DEBUG_EXE "build a exe for debugging" OFF)

# -DANDROID_ARM_NEON=TRUE
# -DANDROID_TOOLCHAIN=clang
#-DCMAKE_TOOLCHAIN_FILE="../android-cmake/android.toolchain.cmake" \
#      -DANDROID_NDK="/home/vagrant/local/android-ndk-r10e"              \
#      -DANDROID_ABI="arm64-v8a"                                         \
#      -DANDROID_NATIVE_API_LEVEL="android-21"                           \
#      -DCMAKE_BUILD_TYPE="Release"   

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#armeabi-v7a，arm64-v8a，x86，x86_64


#-fno-rtti -fno-exceptions -fno-merge-all-constants -fopenmp 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3  -fno-threadsafe-statics -fno-strict-aliasing -fno-omit-frame-pointer -fno-unwind-tables  -fno-stack-protector -pipe -finline-functions -fno-math-errno -fno-trapping-math  -fno-semantic-interposition -fno-plt -ffast-math")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fgraphite-identity -floop-nest-optimize -ftree-loop-distribution  -fipa-pta")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} {EXTRA_CXX_FLAGS}")

# the thirdparty lib, used for simd
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/)
set(MIPP_INCLUDE_DIR {TPT_DIR}/thirdparty/MIPP/install/include)
include_directories(${MIPP_INCLUDE_DIR})
include_directories({TPT_DIR}/template)

set(srcs "code.cc")
set(target "code")


add_library( # Sets the name of the library.
             ${target}
             # Sets the library as a shared library.
             SHARED
             # Provides a relative path to your source file(s).
            ${srcs} )

if(ANDROID_ABI STREQUAL armeabi-v7a)
    set_target_properties(${target} PROPERTIES COMPILE_FLAGS -mfpu=neon)
endif()

if (WITH_DEBUG_EXE)
    add_executable(${target}_exe ${srcs})
endif()