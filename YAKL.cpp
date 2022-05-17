
#include "YAKL.h"

namespace yakl {

  #ifdef YAKL_ARCH_SYCL
    sycl::queue sycl_default_stream;
  #endif

  std::mutex yakl_mtx;

  Gator pool;

  void * functorBuffer;

  bool yakl_is_initialized = false;

  // YAKL allocator and deallocator
  std::function<void *( size_t , char const *)> yaklAllocDeviceFunc = [] ( size_t bytes , char const *label ) -> void* {
    yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
    return nullptr;
  };
  std::function<void ( void * , char const *)>  yaklFreeDeviceFunc  = [] ( void *ptr    , char const *label )          {
    yakl_throw("ERROR: attempting memory free before calling yakl::init()");
  };

  // YAKL allocator and deallocator
  std::function<void *( size_t , char const *)> yaklAllocHostFunc = [] ( size_t bytes , char const *label ) -> void* {
    yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
    return nullptr;
  };
  std::function<void ( void * , char const *)>  yaklFreeHostFunc  = [] ( void *ptr    , char const *label )          {
    yakl_throw("ERROR: attempting memory free before calling yakl::init()");
  };

  void streamCreate(yakl_stream_t * stream)
  {
    #ifdef YAKL_ARCH_CUDA
    cudaStreamCreate(stream);
    #elif YAKL_ARCH_HIP
    hipStreamCreate(stream);
    #else
    stream = nullptr;
    #endif
  }

  void streamDestroy(yakl_stream_t stream)
  {
    #ifdef YAKL_ARCH_CUDA
    cudaStreamDestroy(stream);
    #elif YAKL_ARCH_HIP
    hipStreamDestroy(stream);
    #endif
  }

  void eventCreate(yakl_event_t * event)
  {
    #ifdef YAKL_ARCH_CUDA
    cudaEventCreate(event);
    #elif YAKL_ARCH_HIP
    hipEventCreate(event);
    #else
    event = nullptr;
    #endif
  }

  void eventDestroy(yakl_event_t event)
  {
    #ifdef YAKL_ARCH_CUDA
    cudaEventDestroy(event);
    #elif YAKL_ARCH_HIP
    hipEventDestroy(event);
    #endif
  }
}


extern "C" void gatorInit() {
  yakl::init();
}


extern "C" void gatorFinalize() {
  yakl::finalize();
}


// Fortran binding
extern "C" void* gatorAllocate( size_t bytes ) {
  return yakl::yaklAllocDevice( bytes , "");
}


// Fortran binding
extern "C" void gatorDeallocate( void *ptr ) {
  yakl::yaklFreeDevice( ptr , "");
}

