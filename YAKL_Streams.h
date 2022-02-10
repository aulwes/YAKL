#pragma once

#ifdef YAKL_ARCH_CUDA
typedef cudaStream_t    yakl_stream_t;
#elif YAKL_ARCH_HIP
typedef hipStream_t     yakl_stream_t;
#else
typedef int     yakl_stream_t;
#endif

void streamCreate(yakl_stream_t * stream);

void streamDestroy(yakl_stream_t stream);
