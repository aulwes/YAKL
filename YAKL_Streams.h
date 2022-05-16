#pragma once

#ifdef YAKL_ARCH_CUDA
typedef cudaStream_t    yakl_stream_t;
typedef cudaEvent_t    yakl_event_t;
#elif YAKL_ARCH_HIP
typedef hipStream_t     yakl_stream_t;
typedef hipEvent_t    yakl_event_t;
#else
typedef int     yakl_stream_t;
typedef int     yakl_event_t;
#endif

void streamCreate(yakl_stream_t * stream);
void streamDestroy(yakl_stream_t * stream);

void eventCreate(yakl_event_t * event);
void eventDestroy(yakl_event_t * event);
