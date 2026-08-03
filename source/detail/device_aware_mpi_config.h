#ifndef FFTM_DETAIL_DEVICE_AWARE_MPI_CONFIG_H
#define FFTM_DETAIL_DEVICE_AWARE_MPI_CONFIG_H

// SCFD retains the historical CUDA-specific spelling for this build flag.
// Contain that compatibility name here so FFTM production code can express
// the capability independently of the selected CUDA or HIP backend.
#if defined( SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI )
#define FFTM_ENABLE_DEVICE_AWARE_MPI 1
#endif

#endif // FFTM_DETAIL_DEVICE_AWARE_MPI_CONFIG_H
