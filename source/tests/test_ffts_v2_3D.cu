#include "detail/cuda_fft_test_environment.h"
#define FFTM_TEST_ENV fftm::test::detail::cuda_fft_test_environment<double>
#define FFTM_EGGER_TESTCASE 2
#define FFTM_VERSIONED_TEST_BINARY "test_ffts_v2_3D.bin"
#include "detail/versioned_ffts_3d_test_driver.h"
