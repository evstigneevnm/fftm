#include "detail/cuda_fft_test_environment.h"
#define FFTM_TEST_ENV fftm::test::detail::cuda_fft_test_environment<double>
#define FFTM_EGGER_TESTCASE 4
#define FFTM_VERSIONED_TEST_BINARY "test_fftm_v4_3D.bin"
#include "detail/versioned_fftm_3d_test_driver.h"
