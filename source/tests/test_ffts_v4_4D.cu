#include "detail/cuda_fft_test_environment.h"
#define FFTM_TEST_ENV fftm::test::detail::cuda_fft_test_environment<double>
#define FFTM_EGGER_TESTCASE 4
#define FFTM_VERSIONED_TEST_BINARY "test_ffts_v4_4D.bin"
#include "detail/versioned_ffts_4d_test_driver.h"
