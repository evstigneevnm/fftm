# Include after target definitions so source files remain the first prerequisite.
FFTM_CONFIG_STAMP := $(BUILD_DIR)/.fftm-$(FFTM_BUILD_SCOPE)-config
FFTM_SIGNATURE_VARIABLES := $(FFTM_CONFIG_VARIABLES) CUDA_HOST_FLAGS \
    SCFD_FLAGS_CA SCFD_FLAGS_NCA SCFD_DEVICE_AWARE DEVICE_AWARE_FLAG \
    HIP_PLATFORM_FLAGS HIP_DEVICE_AWARE HIP_ARCH_FLAGS NVCCFLAGS_NO_DEVICE_DEBUG \
    FFTM_ENABLE_FFTM3D_BACKEND EGGER_ROOT FFTM3D_BENCH_FLAGS FFTM3D_WRAPPED_INCLUDES
.PHONY: fftm-config-force
fftm-config-force:

$(FFTM_CONFIG_STAMP): fftm-config-force | $(BUILD_DIR)
	@set -eu; tmp=$$(mktemp '$@.XXXXXX'); trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	{ $(foreach var,$(FFTM_SIGNATURE_VARIABLES),printf '%s\n' $(call fftm_shell_quote,$(var)=$($(var)));) } > "$$tmp"; \
	if ! cmp -s "$$tmp" '$@'; then mv "$$tmp" '$@'; fi

$(FFTM_BUILD_TARGETS): $(FFTM_CONFIG_STAMP) $(MAKEFILE_LIST)
-include $(wildcard $(BUILD_DIR)/*.d)
