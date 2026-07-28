CXX ?= /usr/local/mpi/bin/mpicxx
FFTW_PREFIX ?= /opt/fftw
BUILD_DIR ?= build
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic -fopenmp
CPPFLAGS += -I$(FFTW_PREFIX)/include
LDFLAGS += -L$(FFTW_PREFIX)/lib -Wl,-rpath,$(FFTW_PREFIX)/lib
LDLIBS += -lfftw3_mpi -lfftw3_threads -lfftw3 -lmpi -lm -lpthread -fopenmp

TARGET := $(BUILD_DIR)/fftw_mpi_cpu_benchmark.bin
SOURCE := fftw_mpi_benchmark.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -rf $(BUILD_DIR)
