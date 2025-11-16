#ifndef __FFTM_FFT_DIRECTION_H__
#define __FFTM_FFT_DIRECTION_H__


namespace fftm
{

// Define the enum class
enum class direction: std::uint8_t
{ 
    R2C = 1,
    C2R,
    C2CF,
    C2CB
};

}

#endif // __EXTERNAL_WRAP_FFT_DIRECTION_H__