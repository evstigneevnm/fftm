#ifndef __EXTERNAL_WRAP_FFT_DIRECTION_H__
#define __EXTERNAL_WRAP_FFT_DIRECTION_H__


namespace fftm
{

// Define the enum class
enum class direction: std::uint8_t
{ 
    R2C = 1,
    C2R,
    C2C 
};

}

#endif // __EXTERNAL_WRAP_FFT_DIRECTION_H__