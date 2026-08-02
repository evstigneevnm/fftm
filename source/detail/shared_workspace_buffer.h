#ifndef __FFTM_DETAIL_SHARED_WORKSPACE_BUFFER_H__
#define __FFTM_DETAIL_SHARED_WORKSPACE_BUFFER_H__

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace fftm
{
namespace detail
{

template <class Memory>
class shared_workspace_buffer
{
public:
    using pointer_t = typename Memory::pointer_type;

    shared_workspace_buffer() = default;

    shared_workspace_buffer( const shared_workspace_buffer & ) = delete;
    shared_workspace_buffer &operator=( const shared_workspace_buffer & ) = delete;

    ~shared_workspace_buffer()
    {
        release_noexcept_();
    }

    void require_size_bytes( std::size_t bytes )
    {
        required_bytes_ = std::max( required_bytes_, bytes );
    }

    std::size_t get_work_size() const
    {
        return required_bytes_;
    }

    std::size_t allocated_size_bytes() const
    {
        return allocated_bytes_;
    }

    pointer_t naive_ptr() const
    {
        return data_;
    }

    void activate()
    {
        if ( required_bytes_ <= allocated_bytes_ )
            return;

        release_storage_();
        if ( required_bytes_ != 0 )
        {
            Memory::malloc( &data_, required_bytes_ );
            allocated_bytes_ = required_bytes_;
        }
    }

    void release()
    {
        release_storage_();
        required_bytes_ = 0;
    }

private:
    void release_storage_()
    {
        if ( data_ == nullptr )
        {
            allocated_bytes_ = 0;
            return;
        }

        Memory::free( data_ );
        data_            = nullptr;
        allocated_bytes_ = 0;
    }

    void release_noexcept_() noexcept
    {
        try
        {
            release();
        }
        catch ( ... )
        {
        }
    }

    std::size_t required_bytes_  = 0;
    std::size_t allocated_bytes_ = 0;
    pointer_t   data_            = nullptr;
};

template <class Memory>
inline const void *workspace_memory_type_token()
{
    static const unsigned char token = 0;
    return &token;
}

class reusable_workspace_resource
{
public:
    virtual ~reusable_workspace_resource() = default;

    void acquire( const void *memory_type_token )
    {
        if ( memory_type_token != memory_type_token_() )
            throw std::logic_error( "FFTM reusable workspace backend does not match the plan backend" );
        if ( leased_ )
            throw std::logic_error( "FFTM reusable workspace is already leased by another plan" );
        leased_ = true;
    }

    void release_lease() noexcept
    {
        leased_ = false;
    }

    bool is_leased() const
    {
        return leased_;
    }

    virtual void require_device_size_bytes( std::size_t bytes ) = 0;
    virtual void require_host_size_bytes( std::size_t bytes ) = 0;
    virtual void require_input_size_bytes( std::size_t bytes ) = 0;
    virtual void require_spectrum_size_bytes( std::size_t bytes ) = 0;
    virtual void activate_device() = 0;
    virtual void activate_host() = 0;
    virtual void activate_input() = 0;
    virtual void activate_spectrum() = 0;
    virtual void *device_ptr() const = 0;
    virtual void *host_ptr() const = 0;
    virtual void *input_ptr() const = 0;
    virtual void *spectrum_ptr() const = 0;
    virtual std::size_t device_capacity_bytes() const = 0;
    virtual std::size_t host_capacity_bytes() const = 0;
    virtual std::size_t input_capacity_bytes() const = 0;
    virtual std::size_t spectrum_capacity_bytes() const = 0;

    std::size_t retained_device_capacity_bytes() const
    {
        return device_capacity_bytes() + input_capacity_bytes() + spectrum_capacity_bytes();
    }

protected:
    void ensure_leased_() const
    {
        if ( !leased_ )
            throw std::logic_error( "FFTM reusable workspace operation requires an active plan lease" );
    }

private:
    virtual const void *memory_type_token_() const = 0;

    bool leased_ = false;
};

template <class DeviceMemory>
class backend_reusable_workspace_resource final : public reusable_workspace_resource
{
public:
    using host_memory_t = typename DeviceMemory::host_memory_type;

    void require_device_size_bytes( std::size_t bytes ) override
    {
        ensure_leased_();
        reject_growth_( device_buffer_.allocated_size_bytes(), bytes, "device" );
        device_buffer_.require_size_bytes( bytes );
    }

    void require_host_size_bytes( std::size_t bytes ) override
    {
        ensure_leased_();
        host_buffer_.require_size_bytes( bytes );
    }

    void require_input_size_bytes( std::size_t bytes ) override
    {
        ensure_leased_();
        reject_growth_( input_buffer_.allocated_size_bytes(), bytes, "input" );
        input_buffer_.require_size_bytes( bytes );
    }

    void require_spectrum_size_bytes( std::size_t bytes ) override
    {
        ensure_leased_();
        reject_growth_( spectrum_buffer_.allocated_size_bytes(), bytes, "spectrum" );
        spectrum_buffer_.require_size_bytes( bytes );
    }

    void activate_device() override
    {
        ensure_leased_();
        device_buffer_.activate();
    }

    void activate_host() override
    {
        ensure_leased_();
        host_buffer_.activate();
    }

    void activate_input() override
    {
        ensure_leased_();
        input_buffer_.activate();
    }

    void activate_spectrum() override
    {
        ensure_leased_();
        spectrum_buffer_.activate();
    }

    void *device_ptr() const override
    {
        ensure_leased_();
        return static_cast<void *>( device_buffer_.naive_ptr() );
    }

    void *host_ptr() const override
    {
        ensure_leased_();
        return static_cast<void *>( host_buffer_.naive_ptr() );
    }

    void *input_ptr() const override
    {
        ensure_leased_();
        return static_cast<void *>( input_buffer_.naive_ptr() );
    }

    void *spectrum_ptr() const override
    {
        ensure_leased_();
        return static_cast<void *>( spectrum_buffer_.naive_ptr() );
    }

    std::size_t device_capacity_bytes() const override
    {
        return device_buffer_.allocated_size_bytes();
    }

    std::size_t host_capacity_bytes() const override
    {
        return host_buffer_.allocated_size_bytes();
    }

    std::size_t input_capacity_bytes() const override
    {
        return input_buffer_.allocated_size_bytes();
    }

    std::size_t spectrum_capacity_bytes() const override
    {
        return spectrum_buffer_.allocated_size_bytes();
    }

private:
    static void reject_growth_( std::size_t capacity, std::size_t requested, const char *kind )
    {
        if ( capacity != 0 && requested > capacity )
        {
            throw std::logic_error(
                std::string( "FFTM reusable " ) + kind +
                " workspace cannot grow after activation; evaluate the largest-memory candidate first"
            );
        }
    }

    const void *memory_type_token_() const override
    {
        return workspace_memory_type_token<DeviceMemory>();
    }

    shared_workspace_buffer<DeviceMemory> device_buffer_;
    shared_workspace_buffer<host_memory_t> host_buffer_;
    shared_workspace_buffer<DeviceMemory> input_buffer_;
    shared_workspace_buffer<DeviceMemory> spectrum_buffer_;
};

template <class DeviceMemory>
inline std::shared_ptr<reusable_workspace_resource> make_reusable_workspace_resource()
{
    return std::make_shared<backend_reusable_workspace_resource<DeviceMemory>>();
}

} // namespace detail
} // namespace fftm

#endif
