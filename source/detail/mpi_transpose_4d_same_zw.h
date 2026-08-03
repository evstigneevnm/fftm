#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_4D_SAME_ZW_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_4D_SAME_ZW_H__

#include "device_aware_mpi_config.h"
#include "mpi_transpose_4d_common.h"

namespace fftm
{
namespace detail
{

template <
    class ValueType, class Backend, class MPIComm, class Log, class RuntimeAPI>
class mpi_transpose_4d_same_zw
{
public:
    using value_type        = ValueType;
    using backend_t         = Backend;
    using memory_t          = typename backend_t::memory_type;
    using host_memory_t     = typename memory_t::host_memory_type;
    using partition_t       = ::fftm::partition;
    using mpi_comm_t        = scfd::communication::mpi_comm;
    using contiguous_buf_t  = scfd::arrays::array_nd<value_type, 1, memory_t>;
    using host_buf_t        = scfd::arrays::array_nd<value_type, 1, host_memory_t>;
    using mpi_request_t     = scfd::communication::detail::mpi_request;
    using mpi_dtype_t       = scfd::communication::detail::mpi_data_type;
    using idx_t             = scfd::static_vec::vec<int, 4>;
    using range_t           = rect_4d_t<idx_t>;
    using for_each_t        = typename backend_t::template for_each_nd_type<4, int>;
    using runtime_api_t     = RuntimeAPI;
    using profiler_t        = ::fftm::fftm_profiler;
    using profiler_scope_t  = ::fftm::profile_scope<profiler_t>;
    using memory_profiler_t = ::fftm::fftm_memory_profiler;

    mpi_transpose_4d_same_zw( const MPIComm &mpi, const Log &log = Log() ) : mpi_( mpi ), log_( log )
    {
        static_assert(
            std::is_same<memory_t, typename runtime_api_t::memory_type>::value,
            "mpi_transpose_4d_same_zw requires the selected runtime backend memory "
            "type"
        );
        for_each_.block_size = 128;
    }

    ~mpi_transpose_4d_same_zw()
    {
        free_value_type_();
    }

    void set_profiler( profiler_t *profiler )
    {
        profiler_ = profiler;
    }

    void set_stage_timing_callback(
        void *context, mpi_transpose_4d_stage_timer::callback_t callback, const std::string &prefix
    )
    {
        stage_timer_.set( context, callback, prefix );
    }

    void set_memory_profiler( memory_profiler_t *profiler, const std::string &prefix )
    {
        memory_profiler_       = profiler;
        memory_profile_prefix_ = prefix;
        if ( is_inited_ )
        {
            update_memory_profile_();
        }
    }

    void set_peer_paired_p2p_enabled( bool enabled )
    {
        peer_paired_p2p_enabled_ = enabled;
    }

    void set_native_message_layout_enabled( bool enabled )
    {
        native_message_layout_enabled_ = enabled;
    }

    void use_external_work_area()
    {
        use_external_work_area_ = true;
    }

    std::size_t get_work_size_bytes() const
    {
        return bytes_from_elems_( send_buffer_elems_ ) + bytes_from_elems_( recv_buffer_elems_ );
    }

    void set_external_work_area( void *external_work_area )
    {
        if ( !use_external_work_area_ )
        {
            throw std::logic_error(
                "mpi_transpose_4d_same_zw::set_external_work_area: external work area was not enabled."
            );
        }
        external_work_area_ = external_work_area;
        bind_external_work_area_();
        update_memory_profile_();
    }

    void use_external_host_work_area()
    {
        use_external_host_work_area_ = true;
    }

    std::size_t get_host_work_size_bytes() const
    {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
        return 0;
#else
        return bytes_from_elems_( host_send_buffer_elems_ ) + bytes_from_elems_( host_recv_buffer_elems_ );
#endif
    }

    void set_external_host_work_area( void *external_host_work_area )
    {
        if ( !use_external_host_work_area_ )
        {
            throw std::logic_error(
                "mpi_transpose_4d_same_zw::set_external_host_work_area: external host work area was not enabled."
            );
        }
        external_host_work_area_ = external_host_work_area;
        bind_external_host_work_area_();
        update_memory_profile_();
    }

    void init( const partition_t &input_dim, const partition_t &output_dim, int myid_i, int myid_j, int myid_k )
    {
        auto scope  = profile_scope_( "mpi_transpose_4d_same_zw::init" );
        free_value_type_();
        input_dim_  = input_dim;
        output_dim_ = output_dim;
        myid_i_     = myid_i;
        myid_j_     = myid_j;
        myid_k_     = myid_k;

        if ( input_dim_.size_y.size() != 1 )
            throw std::logic_error( "mpi_transpose_4d_same_zw expects the "
                                    "source layout to keep Y undistributed" );
        if ( output_dim_.size_x.size() != 1 )
            throw std::logic_error( "mpi_transpose_4d_same_zw expects the destination layout to "
                                    "keep X undistributed" );
        if ( input_dim_.size_z != output_dim_.size_z || input_dim_.size_w != output_dim_.size_w )
            throw std::logic_error( "mpi_transpose_4d_same_zw requires Z/W "
                                    "ownership to stay unchanged" );
        if ( input_dim_.size_x.size() != output_dim_.size_y.size() )
            throw std::logic_error( "mpi_transpose_4d_same_zw requires matching source-X and "
                                    "destination-Y partition counts" );

        nx_local_  = input_dim_.size_x.at( myid_i_ );
        nx_global_ = output_dim_.size_x.at( 0 );
        ny_local_  = output_dim_.size_y.at( myid_i_ );
        ny_global_ = input_dim_.size_y.at( 0 );
        nz_local_  = input_dim_.size_z.at( myid_j_ );
        nw_local_  = input_dim_.size_w.at( myid_k_ );

        const int color = myid_j_ * static_cast<int>( input_dim_.size_w.size() ) + myid_k_;
        line_comm_      = std::make_unique<mpi_comm_t>( std::move( mpi_.split( color, myid_i_ ) ) );
        line_comm_info_ = line_comm_->info();

        if ( line_comm_info_.num_procs != static_cast<int>( input_dim_.size_x.size() ) )
            throw std::logic_error( "mpi_transpose_4d_same_zw communicator size mismatch" );

        init_value_type_();
        init_layouts_();
        send_buffer_elems_ = max_buffer_elems_;
        recv_buffer_elems_ = max_buffer_elems_;
        host_send_buffer_elems_ = send_buffer_elems_;
        host_recv_buffer_elems_ = recv_buffer_elems_;
        if ( !use_external_work_area_ )
        {
            send_buffer_.init( send_buffer_elems_ );
            recv_buffer_.init( recv_buffer_elems_ );
        }
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        if ( !use_external_host_work_area_ )
        {
            host_send_buffer_.init( host_send_buffer_elems_ );
            host_recv_buffer_.init( host_recv_buffer_elems_ );
        }
#endif
        send_requests_.assign( line_comm_info_.num_procs, mpi_request_t() );
        recv_requests_.assign( line_comm_info_.num_procs, mpi_request_t() );
        is_inited_ = true;
        update_memory_profile_();
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xzwy_to_yzwx( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_zw::transpose_xzwy_to_yzwx/" ) + mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_forward_shapes_( in, out );
        reset_requests_();
        stage_timing_direction_ = "forward";
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
        if ( peer_paired_p2p_enabled_ && mode == mpi_transpose_3d_mode::p2p_waitany )
        {
            forward_p2p_waitany_peer_paired_( in, out );
            return;
        }
#endif
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "pack", [&]() { pack_forward_all_( in ); }
        );

        switch ( mode )
        {
        case mpi_transpose_3d_mode::p2p_waitall:
            forward_p2p_waitall_( out );
            break;
        case mpi_transpose_3d_mode::p2p_waitany:
            forward_p2p_waitany_( out );
            break;
        case mpi_transpose_3d_mode::alltoallv:
            forward_alltoallv_( out );
            break;
        case mpi_transpose_3d_mode::alltoallw:
            forward_alltoallw_( out );
            break;
        }
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xzwy_to_xzwy_native( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_zw::transpose_xzwy_to_xzwy_native/" ) +
            mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_forward_native_shapes_( in, out );
        reset_requests_();
        stage_timing_direction_ = "forward";
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "pack",
            [&]() {
                if ( native_message_layout_enabled_ )
                    pack_forward_native_layout_all_( in );
                else
                    pack_forward_all_( in );
            }
        );

        switch ( mode )
        {
        case mpi_transpose_3d_mode::p2p_waitall:
            forward_p2p_waitall_( out, true, native_message_layout_enabled_ );
            break;
        case mpi_transpose_3d_mode::p2p_waitany:
            forward_p2p_waitany_( out, true, native_message_layout_enabled_ );
            break;
        case mpi_transpose_3d_mode::alltoallv:
            forward_alltoallv_( out, true, native_message_layout_enabled_ );
            break;
        case mpi_transpose_3d_mode::alltoallw:
            forward_alltoallw_( out, true, native_message_layout_enabled_ );
            break;
        }
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_yzwx_to_xzwy( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_zw::transpose_yzwx_to_xzwy/" ) + mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_backward_shapes_( in, out );
        reset_requests_();
        stage_timing_direction_ = "backward";
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
        if ( peer_paired_p2p_enabled_ && mode == mpi_transpose_3d_mode::p2p_waitany )
        {
            backward_p2p_waitany_peer_paired_( in, out );
            return;
        }
#endif
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "pack", [&]() { pack_backward_all_( in ); }
        );

        switch ( mode )
        {
        case mpi_transpose_3d_mode::p2p_waitall:
            backward_p2p_waitall_( out );
            break;
        case mpi_transpose_3d_mode::p2p_waitany:
            backward_p2p_waitany_( out );
            break;
        case mpi_transpose_3d_mode::alltoallv:
            backward_alltoallv_( out );
            break;
        case mpi_transpose_3d_mode::alltoallw:
            backward_alltoallw_( out );
            break;
        }
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xzwy_native_to_xzwy( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_zw::transpose_xzwy_native_to_xzwy/" ) +
            mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_backward_native_shapes_( in, out );
        reset_requests_();
        stage_timing_direction_ = "backward";
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "pack",
            [&]() {
                if ( native_message_layout_enabled_ )
                    pack_backward_native_layout_all_( in );
                else
                    pack_backward_native_all_( in );
            }
        );

        switch ( mode )
        {
        case mpi_transpose_3d_mode::p2p_waitall:
            backward_p2p_waitall_( out, native_message_layout_enabled_ );
            break;
        case mpi_transpose_3d_mode::p2p_waitany:
            backward_p2p_waitany_( out, native_message_layout_enabled_ );
            break;
        case mpi_transpose_3d_mode::alltoallv:
            backward_alltoallv_( out, native_message_layout_enabled_ );
            break;
        case mpi_transpose_3d_mode::alltoallw:
            backward_alltoallw_( out, native_message_layout_enabled_ );
            break;
        }
    }

public:
    template <class Buffer, class ArrayIn>
    struct pack_forward_functor
    {
        Buffer      buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t y_offset;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer( offset + packed_index_4d( idx, ny, nz, nw ) ) =
                in( idx[3], idx[1], idx[2], static_cast<int>( y_offset ) + idx[0] );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_forward_functor
    {
        Buffer      buffer;
        ArrayOut    out;
        std::size_t offset;
        std::size_t x_offset;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( idx[0], idx[1], idx[2], static_cast<int>( x_offset ) + idx[3] ) =
                buffer( offset + packed_index_4d( idx, ny, nz, nw ) );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_forward_native_functor
    {
        Buffer      buffer;
        ArrayOut    out;
        std::size_t offset;
        std::size_t x_offset;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( static_cast<int>( x_offset ) + idx[3], idx[1], idx[2], idx[0] ) =
                buffer( offset + packed_index_4d( idx, ny, nz, nw ) );
        }
    };

    template <class Buffer, class ArrayIn>
    struct pack_backward_functor
    {
        Buffer      buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t x_offset;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer( offset + packed_index_4d( idx, ny, nz, nw ) ) =
                in( idx[0], idx[1], idx[2], static_cast<int>( x_offset ) + idx[3] );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_backward_functor
    {
        Buffer      buffer;
        ArrayOut    out;
        std::size_t offset;
        std::size_t y_offset;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( idx[3], idx[1], idx[2], static_cast<int>( y_offset ) + idx[0] ) =
                buffer( offset + packed_index_4d( idx, ny, nz, nw ) );
        }
    };

    template <class Buffer, class ArrayIn>
    struct pack_backward_native_functor
    {
        Buffer      buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t x_offset;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer( offset + packed_index_4d( idx, ny, nz, nw ) ) =
                in( static_cast<int>( x_offset ) + idx[3], idx[1], idx[2], idx[0] );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_forward_native_layout_functor
    {
        Buffer      buffer;
        ArrayOut    out;
        std::size_t offset;
        std::size_t x_offset;
        std::size_t nx;
        std::size_t nw;
        std::size_t nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( static_cast<int>( x_offset ) + idx[0], idx[2], idx[1], idx[3] ) =
                buffer( offset + packed_index_4d( idx, nx, nw, nz ) );
        }
    };

    template <class Buffer, class ArrayIn>
    struct pack_backward_native_layout_functor
    {
        Buffer      buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t x_offset;
        std::size_t nx;
        std::size_t nw;
        std::size_t nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer( offset + packed_index_4d( idx, nx, nw, nz ) ) =
                in( static_cast<int>( x_offset ) + idx[0], idx[2], idx[1], idx[3] );
        }
    };

private:
    profiler_scope_t profile_scope_( const std::string &name )
    {
        return profiler_scope_t( profiler_, name );
    }

    void ensure_is_inited_() const
    {
        if ( !is_inited_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw::init must be "
                                    "called before transpose" );
        if ( use_external_work_area_ && external_work_area_ == nullptr )
        {
            throw std::logic_error( "mpi_transpose_4d_same_zw: external work area was not bound." );
        }
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        if ( use_external_host_work_area_ && get_host_work_size_bytes() != 0 && external_host_work_area_ == nullptr )
        {
            throw std::logic_error( "mpi_transpose_4d_same_zw: external host work area was not bound." );
        }
#endif
    }

    void update_memory_profile_()
    {
        if ( memory_profiler_ == nullptr || memory_profile_prefix_.empty() )
        {
            return;
        }

        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/send_buffer",
            use_external_work_area_ ? 0
                                    : static_cast<memory_profiler_t::bytes_type>( send_buffer_.size() ) *
                                          static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/recv_buffer",
            use_external_work_area_ ? 0
                                    : static_cast<memory_profiler_t::bytes_type>( recv_buffer_.size() ) *
                                          static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/host_send_buffer",
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
            0
#else
            use_external_host_work_area_ ? 0
                                         : static_cast<memory_profiler_t::bytes_type>( host_send_buffer_.size() ) *
                                               static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
#endif
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/host_recv_buffer",
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
            0
#else
            use_external_host_work_area_ ? 0
                                         : static_cast<memory_profiler_t::bytes_type>( host_recv_buffer_.size() ) *
                                               static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
#endif
        );
    }

    void bind_external_work_area_()
    {
        if ( external_work_area_ == nullptr )
        {
            return;
        }
        char *raw = static_cast<char *>( external_work_area_ );
        send_buffer_.init_by_raw_data( reinterpret_cast<value_type *>( raw ), send_buffer_elems_ );
        recv_buffer_.init_by_raw_data(
            reinterpret_cast<value_type *>( raw + bytes_from_elems_( send_buffer_elems_ ) ), recv_buffer_elems_
        );
    }

    void bind_external_host_work_area_()
    {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
        return;
#else
        if ( external_host_work_area_ == nullptr )
        {
            return;
        }
        char *raw = static_cast<char *>( external_host_work_area_ );
        host_send_buffer_.init_by_raw_data( reinterpret_cast<value_type *>( raw ), host_send_buffer_elems_ );
        host_recv_buffer_.init_by_raw_data(
            reinterpret_cast<value_type *>( raw + bytes_from_elems_( host_send_buffer_elems_ ) ),
            host_recv_buffer_elems_
        );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void verify_forward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( in_size[3] ) != ny_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw forward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != ny_local_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( out_size[3] ) != nx_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw forward output shape mismatch" );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_forward_native_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( in_size[3] ) != ny_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw native forward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != nx_global_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( out_size[3] ) != ny_local_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw native forward output shape mismatch" );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_backward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != ny_local_ ||
             static_cast<std::size_t>( in_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( in_size[3] ) != nx_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw backward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( out_size[3] ) != ny_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw backward output shape mismatch" );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_backward_native_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_global_ ||
             static_cast<std::size_t>( in_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( in_size[3] ) != ny_local_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw native backward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( out_size[3] ) != ny_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw native backward output shape mismatch" );
    }

    void reset_requests_()
    {
        std::fill( send_requests_.begin(), send_requests_.end(), mpi_request_t() );
        std::fill( recv_requests_.begin(), recv_requests_.end(), mpi_request_t() );
    }

    std::size_t bytes_from_elems_( std::size_t elems ) const
    {
        return elems * sizeof( value_type );
    }

    void copy_send_buffer_to_host_()
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        runtime_api_t::memcpy(
            host_send_buffer_.raw_ptr(), send_buffer_.raw_ptr(), bytes_from_elems_( send_buffer_elems_ ),
            runtime_api_t::device_to_host_kind()
        );
#endif
    }

    void copy_recv_buffer_from_host_()
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr(), host_recv_buffer_.raw_ptr(), bytes_from_elems_( recv_buffer_elems_ ),
            runtime_api_t::host_to_device_kind()
        );
#endif
    }

    void copy_host_forward_chunk_to_device_( int source_i )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr() + forward_recv_offsets_[source_i],
            host_recv_buffer_.raw_ptr() + forward_recv_offsets_[source_i],
            bytes_from_elems_( forward_recv_chunk_elems_( source_i ) ), runtime_api_t::host_to_device_kind()
        );
#endif
    }

    void copy_host_backward_chunk_to_device_( int source_i )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr() + backward_recv_offsets_[source_i],
            host_recv_buffer_.raw_ptr() + backward_recv_offsets_[source_i],
            bytes_from_elems_( backward_recv_chunk_elems_( source_i ) ), runtime_api_t::host_to_device_kind()
        );
#endif
    }

    void init_value_type_()
    {
        mpi_value_type_ = scfd::communication::detail::type_contiguous(
            detail::mpi_int_cast( sizeof( value_type ), "mpi_transpose_4d_same_zw value type extent" ),
            scfd::communication::detail::mpi_data_type_trait<char>::get()
        );
        scfd::communication::detail::type_commit( mpi_value_type_ );
        mpi_value_type_inited_ = true;
    }

    void free_value_type_()
    {
        if ( mpi_value_type_inited_ )
        {
            scfd::communication::detail::type_free( mpi_value_type_ );
            mpi_value_type_inited_ = false;
        }
    }

    std::size_t forward_send_chunk_elems_( int target_i ) const
    {
        return output_dim_.size_y[target_i] * nz_local_ * nw_local_ * nx_local_;
    }

    std::size_t forward_recv_chunk_elems_( int source_i ) const
    {
        return ny_local_ * nz_local_ * nw_local_ * input_dim_.size_x[source_i];
    }

    std::size_t backward_send_chunk_elems_( int target_i ) const
    {
        return ny_local_ * nz_local_ * nw_local_ * input_dim_.size_x[target_i];
    }

    std::size_t backward_recv_chunk_elems_( int source_i ) const
    {
        return output_dim_.size_y[source_i] * nz_local_ * nw_local_ * nx_local_;
    }

    void init_layouts_()
    {
        const int comm_size = line_comm_info_.num_procs;

        forward_send_offsets_.resize( comm_size );
        forward_recv_offsets_.resize( comm_size );
        forward_sendcounts_.resize( comm_size );
        forward_sdispls_.resize( comm_size );
        forward_recvcounts_.resize( comm_size );
        forward_rdispls_.resize( comm_size );
        forward_sendcounts_w_.resize( comm_size );
        forward_sdispls_w_.resize( comm_size );
        forward_sendtypes_w_.assign( comm_size, mpi_value_type_ );
        forward_recvcounts_w_.resize( comm_size );
        forward_rdispls_w_.resize( comm_size );
        forward_recvtypes_w_.assign( comm_size, mpi_value_type_ );

        backward_send_offsets_.resize( comm_size );
        backward_recv_offsets_.resize( comm_size );
        backward_sendcounts_.resize( comm_size );
        backward_sdispls_.resize( comm_size );
        backward_recvcounts_.resize( comm_size );
        backward_rdispls_.resize( comm_size );
        backward_sendcounts_w_.resize( comm_size );
        backward_sdispls_w_.resize( comm_size );
        backward_sendtypes_w_.assign( comm_size, mpi_value_type_ );
        backward_recvcounts_w_.resize( comm_size );
        backward_rdispls_w_.resize( comm_size );
        backward_recvtypes_w_.assign( comm_size, mpi_value_type_ );

        std::size_t packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            forward_send_offsets_[p] = packed_offset;
            packed_offset += forward_send_chunk_elems_( p );
        }
        max_buffer_elems_ = packed_offset;

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            forward_recv_offsets_[p] = packed_offset;
            packed_offset += forward_recv_chunk_elems_( p );
        }
        max_buffer_elems_ = std::max( max_buffer_elems_, packed_offset );

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            backward_send_offsets_[p] = packed_offset;
            packed_offset += backward_send_chunk_elems_( p );
        }
        max_buffer_elems_ = std::max( max_buffer_elems_, packed_offset );

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            backward_recv_offsets_[p] = packed_offset;
            packed_offset += backward_recv_chunk_elems_( p );
        }
        max_buffer_elems_ = std::max( max_buffer_elems_, packed_offset );

        for ( int p = 0; p < comm_size; ++p )
        {
            forward_sendcounts_[p] =
                detail::mpi_int_cast( forward_send_chunk_elems_( p ), "same_zw forward sendcount" );
            forward_sdispls_[p] = detail::mpi_int_cast( forward_send_offsets_[p], "same_zw forward sdispl" );
            forward_recvcounts_[p] =
                detail::mpi_int_cast( forward_recv_chunk_elems_( p ), "same_zw forward recvcount" );
            forward_rdispls_[p] = detail::mpi_int_cast( forward_recv_offsets_[p], "same_zw forward rdispl" );
            forward_sendcounts_w_[p] = forward_sendcounts_[p];
            forward_recvcounts_w_[p] = forward_recvcounts_[p];
            forward_sdispls_w_[p]    = 0;
            forward_rdispls_w_[p]    = 0;

            backward_sendcounts_[p] =
                detail::mpi_int_cast( backward_send_chunk_elems_( p ), "same_zw backward sendcount" );
            backward_sdispls_[p] = detail::mpi_int_cast( backward_send_offsets_[p], "same_zw backward sdispl" );
            backward_recvcounts_[p] =
                detail::mpi_int_cast( backward_recv_chunk_elems_( p ), "same_zw backward recvcount" );
            backward_rdispls_[p] = detail::mpi_int_cast( backward_recv_offsets_[p], "same_zw backward rdispl" );
            backward_sendcounts_w_[p] = backward_sendcounts_[p];
            backward_recvcounts_w_[p] = backward_recvcounts_[p];
            backward_sdispls_w_[p]    = 0;
            backward_rdispls_w_[p]    = 0;
        }
        forward_alltoallw_layout_ready_  = false;
        backward_alltoallw_layout_ready_ = false;
    }

    void ensure_forward_alltoallw_layout_()
    {
        if ( forward_alltoallw_layout_ready_ )
            return;
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            forward_sdispls_w_[p] =
                detail::mpi_int_cast( bytes_from_elems_( forward_send_offsets_[p] ), "same_zw forward sdispl_w" );
            forward_rdispls_w_[p] =
                detail::mpi_int_cast( bytes_from_elems_( forward_recv_offsets_[p] ), "same_zw forward rdispl_w" );
        }
        forward_alltoallw_layout_ready_ = true;
    }

    void ensure_backward_alltoallw_layout_()
    {
        if ( backward_alltoallw_layout_ready_ )
            return;
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            backward_sdispls_w_[p] =
                detail::mpi_int_cast( bytes_from_elems_( backward_send_offsets_[p] ), "same_zw backward sdispl_w" );
            backward_rdispls_w_[p] =
                detail::mpi_int_cast( bytes_from_elems_( backward_recv_offsets_[p] ), "same_zw backward rdispl_w" );
        }
        backward_alltoallw_layout_ready_ = true;
    }

    template <class ArrayIn>
    void launch_pack_forward_chunk_( const ArrayIn &in, int p )
    {
        for_each_(
            pack_forward_functor<contiguous_buf_t, ArrayIn>{
                send_buffer_, in, forward_send_offsets_[p], output_dim_.start_y[p], output_dim_.size_y[p],
                nz_local_, nw_local_ },
            make_range_4d<idx_t>( output_dim_.size_y[p], nz_local_, nw_local_, nx_local_ )
        );
    }

    template <class ArrayIn>
    void pack_forward_chunk_( const ArrayIn &in, int p )
    {
        auto scope = profile_scope_( "pack_forward_chunk" );
        launch_pack_forward_chunk_( in, p );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_forward_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_forward_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            pack_forward_chunk_( in, p );
    }

    template <class ArrayIn>
    void pack_forward_native_layout_chunk_( const ArrayIn &in, int p )
    {
        auto scope = profile_scope_( "pack_forward_native_layout_chunk" );
        const std::size_t block_elems = nx_local_ * nw_local_ * nz_local_;
        runtime_api_t::memcpy(
            send_buffer_.raw_ptr() + forward_send_offsets_[p],
            in.raw_ptr() + output_dim_.start_y[p] * block_elems,
            bytes_from_elems_( forward_send_chunk_elems_( p ) ), runtime_api_t::device_to_device_kind()
        );
    }

    template <class ArrayIn>
    void pack_forward_native_layout_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_forward_native_layout_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            pack_forward_native_layout_chunk_( in, p );
    }

    template <class ArrayOut>
    void unpack_forward_chunk_( int source_i, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_forward_chunk" );
        for_each_(
            unpack_forward_functor<contiguous_buf_t, ArrayOut>{
                recv_buffer_, out, forward_recv_offsets_[source_i], input_dim_.start_x[source_i], ny_local_, nz_local_,
                nw_local_ },
            make_range_4d<idx_t>( ny_local_, nz_local_, nw_local_, input_dim_.size_x[source_i] )
        );
        for_each_.wait();
    }

    template <class ArrayOut>
    void unpack_forward_native_chunk_( int source_i, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_forward_native_chunk" );
        for_each_(
            unpack_forward_native_functor<contiguous_buf_t, ArrayOut>{
                recv_buffer_, out, forward_recv_offsets_[source_i], input_dim_.start_x[source_i], ny_local_,
                nz_local_, nw_local_ },
            make_range_4d<idx_t>( ny_local_, nz_local_, nw_local_, input_dim_.size_x[source_i] )
        );
        for_each_.wait();
    }

    template <class ArrayOut>
    void unpack_forward_native_layout_chunk_( int source_i, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_forward_native_layout_chunk" );
        const std::size_t nx_chunk = input_dim_.size_x[source_i];
        for_each_(
            unpack_forward_native_layout_functor<contiguous_buf_t, ArrayOut>{
                recv_buffer_, out, forward_recv_offsets_[source_i], input_dim_.start_x[source_i], nx_chunk,
                nw_local_, nz_local_ },
            make_range_4d<idx_t>( nx_chunk, nw_local_, nz_local_, ny_local_ )
        );
        for_each_.wait();
    }

    template <class ArrayOut>
    void unpack_forward_chunk_( int source_i, ArrayOut &out, bool native_output )
    {
        unpack_forward_chunk_( source_i, out, native_output, false );
    }

    template <class ArrayOut>
    void unpack_forward_chunk_( int source_i, ArrayOut &out, bool native_output, bool native_message_layout )
    {
        if ( native_output )
        {
            if ( native_message_layout )
                unpack_forward_native_layout_chunk_( source_i, out );
            else
                unpack_forward_native_chunk_( source_i, out );
        }
        else
        {
            unpack_forward_chunk_( source_i, out );
        }
    }

    template <class ArrayIn>
    void launch_pack_backward_chunk_( const ArrayIn &in, int p )
    {
        for_each_(
            pack_backward_functor<contiguous_buf_t, ArrayIn>{
                send_buffer_, in, backward_send_offsets_[p], input_dim_.start_x[p], ny_local_, nz_local_,
                nw_local_ },
            make_range_4d<idx_t>( ny_local_, nz_local_, nw_local_, input_dim_.size_x[p] )
        );
    }

    template <class ArrayIn>
    void pack_backward_chunk_( const ArrayIn &in, int p )
    {
        auto scope = profile_scope_( "pack_backward_chunk" );
        launch_pack_backward_chunk_( in, p );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_backward_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_backward_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            pack_backward_chunk_( in, p );
    }

    template <class ArrayIn>
    void launch_pack_backward_native_chunk_( const ArrayIn &in, int p )
    {
        for_each_(
            pack_backward_native_functor<contiguous_buf_t, ArrayIn>{
                send_buffer_, in, backward_send_offsets_[p], input_dim_.start_x[p], ny_local_, nz_local_,
                nw_local_ },
            make_range_4d<idx_t>( ny_local_, nz_local_, nw_local_, input_dim_.size_x[p] )
        );
    }

    template <class ArrayIn>
    void pack_backward_native_chunk_( const ArrayIn &in, int p )
    {
        auto scope = profile_scope_( "pack_backward_native_chunk" );
        launch_pack_backward_native_chunk_( in, p );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_backward_native_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_backward_native_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            pack_backward_native_chunk_( in, p );
    }

    template <class ArrayIn>
    void launch_pack_backward_native_layout_chunk_( const ArrayIn &in, int p )
    {
        const std::size_t nx_chunk = input_dim_.size_x[p];
        for_each_(
            pack_backward_native_layout_functor<contiguous_buf_t, ArrayIn>{
                send_buffer_, in, backward_send_offsets_[p], input_dim_.start_x[p], nx_chunk, nw_local_,
                nz_local_ },
            make_range_4d<idx_t>( nx_chunk, nw_local_, nz_local_, ny_local_ )
        );
    }

    template <class ArrayIn>
    void pack_backward_native_layout_chunk_( const ArrayIn &in, int p )
    {
        auto scope = profile_scope_( "pack_backward_native_layout_chunk" );
        launch_pack_backward_native_layout_chunk_( in, p );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_backward_native_layout_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_backward_native_layout_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            pack_backward_native_layout_chunk_( in, p );
    }

    template <class ArrayOut>
    void unpack_backward_chunk_( int source_i, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_backward_chunk" );
        for_each_(
            unpack_backward_functor<contiguous_buf_t, ArrayOut>{
                recv_buffer_, out, backward_recv_offsets_[source_i], output_dim_.start_y[source_i],
                output_dim_.size_y[source_i], nz_local_, nw_local_ },
            make_range_4d<idx_t>( output_dim_.size_y[source_i], nz_local_, nw_local_, nx_local_ )
        );
        for_each_.wait();
    }

    template <class ArrayOut>
    void unpack_backward_native_layout_chunk_( int source_i, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_backward_native_layout_chunk" );
        const std::size_t block_elems = nx_local_ * nw_local_ * nz_local_;
        runtime_api_t::memcpy(
            out.raw_ptr() + output_dim_.start_y[source_i] * block_elems,
            recv_buffer_.raw_ptr() + backward_recv_offsets_[source_i],
            bytes_from_elems_( backward_recv_chunk_elems_( source_i ) ), runtime_api_t::device_to_device_kind()
        );
    }

    template <class ArrayOut>
    void unpack_backward_chunk_( int source_i, ArrayOut &out, bool native_message_layout )
    {
        if ( native_message_layout )
            unpack_backward_native_layout_chunk_( source_i, out );
        else
            unpack_backward_chunk_( source_i, out );
    }

#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
    template <class ArrayIn, class ArrayOut>
    void forward_p2p_waitany_peer_paired_( const ArrayIn &in, ArrayOut &out )
    {
        auto      scope     = profile_scope_( "forward_p2p_waitany_peer_paired" );
        const int comm_size = line_comm_info_.num_procs;

        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "post_recv", [&]() {
                for ( int p = 0; p < comm_size; ++p )
                {
                    if ( p == myid_i_ )
                        continue;
                    line_comm_info_.irecv(
                        recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                        mpi_value_type_, p, p, recv_requests_[p]
                    );
                }
            }
        );

        int completed_remote = 0;
        auto drain_ready     = [&]() {
            while ( completed_remote < comm_size - 1 )
            {
                int ready = 0;
                int p     = MPI_UNDEFINED;
                time_mpi_transpose_4d_substage<runtime_api_t>(
                    stage_timer_, stage_timing_direction_, "test_recv_any", [&]() {
                        p = line_comm_info_.testany( comm_size, recv_requests_.data(), &ready );
                    }
                );
                if ( !ready || p == MPI_UNDEFINED )
                    break;
                time_mpi_transpose_4d_substage<runtime_api_t>(
                    stage_timer_, stage_timing_direction_, "unpack", [&]() { unpack_forward_chunk_( p, out ); }
                );
                ++completed_remote;
            }
        };

        for ( int p = 0; p < comm_size; ++p )
        {
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "pack", [&]() { pack_forward_chunk_( in, p ); }
            );
            if ( p == myid_i_ )
            {
                time_mpi_transpose_4d_substage<runtime_api_t>(
                    stage_timer_, stage_timing_direction_, "self_copy", [&]() {
                        runtime_api_t::memcpy(
                            recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_i_],
                            send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
                            bytes_from_elems_( forward_recv_chunk_elems_( myid_i_ ) ),
                            runtime_api_t::device_to_device_kind()
                        );
                    }
                );
                time_mpi_transpose_4d_substage<runtime_api_t>(
                    stage_timer_, stage_timing_direction_, "unpack", [&]() { unpack_forward_chunk_( myid_i_, out ); }
                );
            }
            else
            {
                time_mpi_transpose_4d_substage<runtime_api_t>(
                    stage_timer_, stage_timing_direction_, "post_send_peer", [&]() {
                        line_comm_info_.isend(
                            send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                            mpi_value_type_, p, myid_i_, send_requests_[p]
                        );
                    }
                );
            }
            drain_ready();
        }

        while ( completed_remote < comm_size - 1 )
        {
            int p = MPI_UNDEFINED;
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "wait_recv_any", [&]() {
                    p = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                }
            );
            if ( p == MPI_UNDEFINED )
                break;
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "unpack", [&]() { unpack_forward_chunk_( p, out ); }
            );
            ++completed_remote;
        }

        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "wait_send", [&]() {
                line_comm_info_.waitall( comm_size, send_requests_.data() );
            }
        );
    }

    template <class ArrayIn, class ArrayOut>
    void backward_p2p_waitany_peer_paired_( const ArrayIn &in, ArrayOut &out )
    {
        auto      scope     = profile_scope_( "backward_p2p_waitany_peer_paired" );
        const int comm_size = line_comm_info_.num_procs;

        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "post_recv", [&]() {
                for ( int p = 0; p < comm_size; ++p )
                {
                    if ( p == myid_i_ )
                        continue;
                    line_comm_info_.irecv(
                        recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                        mpi_value_type_, p, myid_i_, recv_requests_[p]
                    );
                }
            }
        );

        int completed_remote = 0;
        auto drain_ready     = [&]() {
            while ( completed_remote < comm_size - 1 )
            {
                int ready = 0;
                int p     = MPI_UNDEFINED;
                time_mpi_transpose_4d_substage<runtime_api_t>(
                    stage_timer_, stage_timing_direction_, "test_recv_any", [&]() {
                        p = line_comm_info_.testany( comm_size, recv_requests_.data(), &ready );
                    }
                );
                if ( !ready || p == MPI_UNDEFINED )
                    break;
                time_mpi_transpose_4d_substage<runtime_api_t>(
                    stage_timer_, stage_timing_direction_, "unpack", [&]() { unpack_backward_chunk_( p, out ); }
                );
                ++completed_remote;
            }
        };

        for ( int p = 0; p < comm_size; ++p )
        {
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "pack", [&]() { pack_backward_chunk_( in, p ); }
            );
            if ( p == myid_i_ )
            {
                time_mpi_transpose_4d_substage<runtime_api_t>(
                    stage_timer_, stage_timing_direction_, "self_copy", [&]() {
                        runtime_api_t::memcpy(
                            recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_i_],
                            send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
                            bytes_from_elems_( backward_recv_chunk_elems_( myid_i_ ) ),
                            runtime_api_t::device_to_device_kind()
                        );
                    }
                );
                time_mpi_transpose_4d_substage<runtime_api_t>(
                    stage_timer_, stage_timing_direction_, "unpack", [&]() { unpack_backward_chunk_( myid_i_, out ); }
                );
            }
            else
            {
                time_mpi_transpose_4d_substage<runtime_api_t>(
                    stage_timer_, stage_timing_direction_, "post_send_peer", [&]() {
                        line_comm_info_.isend(
                            send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                            mpi_value_type_, p, p, send_requests_[p]
                        );
                    }
                );
            }
            drain_ready();
        }

        while ( completed_remote < comm_size - 1 )
        {
            int p = MPI_UNDEFINED;
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "wait_recv_any", [&]() {
                    p = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                }
            );
            if ( p == MPI_UNDEFINED )
                break;
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "unpack", [&]() { unpack_backward_chunk_( p, out ); }
            );
            ++completed_remote;
        }

        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "wait_send", [&]() {
                line_comm_info_.waitall( comm_size, send_requests_.data() );
            }
        );
    }
#endif

    template <class ArrayOut>
    void forward_p2p_waitall_( ArrayOut &out )
    {
        forward_p2p_waitall_( out, false );
    }

    template <class ArrayOut>
    void forward_p2p_waitall_( ArrayOut &out, bool native_output )
    {
        forward_p2p_waitall_( out, native_output, false );
    }

    template <class ArrayOut>
    void forward_p2p_waitall_( ArrayOut &out, bool native_output, bool native_message_layout )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto      scope     = profile_scope_( "forward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                    mpi_value_type_, p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                    mpi_value_type_, p, myid_i_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_i_],
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
                bytes_from_elems_( forward_recv_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p != myid_i_ )
                {
                    copy_host_forward_chunk_to_device_( p );
                }
            }
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
                unpack_forward_chunk_( p, out, native_output, native_message_layout );
        }
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#else
        auto      scope     = profile_scope_( "forward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                    mpi_value_type_, p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                    mpi_value_type_, p, myid_i_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_i_],
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
                bytes_from_elems_( forward_recv_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
                unpack_forward_chunk_( p, out, native_output, native_message_layout );
        }
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#endif
    }

    template <class ArrayOut>
    void forward_p2p_waitany_( ArrayOut &out )
    {
        forward_p2p_waitany_( out, false );
    }

    template <class ArrayOut>
    void forward_p2p_waitany_( ArrayOut &out, bool native_output )
    {
        forward_p2p_waitany_( out, native_output, false );
    }

    template <class ArrayOut>
    void forward_p2p_waitany_( ArrayOut &out, bool native_output, bool native_message_layout )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto      scope     = profile_scope_( "forward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                    mpi_value_type_, p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                    mpi_value_type_, p, myid_i_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_i_],
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
                bytes_from_elems_( forward_recv_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_forward_chunk_( myid_i_, out, native_output, native_message_layout );
        }

        {
            int completed = 0;
            while ( completed < comm_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "wait_recv_any" );
                    p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                {
                    auto phase = profile_scope_( "stage_recv_chunk_to_device" );
                    copy_host_forward_chunk_to_device_( p );
                }
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_forward_chunk_( p, out, native_output, native_message_layout );
                }
                completed++;
            }
        }

        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#else
        auto      scope     = profile_scope_( "forward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                    mpi_value_type_, p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                    mpi_value_type_, p, myid_i_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_i_],
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
                bytes_from_elems_( forward_recv_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_forward_chunk_( myid_i_, out, native_output, native_message_layout );
        }

        {
            int completed = 0;
            while ( completed < comm_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "wait_recv_any" );
                    p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_forward_chunk_( p, out, native_output, native_message_layout );
                }
                completed++;
            }
        }

        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#endif
    }

    template <class ArrayOut>
    void forward_alltoallv_( ArrayOut &out )
    {
        forward_alltoallv_( out, false );
    }

    template <class ArrayOut>
    void forward_alltoallv_( ArrayOut &out, bool native_output )
    {
        forward_alltoallv_( out, native_output, false );
    }

    template <class ArrayOut>
    void forward_alltoallv_( ArrayOut &out, bool native_output, bool native_message_layout )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto scope = profile_scope_( "forward_alltoallv" );
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            line_comm_info_.alltoallv(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), forward_sendcounts_.data(),
                forward_sdispls_.data(), mpi_value_type_, static_cast<void *>( host_recv_buffer_.raw_ptr() ),
                forward_recvcounts_.data(), forward_rdispls_.data(), mpi_value_type_
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            copy_recv_buffer_from_host_();
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_forward_chunk_( p, out, native_output, native_message_layout );
        }
#else
        auto scope = profile_scope_( "forward_alltoallv" );
        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "mpi_alltoallv", [&]() {
                    line_comm_info_.alltoallv(
                        static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_.data(),
                        forward_sdispls_.data(), mpi_value_type_, static_cast<void *>( recv_buffer_.raw_ptr() ),
                        forward_recvcounts_.data(), forward_rdispls_.data(), mpi_value_type_
                    );
                }
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "unpack", [&]() {
                    for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                        unpack_forward_chunk_( p, out, native_output, native_message_layout );
                }
            );
        }
#endif
    }

    template <class ArrayOut>
    void forward_alltoallw_( ArrayOut &out )
    {
        forward_alltoallw_( out, false );
    }

    template <class ArrayOut>
    void forward_alltoallw_( ArrayOut &out, bool native_output )
    {
        forward_alltoallw_( out, native_output, false );
    }

    template <class ArrayOut>
    void forward_alltoallw_( ArrayOut &out, bool native_output, bool native_message_layout )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto scope = profile_scope_( "forward_alltoallw" );
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            ensure_forward_alltoallw_layout_();
        }
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), forward_sendcounts_w_.data(),
                forward_sdispls_w_.data(), forward_sendtypes_w_.data(),
                static_cast<void *>( host_recv_buffer_.raw_ptr() ), forward_recvcounts_w_.data(),
                forward_rdispls_w_.data(), forward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            copy_recv_buffer_from_host_();
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_forward_chunk_( p, out, native_output, native_message_layout );
        }
#else
        auto scope = profile_scope_( "forward_alltoallw" );
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            ensure_forward_alltoallw_layout_();
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "mpi_alltoallw", [&]() {
                    line_comm_info_.alltoallw(
                        static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_w_.data(),
                        forward_sdispls_w_.data(), forward_sendtypes_w_.data(),
                        static_cast<void *>( recv_buffer_.raw_ptr() ), forward_recvcounts_w_.data(),
                        forward_rdispls_w_.data(), forward_recvtypes_w_.data()
                    );
                }
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "unpack", [&]() {
                    for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                        unpack_forward_chunk_( p, out, native_output, native_message_layout );
                }
            );
        }
#endif
    }

    template <class ArrayOut>
    void backward_p2p_waitall_( ArrayOut &out )
    {
        backward_p2p_waitall_( out, false );
    }

    template <class ArrayOut>
    void backward_p2p_waitall_( ArrayOut &out, bool native_message_layout )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto      scope     = profile_scope_( "backward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                    mpi_value_type_, p, myid_i_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_i_],
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
                bytes_from_elems_( backward_recv_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p != myid_i_ )
                {
                    copy_host_backward_chunk_to_device_( p );
                }
            }
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
                unpack_backward_chunk_( p, out, native_message_layout );
        }
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#else
        auto      scope     = profile_scope_( "backward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                    mpi_value_type_, p, myid_i_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_i_],
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
                bytes_from_elems_( backward_recv_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
                unpack_backward_chunk_( p, out );
        }
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#endif
    }

    template <class ArrayOut>
    void backward_p2p_waitany_( ArrayOut &out )
    {
        backward_p2p_waitany_( out, false );
    }

    template <class ArrayOut>
    void backward_p2p_waitany_( ArrayOut &out, bool native_message_layout )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto      scope     = profile_scope_( "backward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                    mpi_value_type_, p, myid_i_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_i_],
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
                bytes_from_elems_( backward_recv_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_backward_chunk_( myid_i_, out, native_message_layout );
        }

        {
            int completed = 0;
            while ( completed < comm_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "wait_recv_any" );
                    p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                {
                    auto phase = profile_scope_( "stage_recv_chunk_to_device" );
                    copy_host_backward_chunk_to_device_( p );
                }
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_backward_chunk_( p, out, native_message_layout );
                }
                completed++;
            }
        }

        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#else
        auto      scope     = profile_scope_( "backward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                    mpi_value_type_, p, myid_i_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_i_],
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
                bytes_from_elems_( backward_recv_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_backward_chunk_( myid_i_, out, native_message_layout );
        }

        {
            int completed = 0;
            while ( completed < comm_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "wait_recv_any" );
                    p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_backward_chunk_( p, out, native_message_layout );
                }
                completed++;
            }
        }

        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#endif
    }

    template <class ArrayOut>
    void backward_alltoallv_( ArrayOut &out )
    {
        backward_alltoallv_( out, false );
    }

    template <class ArrayOut>
    void backward_alltoallv_( ArrayOut &out, bool native_message_layout )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto scope = profile_scope_( "backward_alltoallv" );
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            line_comm_info_.alltoallv(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), backward_sendcounts_.data(),
                backward_sdispls_.data(), mpi_value_type_, static_cast<void *>( host_recv_buffer_.raw_ptr() ),
                backward_recvcounts_.data(), backward_rdispls_.data(), mpi_value_type_
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            copy_recv_buffer_from_host_();
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_backward_chunk_( p, out, native_message_layout );
        }
#else
        auto scope = profile_scope_( "backward_alltoallv" );
        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "mpi_alltoallv", [&]() {
                    line_comm_info_.alltoallv(
                        static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_.data(),
                        backward_sdispls_.data(), mpi_value_type_, static_cast<void *>( recv_buffer_.raw_ptr() ),
                        backward_recvcounts_.data(), backward_rdispls_.data(), mpi_value_type_
                    );
                }
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "unpack", [&]() {
                    for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                        unpack_backward_chunk_( p, out, native_message_layout );
                }
            );
        }
#endif
    }

    template <class ArrayOut>
    void backward_alltoallw_( ArrayOut &out )
    {
        backward_alltoallw_( out, false );
    }

    template <class ArrayOut>
    void backward_alltoallw_( ArrayOut &out, bool native_message_layout )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto scope = profile_scope_( "backward_alltoallw" );
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            ensure_backward_alltoallw_layout_();
        }
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), backward_sendcounts_w_.data(),
                backward_sdispls_w_.data(), backward_sendtypes_w_.data(),
                static_cast<void *>( host_recv_buffer_.raw_ptr() ), backward_recvcounts_w_.data(),
                backward_rdispls_w_.data(), backward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            copy_recv_buffer_from_host_();
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_backward_chunk_( p, out, native_message_layout );
        }
#else
        auto scope = profile_scope_( "backward_alltoallw" );
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            ensure_backward_alltoallw_layout_();
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "mpi_alltoallw", [&]() {
                    line_comm_info_.alltoallw(
                        static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_w_.data(),
                        backward_sdispls_w_.data(), backward_sendtypes_w_.data(),
                        static_cast<void *>( recv_buffer_.raw_ptr() ), backward_recvcounts_w_.data(),
                        backward_rdispls_w_.data(), backward_recvtypes_w_.data()
                    );
                }
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "unpack", [&]() {
                    for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                        unpack_backward_chunk_( p, out, native_message_layout );
                }
            );
        }
#endif
    }

private:
    MPIComm            mpi_;
    Log                log_;
    profiler_t        *profiler_        = nullptr;
    memory_profiler_t *memory_profiler_ = nullptr;
    std::string        memory_profile_prefix_;
    mpi_transpose_4d_stage_timer stage_timer_;
    const char                  *stage_timing_direction_ = "unknown";

    bool is_inited_ = false;
    int  myid_i_    = 0;
    int  myid_j_    = 0;
    int  myid_k_    = 0;

    std::size_t nx_local_  = 0;
    std::size_t nx_global_ = 0;
    std::size_t ny_local_  = 0;
    std::size_t ny_global_ = 0;
    std::size_t nz_local_  = 0;
    std::size_t nw_local_  = 0;

    partition_t input_dim_;
    partition_t output_dim_;

    std::unique_ptr<mpi_comm_t>        line_comm_;
    scfd::communication::mpi_comm_info line_comm_info_;
    contiguous_buf_t                   send_buffer_;
    contiguous_buf_t                   recv_buffer_;
    std::size_t                        send_buffer_elems_      = 0;
    std::size_t                        recv_buffer_elems_      = 0;
    host_buf_t                         host_send_buffer_;
    host_buf_t                         host_recv_buffer_;
    std::size_t                        host_send_buffer_elems_ = 0;
    std::size_t                        host_recv_buffer_elems_ = 0;
    bool                               use_external_work_area_ = false;
    void                              *external_work_area_     = nullptr;
    bool                               use_external_host_work_area_ = false;
    void                              *external_host_work_area_     = nullptr;
    for_each_t                         for_each_;
    std::vector<mpi_request_t>         send_requests_;
    std::vector<mpi_request_t>         recv_requests_;
    mpi_dtype_t                        mpi_value_type_        = mpi_dtype_t();
    bool                               mpi_value_type_inited_ = false;

    std::vector<std::size_t> forward_send_offsets_;
    std::vector<std::size_t> forward_recv_offsets_;
    std::vector<int>         forward_sendcounts_;
    std::vector<int>         forward_sdispls_;
    std::vector<int>         forward_recvcounts_;
    std::vector<int>         forward_rdispls_;
    std::vector<int>         forward_sendcounts_w_;
    std::vector<int>         forward_sdispls_w_;
    std::vector<mpi_dtype_t> forward_sendtypes_w_;
    std::vector<int>         forward_recvcounts_w_;
    std::vector<int>         forward_rdispls_w_;
    std::vector<mpi_dtype_t> forward_recvtypes_w_;

    std::vector<std::size_t> backward_send_offsets_;
    std::vector<std::size_t> backward_recv_offsets_;
    std::vector<int>         backward_sendcounts_;
    std::vector<int>         backward_sdispls_;
    std::vector<int>         backward_recvcounts_;
    std::vector<int>         backward_rdispls_;
    std::vector<int>         backward_sendcounts_w_;
    std::vector<int>         backward_sdispls_w_;
    std::vector<mpi_dtype_t> backward_sendtypes_w_;
    std::vector<int>         backward_recvcounts_w_;
    std::vector<int>         backward_rdispls_w_;
    std::vector<mpi_dtype_t> backward_recvtypes_w_;
    bool                     forward_alltoallw_layout_ready_  = false;
    bool                     backward_alltoallw_layout_ready_ = false;
    bool                     peer_paired_p2p_enabled_         = false;
    bool                     native_message_layout_enabled_   = false;

    std::size_t max_buffer_elems_ = 0;
};

} // namespace detail
} // namespace fftm

#endif // __FFTM_DETAIL_MPI_TRANSPOSE_4D_SAME_ZW_H__
