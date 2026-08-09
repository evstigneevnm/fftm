#ifndef __FFTM_FFT_WRAP_MANY_H__
#define __FFTM_FFT_WRAP_MANY_H__

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <scfd/arrays/array_nd.h>

#include "fft_direction.h"
#include "../profiling.h"

namespace fftm
{
namespace wrap
{

template <template <typename, ::fftm::direction> class BaseFFTWrap, class T>
class fft_wrap_many
{
private:
    using wrap_t       = typename BaseFFTWrap<T, ::fftm::direction::C2CF>::base_t;
    using r2c_wrap_t   = BaseFFTWrap<T, ::fftm::direction::R2C>;
    using c2r_wrap_t   = BaseFFTWrap<T, ::fftm::direction::C2R>;
    using c2cf_wrap_t  = BaseFFTWrap<T, ::fftm::direction::C2CF>;
    using c2cb_wrap_t  = BaseFFTWrap<T, ::fftm::direction::C2CB>;
    using opaque_plan_handle_t = typename c2cf_wrap_t::opaque_plan_handle_t;
    using minimal_reference_c2c_plan_array_handle_t =
        typename c2cf_wrap_t::minimal_reference_c2c_plan_array_handle_t;
    using memory_t     = typename wrap_t::memory_type;
    using ordinal_type = scfd::arrays::ordinal_type;
    using work_array_t = scfd::arrays::array_nd<char, 1, memory_t>;
    using plan_sequence_t = std::vector<wrap_t *>;
    template <::fftm::direction D>
    using typed_plan_sequence_t = std::vector<BaseFFTWrap<T, D> *>;
    using opaque_c2c_plan_sequence_t = std::vector<opaque_plan_handle_t>;
    enum class typed_plan_sequence_kind
    {
        none,
        c2cf,
        c2cb
    };
    struct c2c_plan_array_bundle_t
    {
        std::vector<std::unique_ptr<c2cf_wrap_t>>              plans;
        std::vector<opaque_plan_handle_t>                      handles;
        std::vector<opaque_plan_handle_t>                      context_handles;
        std::vector<typename wrap_t::runtime_api::stream_wrap> streams;
        std::vector<typename wrap_t::runtime_api::stream_t>    external_streams;
        std::vector<std::size_t>                               offsets;
        typename wrap_t::plan_descriptor                       descriptor;
        std::size_t                                            work_stride_bytes = 0;
        std::size_t                                            context_shared_work_size_bytes = 0;
        std::size_t                                            bound_work_stride_bytes = 0;
        bool                                                   owns_raw_handles = false;
        bool                                                   direct_raw_c_api = false;
    };
    struct r2c_c2r_plan_bundle_t
    {
        std::vector<std::unique_ptr<r2c_wrap_t>>               forward_plans;
        std::vector<std::unique_ptr<c2r_wrap_t>>               inverse_plans;
        std::vector<typename wrap_t::runtime_api::stream_wrap> streams;
        std::size_t                                            work_stride_bytes = 0;
        std::size_t                                            bound_work_stride_bytes = 0;
    };

public:
    using real              = T;
    using complex           = typename wrap_t::complex;
    using runtime_api       = typename wrap_t::runtime_api;
    using memory_profiler_t = ::fftm::fftm_memory_profiler;
    using plan_descriptor_t = typename wrap_t::plan_descriptor;
    using plan_sequence_id_t = std::size_t;
    using c2c_plan_array_id_t = std::size_t;
    using r2c_c2r_plan_bundle_id_t = std::size_t;
    using minimal_reference_c2c_plan_array_id_t = std::size_t;

    static plan_sequence_id_t invalid_plan_sequence_id()
    {
        return std::numeric_limits<plan_sequence_id_t>::max();
    }

    static c2c_plan_array_id_t invalid_c2c_plan_array_id()
    {
        return std::numeric_limits<c2c_plan_array_id_t>::max();
    }

    static r2c_c2r_plan_bundle_id_t invalid_r2c_c2r_plan_bundle_id()
    {
        return std::numeric_limits<r2c_c2r_plan_bundle_id_t>::max();
    }

    static minimal_reference_c2c_plan_array_id_t invalid_minimal_reference_c2c_plan_array_id()
    {
        return std::numeric_limits<minimal_reference_c2c_plan_array_id_t>::max();
    }

    fft_wrap_many()
        : work_area_size_( 0 ), activated_( false ), external_activated_( false ), hot_exec_no_sync_( false )
    {
    }

    ~fft_wrap_many()
    {
        release_noexcept_();
    }

    void release()
    {
        release_();
    }

    template <::fftm::direction D, std::size_t Rank>
    void add_plan(
        const std::string &name, const std::array<long long int, Rank> &n,
        const std::array<long long int, Rank> &inembed, long long int istride, long long int idist,
        const std::array<long long int, Rank> &onembed, long long int ostride, long long int odist, long long int batch
    )
    {
        if ( activated_ )
        {
            throw std::logic_error( "fft_wrap_many::add_plan: cannot add new plans after activation." );
        }

        container_.emplace(
            name, std::make_unique<BaseFFTWrap<T, D>>( n, inembed, istride, idist, onembed, ostride, odist, batch )
        );
    }

    template <::fftm::direction D>
    void add_plan_1D(
        const std::string &name, long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch
    )
    {
        add_plan<D>(
            name, std::array<long long int, 1>{ n }, std::array<long long int, 1>{ inembed }, istride, idist,
            std::array<long long int, 1>{ onembed }, ostride, odist, batch
        );
    }

    template <::fftm::direction D>
    void add_plan_1D_default( const std::string &name, long long int n, long long int batch )
    {
        if ( activated_ )
        {
            throw std::logic_error( "fft_wrap_many::add_plan_1D_default: cannot add new plans after activation." );
        }

        container_.emplace( name, std::make_unique<BaseFFTWrap<T, D>>( std::array<long long int, 1>{ n }, batch ) );
    }

    template <::fftm::direction D>
    void add_plan_2D(
        const std::string &name, long long int n0, long long int n1, long long int inembed0, long long int inembed1,
        long long int istride, long long int idist, long long int onembed0, long long int onembed1,
        long long int ostride, long long int odist, long long int batch
    )
    {
        add_plan<D>(
            name, std::array<long long int, 2>{ n0, n1 }, std::array<long long int, 2>{ inembed0, inembed1 }, istride,
            idist, std::array<long long int, 2>{ onembed0, onembed1 }, ostride, odist, batch
        );
    }

    template <::fftm::direction D>
    void add_plan_3D(
        const std::string &name, long long int n0, long long int n1, long long int n2, long long int inembed0,
        long long int inembed1, long long int inembed2, long long int istride, long long int idist,
        long long int onembed0, long long int onembed1, long long int onembed2, long long int ostride,
        long long int odist, long long int batch
    )
    {
        add_plan<D>(
            name, std::array<long long int, 3>{ n0, n1, n2 },
            std::array<long long int, 3>{ inembed0, inembed1, inembed2 }, istride, idist,
            std::array<long long int, 3>{ onembed0, onembed1, onembed2 }, ostride, odist, batch
        );
    }

    r2c_c2r_plan_bundle_id_t make_r2c_c2r_plan_bundle_2D(
        long long int n0, long long int n1,
        long long int forward_inembed0, long long int forward_inembed1,
        long long int forward_istride, long long int forward_idist,
        long long int forward_onembed0, long long int forward_onembed1,
        long long int forward_ostride, long long int forward_odist, long long int forward_batch,
        long long int inverse_inembed0, long long int inverse_inembed1,
        long long int inverse_istride, long long int inverse_idist,
        long long int inverse_onembed0, long long int inverse_onembed1,
        long long int inverse_ostride, long long int inverse_odist, long long int inverse_batch,
        std::size_t lane_count
    )
    {
        if ( activated_ || external_activated_ )
        {
            throw std::logic_error(
                "fft_wrap_many::make_r2c_c2r_plan_bundle_2D: cannot add plan bundles after activation."
            );
        }
        if ( lane_count == 0 )
            throw std::invalid_argument( "fft_wrap_many::make_r2c_c2r_plan_bundle_2D: lane count must be positive" );

        r2c_c2r_plan_bundle_t bundle;
        bundle.forward_plans.reserve( lane_count );
        bundle.inverse_plans.reserve( lane_count );
        bundle.streams.reserve( lane_count );
        for ( std::size_t lane = 0; lane < lane_count; ++lane )
        {
            bundle.streams.emplace_back( true );
            auto forward_plan = std::make_unique<r2c_wrap_t>(
                n0, n1, forward_inembed0, forward_inembed1, forward_istride, forward_idist,
                forward_onembed0, forward_onembed1, forward_ostride, forward_odist, forward_batch
            );
            auto inverse_plan = std::make_unique<c2r_wrap_t>(
                n0, n1, inverse_inembed0, inverse_inembed1, inverse_istride, inverse_idist,
                inverse_onembed0, inverse_onembed1, inverse_ostride, inverse_odist, inverse_batch
            );
            forward_plan->set_stream( bundle.streams.back().stream() );
            inverse_plan->set_stream( bundle.streams.back().stream() );
            bundle.work_stride_bytes = std::max(
                bundle.work_stride_bytes,
                std::max( forward_plan->get_work_size(), inverse_plan->get_work_size() )
            );
            bundle.forward_plans.push_back( std::move( forward_plan ) );
            bundle.inverse_plans.push_back( std::move( inverse_plan ) );
        }

        r2c_c2r_plan_bundles_.push_back( std::move( bundle ) );
        return r2c_c2r_plan_bundles_.size() - 1;
    }

    void activate()
    {
        if ( activated_ )
        {
            throw std::logic_error( "fft_wrap_many::activate: cannot activate again." );
        }
        if ( external_activated_ )
        {
            throw std::logic_error( "fft_wrap_many::activate: cannot use external and internal activation, the class "
                                    "is externally activated." );
        }
        activate_work_size();
        if ( work_area_size_ > 0 )
        {
            work_area_.init( ordinal_cast_( work_area_size_ ) );
            for ( auto &el : container_ )
            {
                el.second->set_work_area( static_cast<void *>( work_area_.raw_ptr() ) );
            }
            bind_r2c_c2r_plan_bundle_work_areas_( static_cast<void *>( work_area_.raw_ptr() ) );
        }

        update_memory_profile_();
        activated_ = true;
    }
    std::size_t activate_work_size()
    {
        for ( const auto &el : container_ )
        {
            work_area_size_ = std::max( work_area_size_, el.second->get_work_size() );
        }
        for ( const auto &bundle : r2c_c2r_plan_bundles_ )
        {
            work_area_size_ = std::max( work_area_size_, r2c_c2r_plan_bundle_work_size_( bundle ) );
        }
        return work_area_size_;
    }

    void set_external_work_area( void *external_work_area )
    {
        if ( activated_ )
        {
            throw std::logic_error(
                "fft_wrap_many::set_external_work_area: cannot set external work_area with activated local work_area."
            );
        }
        for ( auto &el : container_ )
        {
            el.second->set_work_area( external_work_area );
        }
        bind_r2c_c2r_plan_bundle_work_areas_( external_work_area );
        external_activated_ = true;
        update_memory_profile_();
    }
    std::size_t get_work_size() const
    {
        return work_area_size_;
    }

    void set_memory_profiler( memory_profiler_t *profiler, const std::string &prefix )
    {
        memory_profiler_       = profiler;
        memory_profile_prefix_ = prefix;
        update_memory_profile_();
    }

    void set_hot_exec_no_sync_enabled( bool enabled )
    {
        hot_exec_no_sync_ = enabled;
    }

    std::size_t r2c_c2r_plan_bundle_lane_count( r2c_c2r_plan_bundle_id_t bundle_id ) const
    {
        return r2c_c2r_plan_bundle_( bundle_id ).streams.size();
    }

    std::size_t r2c_c2r_plan_bundle_work_size( r2c_c2r_plan_bundle_id_t bundle_id ) const
    {
        return r2c_c2r_plan_bundle_work_size_( r2c_c2r_plan_bundle_( bundle_id ) );
    }

    template <class InPtr, class OutPtr>
    void exec_r2c_c2r_plan_bundle_forward_lane_no_sync(
        r2c_c2r_plan_bundle_id_t bundle_id, std::size_t lane, std::size_t in_offset,
        std::size_t out_offset, InPtr in, OutPtr out
    )
    {
        auto &bundle = r2c_c2r_plan_bundle_( bundle_id );
        if ( lane >= bundle.forward_plans.size() )
            throw std::out_of_range( "fft_wrap_many: R2C/C2R plan-bundle forward lane is out of range" );
        bundle.forward_plans[lane]->exec_no_sync(
            const_cast<void *>( static_cast<const void *>( in + in_offset ) ),
            static_cast<void *>( out + out_offset )
        );
    }

    template <class InPtr, class OutPtr>
    void exec_r2c_c2r_plan_bundle_inverse_lane_no_sync(
        r2c_c2r_plan_bundle_id_t bundle_id, std::size_t lane, std::size_t in_offset,
        std::size_t out_offset, InPtr in, OutPtr out
    )
    {
        auto &bundle = r2c_c2r_plan_bundle_( bundle_id );
        if ( lane >= bundle.inverse_plans.size() )
            throw std::out_of_range( "fft_wrap_many: R2C/C2R plan-bundle inverse lane is out of range" );
        bundle.inverse_plans[lane]->exec_no_sync(
            const_cast<void *>( static_cast<const void *>( in + in_offset ) ),
            static_cast<void *>( out + out_offset )
        );
    }

    bool r2c_c2r_plan_bundle_lane_ready( r2c_c2r_plan_bundle_id_t bundle_id, std::size_t lane ) const
    {
        const auto &bundle = r2c_c2r_plan_bundle_( bundle_id );
        if ( lane >= bundle.streams.size() )
            throw std::out_of_range( "fft_wrap_many: R2C/C2R plan-bundle readiness lane is out of range" );
        return runtime_api::stream_ready( bundle.streams[lane].stream() );
    }

    void synchronize_r2c_c2r_plan_bundle_lane(
        r2c_c2r_plan_bundle_id_t bundle_id, std::size_t lane
    ) const
    {
        const auto &bundle = r2c_c2r_plan_bundle_( bundle_id );
        if ( lane >= bundle.streams.size() )
            throw std::out_of_range( "fft_wrap_many: R2C/C2R plan-bundle synchronization lane is out of range" );
        runtime_api::stream_synchronize( bundle.streams[lane].stream() );
    }

    void synchronize_r2c_c2r_plan_bundle( r2c_c2r_plan_bundle_id_t bundle_id ) const
    {
        const auto &bundle = r2c_c2r_plan_bundle_( bundle_id );
        for ( const auto &stream : bundle.streams )
            runtime_api::stream_synchronize( stream.stream() );
    }

    template <class ArrayIn, class ArrayOut>
    void exec( const std::string &name, const ArrayIn &in, ArrayOut &out )
    {
        wrap_t *plan = container_.at( name ).get();
        if ( hot_exec_no_sync_ )
            plan->exec_no_sync( in.raw_ptr(), out.raw_ptr() );
        else
            plan->exec( in.raw_ptr(), out.raw_ptr() );
    }

    template <class ArrayIn, class ArrayOut>
    void exec_named_no_sync( const std::string &name, const ArrayIn &in, ArrayOut &out )
    {
        container_.at( name )->exec_no_sync( in.raw_ptr(), out.raw_ptr() );
    }

    void synchronize_device_execution() const
    {
        runtime_api::device_synchronize();
    }

    void bind_plan_to_owned_nonblocking_stream( const std::string &name )
    {
        if ( !activated_ && !external_activated_ )
        {
            throw std::logic_error(
                "fft_wrap_many::bind_plan_to_owned_nonblocking_stream: plans must be activated first"
            );
        }
        if ( owned_plan_streams_.count( name ) != 0 )
        {
            throw std::logic_error(
                "fft_wrap_many::bind_plan_to_owned_nonblocking_stream: plan already has an owned stream"
            );
        }

        typename runtime_api::stream_t stream = runtime_api::create_nonblocking_stream();
        try
        {
            const auto inserted = owned_plan_streams_.emplace( name, stream );
            if ( !inserted.second )
                throw std::logic_error( "fft_wrap_many: failed to register an owned plan stream" );
            container_.at( name )->set_stream( stream );
        }
        catch ( ... )
        {
            owned_plan_streams_.erase( name );
            runtime_api::destroy_stream( stream );
            throw;
        }
    }

    void synchronize_owned_plan_stream( const std::string &name ) const
    {
        const auto it = owned_plan_streams_.find( name );
        if ( it == owned_plan_streams_.end() )
            throw std::out_of_range( "fft_wrap_many: plan has no owned execution stream" );
        runtime_api::stream_synchronize( it->second );
    }

    void exec_raw( const std::string &name, void *in, void *out )
    {
        wrap_t *plan = container_.at( name ).get();
        if ( hot_exec_no_sync_ )
            plan->exec_no_sync( in, out );
        else
            plan->exec( in, out );
    }

    void exec_raw_direction( const std::string &name, ::fftm::direction exec_dir, void *in, void *out )
    {
        wrap_t *plan = container_.at( name ).get();
        if ( hot_exec_no_sync_ )
            plan->exec_direction_no_sync( exec_dir, in, out );
        else
            plan->exec_direction( exec_dir, in, out );
    }

    plan_sequence_id_t make_plan_sequence( const std::vector<std::string> &names )
    {
        plan_sequence_t sequence;
        sequence.reserve( names.size() );
        for ( const auto &name : names )
        {
            sequence.push_back( container_.at( name ).get() );
        }

        typed_plan_sequence_kind typed_kind  = typed_plan_sequence_kind::none;
        std::size_t              typed_index = invalid_plan_sequence_id();
        std::size_t              opaque_index = invalid_plan_sequence_id();
        if ( !sequence.empty() )
        {
            const auto dir = sequence.front()->get_direction();
            bool       same_direction = true;
            for ( const auto *plan : sequence )
            {
                same_direction = same_direction && plan->get_direction() == dir;
            }
            if ( same_direction && dir == ::fftm::direction::C2CF )
            {
                typed_plan_sequence_t<::fftm::direction::C2CF> typed_sequence;
                opaque_c2c_plan_sequence_t opaque_sequence;
                typed_sequence.reserve( sequence.size() );
                opaque_sequence.reserve( sequence.size() );
                for ( auto *plan : sequence )
                {
                    auto *typed_plan = dynamic_cast<BaseFFTWrap<T, ::fftm::direction::C2CF> *>( plan );
                    if ( typed_plan == nullptr )
                    {
                        throw std::logic_error( "fft_wrap_many::make_plan_sequence: C2CF sequence type mismatch." );
                    }
                    typed_sequence.push_back( typed_plan );
                    opaque_sequence.push_back( typed_plan->opaque_plan_handle() );
                }
                typed_index = c2cf_plan_sequences_.size();
                c2cf_plan_sequences_.push_back( std::move( typed_sequence ) );
                opaque_index = opaque_c2c_plan_sequences_.size();
                opaque_c2c_plan_sequences_.push_back( std::move( opaque_sequence ) );
                typed_kind = typed_plan_sequence_kind::c2cf;
            }
            else if ( same_direction && dir == ::fftm::direction::C2CB )
            {
                typed_plan_sequence_t<::fftm::direction::C2CB> typed_sequence;
                opaque_c2c_plan_sequence_t opaque_sequence;
                typed_sequence.reserve( sequence.size() );
                opaque_sequence.reserve( sequence.size() );
                for ( auto *plan : sequence )
                {
                    auto *typed_plan = dynamic_cast<BaseFFTWrap<T, ::fftm::direction::C2CB> *>( plan );
                    if ( typed_plan == nullptr )
                    {
                        throw std::logic_error( "fft_wrap_many::make_plan_sequence: C2CB sequence type mismatch." );
                    }
                    typed_sequence.push_back( typed_plan );
                    opaque_sequence.push_back( typed_plan->opaque_plan_handle() );
                }
                typed_index = c2cb_plan_sequences_.size();
                c2cb_plan_sequences_.push_back( std::move( typed_sequence ) );
                opaque_index = opaque_c2c_plan_sequences_.size();
                opaque_c2c_plan_sequences_.push_back( std::move( opaque_sequence ) );
                typed_kind = typed_plan_sequence_kind::c2cb;
            }
        }

        plan_sequences_.push_back( std::move( sequence ) );
        plan_sequence_kinds_.push_back( typed_kind );
        plan_sequence_typed_indices_.push_back( typed_index );
        plan_sequence_opaque_indices_.push_back( opaque_index );
        return plan_sequences_.size() - 1;
    }

    template <class InPtr, class OutPtr>
    void exec_plan_sequence_offsets(
        plan_sequence_id_t sequence_id, const std::vector<std::size_t> &offsets, InPtr in, OutPtr out
    )
    {
        const auto &sequence = plan_sequence_( sequence_id );
        if ( sequence.size() != offsets.size() )
        {
            throw std::logic_error( "fft_wrap_many::exec_plan_sequence_offsets: plan/offset count mismatch." );
        }
        for ( std::size_t i = 0; i < sequence.size(); ++i )
        {
            sequence[i]->exec(
                static_cast<void *>( in + offsets[i] ), static_cast<void *>( out + offsets[i] )
            );
        }
    }

    template <class InPtr, class OutPtr>
    void exec_plan_sequence_offsets_direction(
        plan_sequence_id_t sequence_id, const std::vector<std::size_t> &offsets, ::fftm::direction exec_dir,
        InPtr in, OutPtr out
    )
    {
        const auto &sequence = plan_sequence_( sequence_id );
        if ( sequence.size() != offsets.size() )
        {
            throw std::logic_error( "fft_wrap_many::exec_plan_sequence_offsets_direction: plan/offset count mismatch." );
        }
        for ( std::size_t i = 0; i < sequence.size(); ++i )
        {
            sequence[i]->exec_direction(
                exec_dir, static_cast<void *>( in + offsets[i] ), static_cast<void *>( out + offsets[i] )
            );
        }
    }

    template <class InPtr, class OutPtr>
    void exec_plan_sequence_offsets_direction_repeated(
        plan_sequence_id_t sequence_id, const std::vector<std::size_t> &offsets, ::fftm::direction exec_dir,
        InPtr in, OutPtr out, std::size_t repeats
    )
    {
        for ( std::size_t repeat = 0; repeat < repeats; ++repeat )
        {
            exec_plan_sequence_offsets_direction( sequence_id, offsets, exec_dir, in, out );
        }
    }

    template <class InPtr, class OutPtr>
    void exec_plan_sequence_offsets_tight(
        plan_sequence_id_t sequence_id, const std::vector<std::size_t> &offsets, InPtr in, OutPtr out
    )
    {
        if ( sequence_id == invalid_plan_sequence_id() || sequence_id >= plan_sequence_kinds_.size() )
        {
            throw std::out_of_range( "fft_wrap_many::exec_plan_sequence_offsets_tight: invalid plan sequence id" );
        }
        switch ( plan_sequence_kinds_[sequence_id] )
        {
        case typed_plan_sequence_kind::c2cf:
            exec_typed_plan_sequence_offsets_(
                c2cf_plan_sequences_.at( plan_sequence_typed_indices_.at( sequence_id ) ), offsets, in, out
            );
            return;
        case typed_plan_sequence_kind::c2cb:
            exec_typed_plan_sequence_offsets_(
                c2cb_plan_sequences_.at( plan_sequence_typed_indices_.at( sequence_id ) ), offsets, in, out
            );
            return;
        case typed_plan_sequence_kind::none:
            break;
        }
        throw std::logic_error( "fft_wrap_many::exec_plan_sequence_offsets_tight: sequence is not typed." );
    }

    template <class InPtr, class OutPtr>
    void exec_plan_sequence_offsets_tight_direction(
        plan_sequence_id_t sequence_id, const std::vector<std::size_t> &offsets, ::fftm::direction exec_dir,
        InPtr in, OutPtr out
    )
    {
        if ( sequence_id == invalid_plan_sequence_id() || sequence_id >= plan_sequence_kinds_.size() )
        {
            throw std::out_of_range(
                "fft_wrap_many::exec_plan_sequence_offsets_tight_direction: invalid plan sequence id"
            );
        }
        switch ( plan_sequence_kinds_[sequence_id] )
        {
        case typed_plan_sequence_kind::c2cf:
            exec_typed_plan_sequence_offsets_direction_(
                c2cf_plan_sequences_.at( plan_sequence_typed_indices_.at( sequence_id ) ), offsets, exec_dir, in, out
            );
            return;
        case typed_plan_sequence_kind::c2cb:
            exec_typed_plan_sequence_offsets_direction_(
                c2cb_plan_sequences_.at( plan_sequence_typed_indices_.at( sequence_id ) ), offsets, exec_dir, in, out
            );
            return;
        case typed_plan_sequence_kind::none:
            break;
        }
        exec_plan_sequence_offsets_direction( sequence_id, offsets, exec_dir, in, out );
    }

    template <class InPtr, class OutPtr>
    void exec_plan_sequence_offsets_tight_direction_repeated(
        plan_sequence_id_t sequence_id, const std::vector<std::size_t> &offsets, ::fftm::direction exec_dir,
        InPtr in, OutPtr out, std::size_t repeats
    )
    {
        for ( std::size_t repeat = 0; repeat < repeats; ++repeat )
        {
            exec_plan_sequence_offsets_tight_direction( sequence_id, offsets, exec_dir, in, out );
        }
    }

    template <class InPtr, class OutPtr>
    void exec_plan_sequence_offsets_opaque_direction(
        plan_sequence_id_t sequence_id, const std::vector<std::size_t> &offsets, ::fftm::direction exec_dir,
        InPtr in, OutPtr out
    )
    {
        if ( sequence_id == invalid_plan_sequence_id() || sequence_id >= plan_sequence_opaque_indices_.size() )
        {
            throw std::out_of_range(
                "fft_wrap_many::exec_plan_sequence_offsets_opaque_direction: invalid plan sequence id"
            );
        }
        const std::size_t opaque_index = plan_sequence_opaque_indices_.at( sequence_id );
        if ( opaque_index == invalid_plan_sequence_id() )
        {
            throw std::logic_error(
                "fft_wrap_many::exec_plan_sequence_offsets_opaque_direction: sequence has no opaque C2C plan array"
            );
        }
        const auto &sequence = opaque_c2c_plan_sequences_.at( opaque_index );
        if ( sequence.size() != offsets.size() )
        {
            throw std::logic_error(
                "fft_wrap_many::exec_plan_sequence_offsets_opaque_direction: plan/offset count mismatch"
            );
        }
        for ( std::size_t i = 0; i < sequence.size(); ++i )
        {
            c2cf_wrap_t::exec_opaque_c2c(
                sequence[i], exec_dir, static_cast<void *>( in + offsets[i] ),
                static_cast<void *>( out + offsets[i] )
            );
        }
    }

    template <class InPtr, class OutPtr>
    void exec_plan_sequence_offsets_direction_no_sync(
        plan_sequence_id_t sequence_id, const std::vector<std::size_t> &offsets, ::fftm::direction exec_dir,
        InPtr in, OutPtr out
    )
    {
        if ( sequence_id == invalid_plan_sequence_id() || sequence_id >= plan_sequence_opaque_indices_.size() )
        {
            throw std::out_of_range(
                "fft_wrap_many::exec_plan_sequence_offsets_direction_no_sync: invalid plan sequence id"
            );
        }
        const std::size_t opaque_index = plan_sequence_opaque_indices_.at( sequence_id );
        if ( opaque_index == invalid_plan_sequence_id() )
        {
            throw std::logic_error(
                "fft_wrap_many::exec_plan_sequence_offsets_direction_no_sync: sequence has no opaque C2C plan array"
            );
        }
        const auto &sequence = opaque_c2c_plan_sequences_.at( opaque_index );
        if ( sequence.size() != offsets.size() )
        {
            throw std::logic_error(
                "fft_wrap_many::exec_plan_sequence_offsets_direction_no_sync: plan/offset count mismatch"
            );
        }
        for ( std::size_t i = 0; i < sequence.size(); ++i )
        {
            c2cf_wrap_t::exec_opaque_c2c_no_sync(
                sequence[i], exec_dir, static_cast<void *>( in + offsets[i] ),
                static_cast<void *>( out + offsets[i] )
            );
        }
    }

    template <class InPtr, class OutPtr>
    void exec_plan_sequence_offsets_no_sync(
        plan_sequence_id_t sequence_id, const std::vector<std::size_t> &offsets, InPtr in, OutPtr out
    )
    {
        if ( sequence_id == invalid_plan_sequence_id() || sequence_id >= plan_sequence_kinds_.size() )
        {
            throw std::out_of_range( "fft_wrap_many::exec_plan_sequence_offsets_no_sync: invalid plan sequence id" );
        }
        switch ( plan_sequence_kinds_[sequence_id] )
        {
        case typed_plan_sequence_kind::c2cf:
            exec_plan_sequence_offsets_direction_no_sync( sequence_id, offsets, ::fftm::direction::C2CF, in, out );
            return;
        case typed_plan_sequence_kind::c2cb:
            exec_plan_sequence_offsets_direction_no_sync( sequence_id, offsets, ::fftm::direction::C2CB, in, out );
            return;
        case typed_plan_sequence_kind::none:
            break;
        }
        throw std::logic_error(
            "fft_wrap_many::exec_plan_sequence_offsets_no_sync: sequence has no typed C2C direction"
        );
    }

    template <class InPtr, class OutPtr>
    void exec_plan_sequence_dual_offsets_no_sync(
        plan_sequence_id_t sequence_id, const std::vector<std::size_t> &in_offsets,
        const std::vector<std::size_t> &out_offsets, InPtr in, OutPtr out
    )
    {
        const auto &sequence = plan_sequence_( sequence_id );
        if ( sequence.size() != in_offsets.size() || sequence.size() != out_offsets.size() )
        {
            throw std::logic_error(
                "fft_wrap_many::exec_plan_sequence_dual_offsets_no_sync: plan/offset count mismatch"
            );
        }
        for ( std::size_t i = 0; i < sequence.size(); ++i )
        {
            sequence[i]->exec_no_sync(
                static_cast<void *>( in + in_offsets[i] ), static_cast<void *>( out + out_offsets[i] )
            );
        }
    }

    template <class InPtr, class OutPtr>
    void exec_plan_sequence_offsets_direction_no_sync_repeated(
        plan_sequence_id_t sequence_id, const std::vector<std::size_t> &offsets, ::fftm::direction exec_dir,
        InPtr in, OutPtr out, std::size_t repeats
    )
    {
        for ( std::size_t repeat = 0; repeat < repeats; ++repeat )
        {
            exec_plan_sequence_offsets_direction_no_sync( sequence_id, offsets, exec_dir, in, out );
        }
    }

    template <class InPtr, class OutPtr>
    void exec_plan_sequence_offsets_opaque_direction_repeated(
        plan_sequence_id_t sequence_id, const std::vector<std::size_t> &offsets, ::fftm::direction exec_dir,
        InPtr in, OutPtr out, std::size_t repeats
    )
    {
        for ( std::size_t repeat = 0; repeat < repeats; ++repeat )
        {
            exec_plan_sequence_offsets_opaque_direction( sequence_id, offsets, exec_dir, in, out );
        }
    }

    c2c_plan_array_id_t make_c2c_plan_array_1D(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets
    )
    {
        if ( activated_ )
        {
            throw std::logic_error( "fft_wrap_many::make_c2c_plan_array_1D: cannot add plan arrays after activation." );
        }

        c2c_plan_array_bundle_t bundle;
        bundle.offsets = offsets;
        bundle.plans.reserve( offsets.size() );
        bundle.handles.reserve( offsets.size() );
        bundle.streams.reserve( offsets.size() );
        for ( std::size_t i = 0; i < offsets.size(); ++i )
        {
            bundle.streams.emplace_back( true );
            auto plan = std::make_unique<c2cf_wrap_t>(
                n, inembed, istride, idist, onembed, ostride, odist, batch
            );
            plan->set_stream( bundle.streams.back().stream() );
            bundle.work_stride_bytes = std::max( bundle.work_stride_bytes, plan->get_work_size() );
            if ( i == 0 )
                bundle.descriptor = plan->descriptor();
            bundle.handles.push_back( plan->opaque_plan_handle() );
            bundle.plans.push_back( std::move( plan ) );
        }

        c2c_plan_array_bundles_.push_back( std::move( bundle ) );
        return c2c_plan_array_bundles_.size() - 1;
    }

    c2c_plan_array_id_t make_raw_c2c_plan_array_1D(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets
    )
    {
        if ( activated_ )
        {
            throw std::logic_error(
                "fft_wrap_many::make_raw_c2c_plan_array_1D: cannot add plan arrays after activation."
            );
        }
        return make_raw_c2c_plan_array_1D_unchecked_(
            n, inembed, istride, idist, onembed, ostride, odist, batch, offsets
        );
    }

    c2c_plan_array_id_t make_raw_c2c_plan_array_1D_with_streams(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets, const std::vector<typename runtime_api::stream_t> &streams
    )
    {
        if ( activated_ )
        {
            throw std::logic_error(
                "fft_wrap_many::make_raw_c2c_plan_array_1D_with_streams: cannot add plan arrays after activation."
            );
        }
        return make_raw_c2c_plan_array_1D_with_streams_unchecked_(
            n, inembed, istride, idist, onembed, ostride, odist, batch, offsets, streams
        );
    }

    c2c_plan_array_id_t make_raw_c2c_plan_array_1D_diagnostic(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets
    )
    {
        return make_raw_c2c_plan_array_1D_unchecked_(
            n, inembed, istride, idist, onembed, ostride, odist, batch, offsets
        );
    }

    c2c_plan_array_id_t make_raw_c2c_plan_array_1D_diagnostic_with_streams(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets, const std::vector<typename runtime_api::stream_t> &streams
    )
    {
        return make_raw_c2c_plan_array_1D_with_streams_unchecked_(
            n, inembed, istride, idist, onembed, ostride, odist, batch, offsets, streams
        );
    }

    c2c_plan_array_id_t make_reference_style_raw_c2c_plan_array_1D_diagnostic_with_streams(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets, const std::vector<typename runtime_api::stream_t> &streams
    )
    {
        return make_reference_style_raw_c2c_plan_array_1D_with_streams_unchecked_(
            n, inembed, istride, idist, onembed, ostride, odist, batch, offsets, streams
        );
    }

    c2c_plan_array_id_t make_reference_style_raw_c2c_plan_array_1D_diagnostic_with_owned_streams(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets
    )
    {
        return make_reference_style_raw_c2c_plan_array_1D_with_owned_streams_unchecked_(
            n, inembed, istride, idist, onembed, ostride, odist, batch, offsets
        );
    }

    c2c_plan_array_id_t make_direct_c_raw_c2c_plan_array_1D_diagnostic_with_owned_streams(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets
    )
    {
        return make_direct_c_raw_c2c_plan_array_1D_with_owned_streams_unchecked_(
            n, inembed, istride, idist, onembed, ostride, odist, batch, offsets
        );
    }

    minimal_reference_c2c_plan_array_id_t make_minimal_reference_c2c_plan_array_1D_diagnostic(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets
    )
    {
        minimal_reference_c2c_plan_arrays_.push_back(
            c2cf_wrap_t::create_minimal_reference_c2c_plan_array_1d(
                n, inembed, istride, idist, onembed, ostride, odist, batch, offsets
            )
        );
        return minimal_reference_c2c_plan_arrays_.size() - 1;
    }

    c2c_plan_array_id_t make_reference_opt0_local_plan_context_1D(
        long long int z_n, long long int z_batch, long long int y_n, long long int y_inembed,
        long long int y_istride, long long int y_idist, long long int y_onembed,
        long long int y_ostride, long long int y_odist, long long int y_batch,
        long long int x_n, long long int x_inembed, long long int x_istride,
        long long int x_idist, long long int x_onembed, long long int x_ostride,
        long long int x_odist, long long int x_batch, const std::vector<std::size_t> &offsets,
        const std::vector<typename runtime_api::stream_t> &streams
    )
    {
        if ( activated_ )
        {
            throw std::logic_error(
                "fft_wrap_many::make_reference_opt0_local_plan_context_1D: cannot add plan arrays after activation."
            );
        }
        if ( streams.size() < offsets.size() )
        {
            throw std::logic_error(
                "fft_wrap_many::make_reference_opt0_local_plan_context_1D: stream count mismatch"
            );
        }

        c2c_plan_array_bundle_t bundle;
        bundle.offsets          = offsets;
        bundle.owns_raw_handles = true;
        bundle.external_streams.assign( streams.begin(), streams.begin() + offsets.size() );
        bundle.handles.reserve( offsets.size() );
        bundle.context_handles.reserve( 3 );
        bundle.descriptor.dir            = ::fftm::direction::C2CF;
        bundle.descriptor.rank           = 1;
        bundle.descriptor.default_layout = false;
        bundle.descriptor.n[0]           = y_n;
        bundle.descriptor.inembed[0]     = y_inembed;
        bundle.descriptor.istride        = y_istride;
        bundle.descriptor.idist          = y_idist;
        bundle.descriptor.onembed[0]     = y_onembed;
        bundle.descriptor.ostride        = y_ostride;
        bundle.descriptor.odist          = y_odist;
        bundle.descriptor.batch          = y_batch;

        try
        {
            const opaque_plan_handle_t z_r2c = r2c_wrap_t::create_empty_opaque_plan();
            bundle.context_handles.push_back( z_r2c );
            const opaque_plan_handle_t z_c2r = c2r_wrap_t::create_empty_opaque_plan();
            bundle.context_handles.push_back( z_c2r );
            const opaque_plan_handle_t x_c2c = c2cf_wrap_t::create_empty_opaque_plan();
            bundle.context_handles.push_back( x_c2c );

            bundle.context_shared_work_size_bytes = std::max(
                bundle.context_shared_work_size_bytes,
                r2c_wrap_t::make_opaque_1d_default( z_r2c, z_n, z_batch )
            );
            bundle.context_shared_work_size_bytes = std::max(
                bundle.context_shared_work_size_bytes,
                c2r_wrap_t::make_opaque_1d_default( z_c2r, z_n, z_batch )
            );

            for ( std::size_t i = 0; i < offsets.size(); ++i )
            {
                const opaque_plan_handle_t y_c2c = c2cf_wrap_t::create_empty_opaque_plan();
                bundle.handles.push_back( y_c2c );
                const std::size_t work_size = c2cf_wrap_t::make_opaque_1d(
                    y_c2c, y_n, y_inembed, y_istride, y_idist, y_onembed, y_ostride, y_odist, y_batch
                );
                c2cf_wrap_t::set_opaque_stream( y_c2c, bundle.external_streams[i] );
                bundle.work_stride_bytes = std::max( bundle.work_stride_bytes, work_size );
            }

            bundle.context_shared_work_size_bytes = std::max(
                bundle.context_shared_work_size_bytes,
                c2cf_wrap_t::make_opaque_1d(
                    x_c2c, x_n, x_inembed, x_istride, x_idist, x_onembed, x_ostride, x_odist, x_batch
                )
            );
            bundle.descriptor.work_size = bundle.work_stride_bytes;
        }
        catch ( ... )
        {
            for ( auto handle : bundle.handles )
                c2cf_wrap_t::destroy_opaque_plan_noexcept( handle );
            for ( auto handle : bundle.context_handles )
                c2cf_wrap_t::destroy_opaque_plan_noexcept( handle );
            throw;
        }

        c2c_plan_array_bundles_.push_back( std::move( bundle ) );
        return c2c_plan_array_bundles_.size() - 1;
    }

private:
    c2c_plan_array_id_t make_raw_c2c_plan_array_1D_unchecked_(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets
    )
    {
        c2c_plan_array_bundle_t bundle;
        bundle.offsets          = offsets;
        bundle.owns_raw_handles = true;
        bundle.handles.reserve( offsets.size() );
        bundle.descriptor.dir            = ::fftm::direction::C2CF;
        bundle.descriptor.rank           = 1;
        bundle.descriptor.default_layout = false;
        bundle.descriptor.n[0]           = n;
        bundle.descriptor.inembed[0]     = inembed;
        bundle.descriptor.istride        = istride;
        bundle.descriptor.idist          = idist;
        bundle.descriptor.onembed[0]     = onembed;
        bundle.descriptor.ostride        = ostride;
        bundle.descriptor.odist          = odist;
        bundle.descriptor.batch          = batch;
        try
        {
            for ( std::size_t i = 0; i < offsets.size(); ++i )
            {
                std::size_t work_size = 0;
                bundle.handles.push_back( c2cf_wrap_t::create_opaque_c2c_1d(
                    n, inembed, istride, idist, onembed, ostride, odist, batch, work_size
                ) );
                bundle.work_stride_bytes = std::max( bundle.work_stride_bytes, work_size );
            }
            bundle.descriptor.work_size = bundle.work_stride_bytes;
        }
        catch ( ... )
        {
            for ( auto handle : bundle.handles )
                c2cf_wrap_t::destroy_opaque_plan_noexcept( handle );
            throw;
        }

        c2c_plan_array_bundles_.push_back( std::move( bundle ) );
        return c2c_plan_array_bundles_.size() - 1;
    }

    c2c_plan_array_id_t make_raw_c2c_plan_array_1D_with_streams_unchecked_(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets, const std::vector<typename runtime_api::stream_t> &streams
    )
    {
        if ( streams.size() < offsets.size() )
        {
            throw std::logic_error( "fft_wrap_many::make_raw_c2c_plan_array_1D_with_streams: stream count mismatch" );
        }

        c2c_plan_array_bundle_t bundle;
        bundle.offsets          = offsets;
        bundle.owns_raw_handles = true;
        bundle.external_streams.assign( streams.begin(), streams.begin() + offsets.size() );
        bundle.handles.reserve( offsets.size() );
        bundle.descriptor.dir            = ::fftm::direction::C2CF;
        bundle.descriptor.rank           = 1;
        bundle.descriptor.default_layout = false;
        bundle.descriptor.n[0]           = n;
        bundle.descriptor.inembed[0]     = inembed;
        bundle.descriptor.istride        = istride;
        bundle.descriptor.idist          = idist;
        bundle.descriptor.onembed[0]     = onembed;
        bundle.descriptor.ostride        = ostride;
        bundle.descriptor.odist          = odist;
        bundle.descriptor.batch          = batch;
        try
        {
            for ( std::size_t i = 0; i < offsets.size(); ++i )
            {
                std::size_t work_size = 0;
                bundle.handles.push_back( c2cf_wrap_t::create_opaque_c2c_1d_with_stream(
                    n, inembed, istride, idist, onembed, ostride, odist, batch, work_size,
                    bundle.external_streams[i]
                ) );
                bundle.work_stride_bytes = std::max( bundle.work_stride_bytes, work_size );
            }
            bundle.descriptor.work_size = bundle.work_stride_bytes;
        }
        catch ( ... )
        {
            for ( auto handle : bundle.handles )
                c2cf_wrap_t::destroy_opaque_plan_noexcept( handle );
            throw;
        }

        c2c_plan_array_bundles_.push_back( std::move( bundle ) );
        return c2c_plan_array_bundles_.size() - 1;
    }

    c2c_plan_array_id_t make_reference_style_raw_c2c_plan_array_1D_with_streams_unchecked_(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets, const std::vector<typename runtime_api::stream_t> &streams
    )
    {
        if ( streams.size() < offsets.size() )
        {
            throw std::logic_error(
                "fft_wrap_many::make_reference_style_raw_c2c_plan_array_1D_with_streams: stream count mismatch"
            );
        }

        c2c_plan_array_bundle_t bundle;
        bundle.offsets          = offsets;
        bundle.owns_raw_handles = true;
        bundle.external_streams.assign( streams.begin(), streams.begin() + offsets.size() );
        bundle.handles.reserve( offsets.size() );
        bundle.descriptor.dir            = ::fftm::direction::C2CF;
        bundle.descriptor.rank           = 1;
        bundle.descriptor.default_layout = false;
        bundle.descriptor.n[0]           = n;
        bundle.descriptor.inembed[0]     = inembed;
        bundle.descriptor.istride        = istride;
        bundle.descriptor.idist          = idist;
        bundle.descriptor.onembed[0]     = onembed;
        bundle.descriptor.ostride        = ostride;
        bundle.descriptor.odist          = odist;
        bundle.descriptor.batch          = batch;
        try
        {
            for ( std::size_t i = 0; i < offsets.size(); ++i )
            {
                const opaque_plan_handle_t handle = c2cf_wrap_t::create_empty_opaque_plan();
                bundle.handles.push_back( handle );
                const std::size_t work_size = c2cf_wrap_t::make_opaque_1d(
                    handle, n, inembed, istride, idist, onembed, ostride, odist, batch
                );
                c2cf_wrap_t::set_opaque_stream( handle, bundle.external_streams[i] );
                bundle.work_stride_bytes = std::max( bundle.work_stride_bytes, work_size );
            }
            bundle.descriptor.work_size = bundle.work_stride_bytes;
        }
        catch ( ... )
        {
            for ( auto handle : bundle.handles )
                c2cf_wrap_t::destroy_opaque_plan_noexcept( handle );
            throw;
        }

        c2c_plan_array_bundles_.push_back( std::move( bundle ) );
        return c2c_plan_array_bundles_.size() - 1;
    }

    c2c_plan_array_id_t make_reference_style_raw_c2c_plan_array_1D_with_owned_streams_unchecked_(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets
    )
    {
        c2c_plan_array_bundle_t bundle;
        bundle.offsets          = offsets;
        bundle.owns_raw_handles = true;
        bundle.handles.reserve( offsets.size() );
        bundle.streams.reserve( offsets.size() );
        bundle.descriptor.dir            = ::fftm::direction::C2CF;
        bundle.descriptor.rank           = 1;
        bundle.descriptor.default_layout = false;
        bundle.descriptor.n[0]           = n;
        bundle.descriptor.inembed[0]     = inembed;
        bundle.descriptor.istride        = istride;
        bundle.descriptor.idist          = idist;
        bundle.descriptor.onembed[0]     = onembed;
        bundle.descriptor.ostride        = ostride;
        bundle.descriptor.odist          = odist;
        bundle.descriptor.batch          = batch;
        try
        {
            for ( std::size_t i = 0; i < offsets.size(); ++i )
            {
                bundle.streams.emplace_back( true );
                const opaque_plan_handle_t handle = c2cf_wrap_t::create_empty_opaque_plan();
                bundle.handles.push_back( handle );
                const std::size_t work_size = c2cf_wrap_t::make_opaque_1d(
                    handle, n, inembed, istride, idist, onembed, ostride, odist, batch
                );
                c2cf_wrap_t::set_opaque_stream( handle, bundle.streams.back().stream() );
                bundle.work_stride_bytes = std::max( bundle.work_stride_bytes, work_size );
            }
            bundle.descriptor.work_size = bundle.work_stride_bytes;
        }
        catch ( ... )
        {
            for ( auto handle : bundle.handles )
                c2cf_wrap_t::destroy_opaque_plan_noexcept( handle );
            throw;
        }

        c2c_plan_array_bundles_.push_back( std::move( bundle ) );
        return c2c_plan_array_bundles_.size() - 1;
    }

    c2c_plan_array_id_t make_direct_c_raw_c2c_plan_array_1D_with_owned_streams_unchecked_(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets
    )
    {
        c2c_plan_array_bundle_t bundle;
        bundle.offsets          = offsets;
        bundle.owns_raw_handles = true;
        bundle.direct_raw_c_api = true;
        bundle.handles.reserve( offsets.size() );
        bundle.streams.reserve( offsets.size() );
        bundle.descriptor.dir            = ::fftm::direction::C2CF;
        bundle.descriptor.rank           = 1;
        bundle.descriptor.default_layout = false;
        bundle.descriptor.n[0]           = n;
        bundle.descriptor.inembed[0]     = inembed;
        bundle.descriptor.istride        = istride;
        bundle.descriptor.idist          = idist;
        bundle.descriptor.onembed[0]     = onembed;
        bundle.descriptor.ostride        = ostride;
        bundle.descriptor.odist          = odist;
        bundle.descriptor.batch          = batch;
        try
        {
            for ( std::size_t i = 0; i < offsets.size(); ++i )
            {
                bundle.streams.emplace_back( true );
                std::size_t work_size = 0;
                bundle.handles.push_back( c2cf_wrap_t::create_direct_raw_reference_c2c_1d_with_stream(
                    n, inembed, istride, idist, onembed, ostride, odist, batch, work_size,
                    bundle.streams.back().stream()
                ) );
                bundle.work_stride_bytes = std::max( bundle.work_stride_bytes, work_size );
            }
            bundle.descriptor.work_size = bundle.work_stride_bytes;
        }
        catch ( ... )
        {
            for ( auto handle : bundle.handles )
                c2cf_wrap_t::destroy_opaque_plan_noexcept( handle );
            throw;
        }

        c2c_plan_array_bundles_.push_back( std::move( bundle ) );
        return c2c_plan_array_bundles_.size() - 1;
    }

public:

    std::size_t c2c_plan_array_size( c2c_plan_array_id_t bundle_id ) const
    {
        return c2c_plan_array_bundle_( bundle_id ).handles.size();
    }

    std::size_t c2c_plan_array_work_size( c2c_plan_array_id_t bundle_id ) const
    {
        return c2c_plan_array_bundle_( bundle_id ).work_stride_bytes;
    }

    std::uintptr_t c2c_plan_array_first_handle_token( c2c_plan_array_id_t bundle_id ) const
    {
        const auto &bundle = c2c_plan_array_bundle_( bundle_id );
        return bundle.handles.empty() ? 0 : static_cast<std::uintptr_t>( bundle.handles.front() );
    }

    std::uintptr_t c2c_plan_array_last_handle_token( c2c_plan_array_id_t bundle_id ) const
    {
        const auto &bundle = c2c_plan_array_bundle_( bundle_id );
        return bundle.handles.empty() ? 0 : static_cast<std::uintptr_t>( bundle.handles.back() );
    }

    std::uintptr_t c2c_plan_array_first_stream_token( c2c_plan_array_id_t bundle_id ) const
    {
        const auto &bundle = c2c_plan_array_bundle_( bundle_id );
        if ( !bundle.external_streams.empty() )
            return reinterpret_cast<std::uintptr_t>( bundle.external_streams.front() );
        return bundle.streams.empty() ? 0 : reinterpret_cast<std::uintptr_t>( bundle.streams.front().stream() );
    }

    std::uintptr_t c2c_plan_array_last_stream_token( c2c_plan_array_id_t bundle_id ) const
    {
        const auto &bundle = c2c_plan_array_bundle_( bundle_id );
        if ( !bundle.external_streams.empty() )
            return reinterpret_cast<std::uintptr_t>( bundle.external_streams.back() );
        return bundle.streams.empty() ? 0 : reinterpret_cast<std::uintptr_t>( bundle.streams.back().stream() );
    }

    plan_descriptor_t c2c_plan_array_descriptor( c2c_plan_array_id_t bundle_id ) const
    {
        return c2c_plan_array_bundle_( bundle_id ).descriptor;
    }

    std::size_t minimal_reference_c2c_plan_array_size(
        minimal_reference_c2c_plan_array_id_t bundle_id
    ) const
    {
        return c2cf_wrap_t::minimal_reference_c2c_plan_array_size(
            minimal_reference_c2c_plan_array_( bundle_id )
        );
    }

    std::size_t minimal_reference_c2c_plan_array_work_size(
        minimal_reference_c2c_plan_array_id_t bundle_id
    ) const
    {
        return c2cf_wrap_t::minimal_reference_c2c_plan_array_work_size(
            minimal_reference_c2c_plan_array_( bundle_id )
        );
    }

    plan_descriptor_t minimal_reference_c2c_plan_array_descriptor(
        minimal_reference_c2c_plan_array_id_t bundle_id
    ) const
    {
        return c2cf_wrap_t::minimal_reference_c2c_plan_array_descriptor(
            minimal_reference_c2c_plan_array_( bundle_id )
        );
    }

    std::uintptr_t minimal_reference_c2c_plan_array_first_handle_token(
        minimal_reference_c2c_plan_array_id_t bundle_id
    ) const
    {
        return c2cf_wrap_t::minimal_reference_c2c_plan_array_first_handle_token(
            minimal_reference_c2c_plan_array_( bundle_id )
        );
    }

    std::uintptr_t minimal_reference_c2c_plan_array_last_handle_token(
        minimal_reference_c2c_plan_array_id_t bundle_id
    ) const
    {
        return c2cf_wrap_t::minimal_reference_c2c_plan_array_last_handle_token(
            minimal_reference_c2c_plan_array_( bundle_id )
        );
    }

    std::uintptr_t minimal_reference_c2c_plan_array_first_stream_token(
        minimal_reference_c2c_plan_array_id_t bundle_id
    ) const
    {
        return c2cf_wrap_t::minimal_reference_c2c_plan_array_first_stream_token(
            minimal_reference_c2c_plan_array_( bundle_id )
        );
    }

    std::uintptr_t minimal_reference_c2c_plan_array_last_stream_token(
        minimal_reference_c2c_plan_array_id_t bundle_id
    ) const
    {
        return c2cf_wrap_t::minimal_reference_c2c_plan_array_last_stream_token(
            minimal_reference_c2c_plan_array_( bundle_id )
        );
    }

    void bind_c2c_plan_array_work_areas(
        c2c_plan_array_id_t bundle_id, void *base_work_area, std::size_t work_stride_bytes
    )
    {
        auto &bundle = c2c_plan_array_bundle_( bundle_id );
        const std::size_t stride = std::max( work_stride_bytes, bundle.work_stride_bytes );
        char *raw = static_cast<char *>( base_work_area );
        if ( bundle.owns_raw_handles )
        {
            for ( auto handle : bundle.context_handles )
            {
                c2cf_wrap_t::set_opaque_work_area( handle, static_cast<void *>( raw ) );
            }
            for ( std::size_t i = 0; i < bundle.handles.size(); ++i )
            {
                if ( bundle.direct_raw_c_api )
                    c2cf_wrap_t::set_direct_raw_work_area( bundle.handles[i], static_cast<void *>( raw + i * stride ) );
                else
                    c2cf_wrap_t::set_opaque_work_area( bundle.handles[i], static_cast<void *>( raw + i * stride ) );
            }
        }
        else
        {
            for ( std::size_t i = 0; i < bundle.plans.size(); ++i )
            {
                bundle.plans[i]->set_work_area( static_cast<void *>( raw + i * stride ) );
                bundle.handles[i] = bundle.plans[i]->opaque_plan_handle();
            }
        }
        bundle.bound_work_stride_bytes = stride;
    }

    void bind_minimal_reference_c2c_plan_array_work_areas(
        minimal_reference_c2c_plan_array_id_t bundle_id, void *base_work_area, std::size_t work_stride_bytes
    )
    {
        c2cf_wrap_t::bind_minimal_reference_c2c_plan_array_work_areas(
            minimal_reference_c2c_plan_array_( bundle_id ), base_work_area, work_stride_bytes
        );
    }

    void bind_c2c_plan_array_streams(
        c2c_plan_array_id_t bundle_id, const std::vector<typename runtime_api::stream_t> &streams
    )
    {
        auto &bundle = c2c_plan_array_bundle_( bundle_id );
        if ( streams.size() < bundle.handles.size() )
        {
            throw std::logic_error( "fft_wrap_many::bind_c2c_plan_array_streams: stream count mismatch" );
        }
        bundle.external_streams.assign( streams.begin(), streams.begin() + bundle.handles.size() );
        for ( std::size_t i = 0; i < bundle.handles.size(); ++i )
        {
            if ( bundle.owns_raw_handles )
            {
                c2cf_wrap_t::set_opaque_stream( bundle.handles[i], bundle.external_streams[i] );
            }
            else
            {
                bundle.plans[i]->set_stream( bundle.external_streams[i] );
                bundle.handles[i] = bundle.plans[i]->opaque_plan_handle();
            }
        }
    }

    void synchronize_c2c_plan_array_streams( c2c_plan_array_id_t bundle_id )
    {
        auto &bundle = c2c_plan_array_bundle_( bundle_id );
        if ( !bundle.external_streams.empty() )
        {
            for ( auto stream : bundle.external_streams )
            {
                runtime_api::stream_synchronize( stream );
            }
            return;
        }
        for ( auto &stream : bundle.streams )
        {
            runtime_api::stream_synchronize( stream.stream() );
        }
    }

    void synchronize_minimal_reference_c2c_plan_array_streams(
        minimal_reference_c2c_plan_array_id_t bundle_id
    )
    {
        c2cf_wrap_t::synchronize_minimal_reference_c2c_plan_array_streams(
            minimal_reference_c2c_plan_array_( bundle_id )
        );
    }

    template <class InPtr, class OutPtr>
    void exec_c2c_plan_array_direction(
        c2c_plan_array_id_t bundle_id, ::fftm::direction exec_dir, InPtr in, OutPtr out
    )
    {
        const auto &bundle = c2c_plan_array_bundle_( bundle_id );
        for ( std::size_t i = 0; i < bundle.handles.size(); ++i )
        {
            if ( bundle.direct_raw_c_api )
            {
                c2cf_wrap_t::exec_direct_raw_c2c(
                    bundle.handles[i], exec_dir, static_cast<void *>( in + bundle.offsets[i] ),
                    static_cast<void *>( out + bundle.offsets[i] )
                );
            }
            else
            {
                c2cf_wrap_t::exec_opaque_c2c(
                    bundle.handles[i], exec_dir, static_cast<void *>( in + bundle.offsets[i] ),
                    static_cast<void *>( out + bundle.offsets[i] )
                );
            }
        }
    }

    template <class InPtr, class OutPtr>
    void exec_c2c_plan_array_direction_no_sync(
        c2c_plan_array_id_t bundle_id, ::fftm::direction exec_dir, InPtr in, OutPtr out
    )
    {
        const auto &bundle = c2c_plan_array_bundle_( bundle_id );
        for ( std::size_t i = 0; i < bundle.handles.size(); ++i )
        {
            if ( bundle.direct_raw_c_api )
            {
                c2cf_wrap_t::exec_direct_raw_c2c_no_sync(
                    bundle.handles[i], exec_dir, static_cast<void *>( in + bundle.offsets[i] ),
                    static_cast<void *>( out + bundle.offsets[i] )
                );
            }
            else
            {
                c2cf_wrap_t::exec_opaque_c2c_no_sync(
                    bundle.handles[i], exec_dir, static_cast<void *>( in + bundle.offsets[i] ),
                    static_cast<void *>( out + bundle.offsets[i] )
                );
            }
        }
    }

    template <class InPtr, class OutPtr>
    void exec_c2c_plan_array_direction_dual_offsets_no_sync(
        c2c_plan_array_id_t bundle_id, ::fftm::direction exec_dir, const std::vector<std::size_t> &in_offsets,
        const std::vector<std::size_t> &out_offsets, InPtr in, OutPtr out
    )
    {
        const auto &bundle = c2c_plan_array_bundle_( bundle_id );
        if ( bundle.handles.size() != in_offsets.size() || bundle.handles.size() != out_offsets.size() )
        {
            throw std::logic_error(
                "fft_wrap_many::exec_c2c_plan_array_direction_dual_offsets_no_sync: offset count mismatch"
            );
        }
        for ( std::size_t i = 0; i < bundle.handles.size(); ++i )
        {
            if ( bundle.direct_raw_c_api )
            {
                c2cf_wrap_t::exec_direct_raw_c2c_no_sync(
                    bundle.handles[i], exec_dir, static_cast<void *>( in + in_offsets[i] ),
                    static_cast<void *>( out + out_offsets[i] )
                );
            }
            else
            {
                c2cf_wrap_t::exec_opaque_c2c_no_sync(
                    bundle.handles[i], exec_dir, static_cast<void *>( in + in_offsets[i] ),
                    static_cast<void *>( out + out_offsets[i] )
                );
            }
        }
    }

    template <class InPtr, class OutPtr>
    void exec_c2c_plan_array_direction_repeated(
        c2c_plan_array_id_t bundle_id, ::fftm::direction exec_dir, InPtr in, OutPtr out, std::size_t repeats
    )
    {
        for ( std::size_t repeat = 0; repeat < repeats; ++repeat )
        {
            exec_c2c_plan_array_direction( bundle_id, exec_dir, in, out );
        }
    }

    template <class InPtr, class OutPtr>
    void exec_c2c_plan_array_direction_no_sync_repeated(
        c2c_plan_array_id_t bundle_id, ::fftm::direction exec_dir, InPtr in, OutPtr out, std::size_t repeats
    )
    {
        for ( std::size_t repeat = 0; repeat < repeats; ++repeat )
        {
            exec_c2c_plan_array_direction_no_sync( bundle_id, exec_dir, in, out );
        }
    }

    template <class InPtr, class OutPtr>
    void exec_minimal_reference_c2c_plan_array_direction(
        minimal_reference_c2c_plan_array_id_t bundle_id, ::fftm::direction exec_dir, InPtr in, OutPtr out
    )
    {
        c2cf_wrap_t::exec_minimal_reference_c2c_plan_array(
            minimal_reference_c2c_plan_array_( bundle_id ), exec_dir, static_cast<void *>( in ),
            static_cast<void *>( out )
        );
    }

    template <class InPtr, class OutPtr>
    void exec_minimal_reference_c2c_plan_array_direction_no_sync(
        minimal_reference_c2c_plan_array_id_t bundle_id, ::fftm::direction exec_dir, InPtr in, OutPtr out
    )
    {
        c2cf_wrap_t::exec_minimal_reference_c2c_plan_array_no_sync(
            minimal_reference_c2c_plan_array_( bundle_id ), exec_dir, static_cast<void *>( in ),
            static_cast<void *>( out )
        );
    }

    template <class InPtr, class OutPtr>
    void exec_minimal_reference_c2c_plan_array_direction_repeated(
        minimal_reference_c2c_plan_array_id_t bundle_id, ::fftm::direction exec_dir, InPtr in, OutPtr out,
        std::size_t repeats
    )
    {
        c2cf_wrap_t::exec_minimal_reference_c2c_plan_array_repeated(
            minimal_reference_c2c_plan_array_( bundle_id ), exec_dir, static_cast<void *>( in ),
            static_cast<void *>( out ), repeats
        );
    }

    template <class InPtr, class OutPtr>
    void exec_minimal_reference_c2c_plan_array_direction_no_sync_repeated(
        minimal_reference_c2c_plan_array_id_t bundle_id, ::fftm::direction exec_dir, InPtr in, OutPtr out,
        std::size_t repeats
    )
    {
        c2cf_wrap_t::exec_minimal_reference_c2c_plan_array_no_sync_repeated(
            minimal_reference_c2c_plan_array_( bundle_id ), exec_dir, static_cast<void *>( in ),
            static_cast<void *>( out ), repeats
        );
    }

    std::size_t plan_sequence_size( plan_sequence_id_t sequence_id ) const
    {
        return plan_sequence_( sequence_id ).size();
    }

    const char *plan_sequence_kind_name( plan_sequence_id_t sequence_id ) const
    {
        if ( sequence_id == invalid_plan_sequence_id() || sequence_id >= plan_sequence_kinds_.size() )
        {
            return "invalid";
        }
        switch ( plan_sequence_kinds_[sequence_id] )
        {
        case typed_plan_sequence_kind::c2cf:
            return "c2cf";
        case typed_plan_sequence_kind::c2cb:
            return "c2cb";
        case typed_plan_sequence_kind::none:
            return "none";
        }
        return "unknown";
    }

    std::size_t work_size( const std::string &name ) const
    {
        return container_.at( name )->get_work_size();
    }

    void set_work_area( const std::string &name, void *work_area )
    {
        container_.at( name )->set_work_area( work_area );
    }

    void set_stream( const std::string &name, typename runtime_api::stream_t stream )
    {
        container_.at( name )->set_stream( stream );
    }

    void recreate_plan_with_stream_and_work_area(
        const std::string &name, typename runtime_api::stream_t stream, void *work_area
    )
    {
        container_.at( name )->recreate_with_stream_and_work_area( stream, work_area );
    }

    plan_descriptor_t plan_descriptor( const std::string &name ) const
    {
        return container_.at( name )->descriptor();
    }

    std::vector<std::pair<std::string, plan_descriptor_t>> plan_descriptors() const
    {
        std::vector<std::pair<std::string, plan_descriptor_t>> result;
        result.reserve( container_.size() );
        for ( const auto &el : container_ )
        {
            result.emplace_back( el.first, el.second->descriptor() );
        }
        return result;
    }

private:
    ordinal_type ordinal_cast_( std::size_t value ) const
    {
        const auto max_value = static_cast<std::size_t>( std::numeric_limits<ordinal_type>::max() );
        if ( value > max_value )
        {
            throw std::overflow_error(
                "fft_wrap_many::activate: work area size exceeds scfd::arrays::ordinal_type range. "
                "Rebuild with a wider SCFD_ARRAYS_ORDINAL_TYPE."
            );
        }
        return static_cast<ordinal_type>( value );
    }

    void update_memory_profile_()
    {
        if ( memory_profiler_ == nullptr || memory_profile_prefix_.empty() )
        {
            return;
        }

        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/work_area",
            external_activated_ ? 0 : static_cast<memory_profiler_t::bytes_type>( work_area_size_ )
        );
    }

    const plan_sequence_t &plan_sequence_( plan_sequence_id_t sequence_id ) const
    {
        if ( sequence_id == invalid_plan_sequence_id() || sequence_id >= plan_sequences_.size() )
        {
            throw std::out_of_range( "fft_wrap_many: invalid plan sequence id" );
        }
        return plan_sequences_[sequence_id];
    }

    c2c_plan_array_bundle_t &c2c_plan_array_bundle_( c2c_plan_array_id_t bundle_id )
    {
        if ( bundle_id == invalid_c2c_plan_array_id() || bundle_id >= c2c_plan_array_bundles_.size() )
        {
            throw std::out_of_range( "fft_wrap_many: invalid C2C plan-array id" );
        }
        return c2c_plan_array_bundles_[bundle_id];
    }

    minimal_reference_c2c_plan_array_handle_t minimal_reference_c2c_plan_array_(
        minimal_reference_c2c_plan_array_id_t bundle_id
    ) const
    {
        if ( bundle_id == invalid_minimal_reference_c2c_plan_array_id() ||
             bundle_id >= minimal_reference_c2c_plan_arrays_.size() )
        {
            throw std::out_of_range( "fft_wrap_many: invalid minimal reference C2C plan-array id" );
        }
        return minimal_reference_c2c_plan_arrays_[bundle_id];
    }

    void destroy_minimal_reference_c2c_plan_arrays_()
    {
        for ( auto &bundle : minimal_reference_c2c_plan_arrays_ )
        {
            c2cf_wrap_t::destroy_minimal_reference_c2c_plan_array_noexcept( bundle );
            bundle = nullptr;
        }
    }

    void destroy_raw_c2c_plan_array_bundles_()
    {
        for ( auto &bundle : c2c_plan_array_bundles_ )
        {
            if ( !bundle.owns_raw_handles )
                continue;
            for ( auto &handle : bundle.handles )
            {
                c2cf_wrap_t::destroy_opaque_plan_noexcept( handle );
                handle = 0;
            }
            for ( auto &handle : bundle.context_handles )
            {
                c2cf_wrap_t::destroy_opaque_plan_noexcept( handle );
                handle = 0;
            }
        }
    }

    void release_()
    {
        // Sequence entries are non-owning plan pointers. Drop them before the
        // plan containers so no stale references survive a manual release.
        plan_sequences_.clear();
        c2cf_plan_sequences_.clear();
        c2cb_plan_sequences_.clear();
        opaque_c2c_plan_sequences_.clear();
        plan_sequence_kinds_.clear();
        plan_sequence_typed_indices_.clear();
        plan_sequence_opaque_indices_.clear();

        destroy_minimal_reference_c2c_plan_arrays_();
        minimal_reference_c2c_plan_arrays_.clear();

        destroy_raw_c2c_plan_array_bundles_();
        for ( auto &bundle : c2c_plan_array_bundles_ )
        {
            // cuFFT handles are bound to these streams. Destroy the handles
            // first; vector member destruction would otherwise do the reverse.
            bundle.plans.clear();
            bundle.handles.clear();
            bundle.context_handles.clear();
            bundle.streams.clear();
            bundle.external_streams.clear();
            bundle.offsets.clear();
        }
        c2c_plan_array_bundles_.clear();

        for ( auto &bundle : r2c_c2r_plan_bundles_ )
        {
            bundle.forward_plans.clear();
            bundle.inverse_plans.clear();
            bundle.streams.clear();
        }
        r2c_c2r_plan_bundles_.clear();

        container_.clear();
        for ( const auto &entry : owned_plan_streams_ )
            runtime_api::destroy_stream( entry.second );
        owned_plan_streams_.clear();
        if ( !work_area_.is_free() )
            work_area_.free();

        work_area_size_    = 0;
        activated_         = false;
        external_activated_ = false;
        hot_exec_no_sync_  = false;
        update_memory_profile_();
    }

    void release_noexcept_() noexcept
    {
        try
        {
            release_();
        }
        catch ( ... )
        {
        }
    }

    static std::size_t aligned_plan_bundle_work_stride_( std::size_t bytes )
    {
        if ( bytes == 0 )
            return 0;
        const std::size_t alignment = 256;
        if ( bytes > std::numeric_limits<std::size_t>::max() - ( alignment - 1 ) )
            throw std::overflow_error( "fft_wrap_many: R2C/C2R plan-bundle work stride overflow" );
        return ( ( bytes + alignment - 1 ) / alignment ) * alignment;
    }

    static std::size_t r2c_c2r_plan_bundle_work_size_( const r2c_c2r_plan_bundle_t &bundle )
    {
        const std::size_t stride = aligned_plan_bundle_work_stride_( bundle.work_stride_bytes );
        if ( stride != 0 && bundle.streams.size() > std::numeric_limits<std::size_t>::max() / stride )
            throw std::overflow_error( "fft_wrap_many: R2C/C2R plan-bundle total work size overflow" );
        return stride * bundle.streams.size();
    }

    void bind_r2c_c2r_plan_bundle_work_areas_( void *base_work_area )
    {
        char *raw = static_cast<char *>( base_work_area );
        for ( auto &bundle : r2c_c2r_plan_bundles_ )
        {
            const std::size_t stride = aligned_plan_bundle_work_stride_( bundle.work_stride_bytes );
            for ( std::size_t lane = 0; lane < bundle.streams.size(); ++lane )
            {
                void *lane_work = raw == nullptr ? nullptr : static_cast<void *>( raw + lane * stride );
                bundle.forward_plans[lane]->set_work_area( lane_work );
                bundle.inverse_plans[lane]->set_work_area( lane_work );
            }
            bundle.bound_work_stride_bytes = stride;
        }
    }

    r2c_c2r_plan_bundle_t &r2c_c2r_plan_bundle_( r2c_c2r_plan_bundle_id_t bundle_id )
    {
        if ( bundle_id == invalid_r2c_c2r_plan_bundle_id() || bundle_id >= r2c_c2r_plan_bundles_.size() )
            throw std::out_of_range( "fft_wrap_many: invalid R2C/C2R plan-bundle id" );
        return r2c_c2r_plan_bundles_[bundle_id];
    }

    const r2c_c2r_plan_bundle_t &r2c_c2r_plan_bundle_( r2c_c2r_plan_bundle_id_t bundle_id ) const
    {
        if ( bundle_id == invalid_r2c_c2r_plan_bundle_id() || bundle_id >= r2c_c2r_plan_bundles_.size() )
            throw std::out_of_range( "fft_wrap_many: invalid R2C/C2R plan-bundle id" );
        return r2c_c2r_plan_bundles_[bundle_id];
    }

    const c2c_plan_array_bundle_t &c2c_plan_array_bundle_( c2c_plan_array_id_t bundle_id ) const
    {
        if ( bundle_id == invalid_c2c_plan_array_id() || bundle_id >= c2c_plan_array_bundles_.size() )
        {
            throw std::out_of_range( "fft_wrap_many: invalid C2C plan-array id" );
        }
        return c2c_plan_array_bundles_[bundle_id];
    }

    template <::fftm::direction D, class InPtr, class OutPtr>
    void exec_typed_plan_sequence_offsets_(
        const typed_plan_sequence_t<D> &sequence, const std::vector<std::size_t> &offsets, InPtr in, OutPtr out
    )
    {
        if ( sequence.size() != offsets.size() )
        {
            throw std::logic_error( "fft_wrap_many::exec_plan_sequence_offsets_tight: plan/offset count mismatch." );
        }
        for ( std::size_t i = 0; i < sequence.size(); ++i )
        {
            sequence[i]->exec_typed(
                static_cast<void *>( in + offsets[i] ), static_cast<void *>( out + offsets[i] )
            );
        }
    }

    template <::fftm::direction D, class InPtr, class OutPtr>
    void exec_typed_plan_sequence_offsets_direction_(
        const typed_plan_sequence_t<D> &sequence, const std::vector<std::size_t> &offsets,
        ::fftm::direction exec_dir, InPtr in, OutPtr out
    )
    {
        if ( sequence.size() != offsets.size() )
        {
            throw std::logic_error(
                "fft_wrap_many::exec_plan_sequence_offsets_tight_direction: plan/offset count mismatch."
            );
        }
        for ( std::size_t i = 0; i < sequence.size(); ++i )
        {
            sequence[i]->exec_typed_direction(
                exec_dir, static_cast<void *>( in + offsets[i] ), static_cast<void *>( out + offsets[i] )
            );
        }
    }

    work_array_t                                   work_area_;
    std::size_t                                    work_area_size_;
    bool                                           activated_, external_activated_;
    bool                                           hot_exec_no_sync_;
    std::map<std::string, std::unique_ptr<wrap_t>> container_;
    std::map<std::string, typename runtime_api::stream_t>       owned_plan_streams_;
    std::vector<plan_sequence_t>                   plan_sequences_;
    std::vector<typed_plan_sequence_t<::fftm::direction::C2CF>> c2cf_plan_sequences_;
    std::vector<typed_plan_sequence_t<::fftm::direction::C2CB>> c2cb_plan_sequences_;
    std::vector<opaque_c2c_plan_sequence_t>                    opaque_c2c_plan_sequences_;
    std::vector<c2c_plan_array_bundle_t>                        c2c_plan_array_bundles_;
    std::vector<r2c_c2r_plan_bundle_t>                          r2c_c2r_plan_bundles_;
    std::vector<minimal_reference_c2c_plan_array_handle_t>       minimal_reference_c2c_plan_arrays_;
    std::vector<typed_plan_sequence_kind>                      plan_sequence_kinds_;
    std::vector<std::size_t>                                   plan_sequence_typed_indices_;
    std::vector<std::size_t>                                   plan_sequence_opaque_indices_;
    memory_profiler_t                             *memory_profiler_ = nullptr;
    std::string                                    memory_profile_prefix_;
};

} // namespace wrap
} // namespace fftm

#endif
