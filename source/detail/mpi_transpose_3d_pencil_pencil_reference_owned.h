#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_H__

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/utils/safe_call.h>
#include <scfd/utils/system_timer_event.h>

#include "../fft_direction.h"
#include "../fft_partitioning.h"
#include "../profiling.h"
#include "mpi_transpose_3d.h"
#include "mpi_transpose_3d_pencil_pencil_plan_state.h"

namespace fftm
{

struct fftm_native_stage_timing
{
    std::string stage;
    double      ms = 0.0;
};

namespace detail
{

template <class BaseFFT, class ValueType, class Backend, class MPIComm, class Log, class RuntimeAPI, class OptionalProfiler>
class mpi_transpose_3d_pencil_pencil_reference_owned
{
public:
    using base_fft_t        = BaseFFT;
    using value_type        = ValueType;
    using backend_t         = Backend;
    using memory_t          = typename backend_t::memory_type;
    using host_memory_t     = typename memory_t::host_memory_type;
    using runtime_api_t     = RuntimeAPI;
    using partition_t       = ::fftm::partition;
    using plan_state_t      = ::fftm::detail::mpi_transpose_3d_pencil_pencil_plan_state;
    using mpi_comm_t        = scfd::communication::mpi_comm;
    using contiguous_buf_t  = scfd::arrays::array_nd<value_type, 1, memory_t>;
    using host_buf_t        = scfd::arrays::array_nd<value_type, 1, host_memory_t>;
    using mpi_request_t     = scfd::communication::detail::mpi_request;
    using mpi_dtype_t       = scfd::communication::detail::mpi_data_type;
    using profiler_t        = ::fftm::fftm_profiler;
    using profiler_scope_t  = ::fftm::profile_scope<profiler_t>;
    using memory_profiler_t = ::fftm::fftm_memory_profiler;

private:
    enum class deferred_send_stage
    {
        none,
        forward_first,
        forward_second,
        backward_second,
        backward_first
    };

    enum class deferred_send_communicator
    {
        row,
        line
    };

    struct deferred_send_state
    {
        deferred_send_stage        stage = deferred_send_stage::none;
        deferred_send_communicator communicator = deferred_send_communicator::row;
        std::vector<mpi_request_t> requests;
        int                        remaining = 0;

        bool active() const
        {
            return stage != deferred_send_stage::none;
        }

        void reset()
        {
            stage     = deferred_send_stage::none;
            remaining = 0;
            requests.clear();
        }
    };

public:
    using plan_sequence_id_t = typename base_fft_t::plan_sequence_id_t;
    using c2c_plan_array_id_t = typename base_fft_t::c2c_plan_array_id_t;

    struct large_byte_datatype_cache
    {
        std::vector<mpi_dtype_t>   datatypes;
        std::vector<std::size_t>   bytes;
        std::vector<int>           blocks;
        int                        active_count = 0;
        int                        total_blocks = 0;
    };

    struct native_schedule_slot
    {
        std::size_t offset_elems = 0;
        std::size_t elems        = 0;
        int         mpi_peer     = 0;
        int         mpi_tag      = 0;
        bool        valid        = false;
    };

    mpi_transpose_3d_pencil_pencil_reference_owned(
        base_fft_t &base_fft, const MPIComm &mpi, const Log &log, OptionalProfiler &profiler
    )
        : base_fft_( base_fft ), mpi_( mpi ), log_( log ), optional_profiler_( profiler )
    {
        static_assert(
            std::is_same<memory_t, typename runtime_api_t::memory_type>::value,
            "mpi_transpose_3d_pencil_pencil_reference_owned requires matching backend/runtime memory types"
        );
    }

    ~mpi_transpose_3d_pencil_pencil_reference_owned()
    {
        drain_deferred_send_noexcept_();
        free_persistent_send_states_();
        free_large_byte_datatype_caches_();
        free_forward_direct_recvtypes_();
        free_value_type_();
    }

    void quiesce_for_resource_release()
    {
        drain_deferred_send_noexcept_();
        free_persistent_send_states_();
        free_large_byte_datatype_caches_();
        free_forward_direct_recvtypes_();
    }

#include "mpi_transpose_3d_pencil_pencil_reference_owned_configuration_api.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_workspace_api.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_initialization.inc"

public:
#include "mpi_transpose_3d_pencil_pencil_reference_owned_pipeline_execution.inc"

private:
    profiler_scope_t profile_scope_( const std::string &name )
    {
        return profiler_scope_t( profiler_, name );
    }

#include "mpi_transpose_3d_pencil_pencil_reference_owned_stage_diagnostics.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_local_fft_execution.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_local_fft_diagnostics.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_workspace_binding.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_local_fft_synchronization.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_plan_schedule.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_initialization_validation.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_transport_policy_datatypes.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_transport_synchronization.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_schedule_reporting.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_transport_posting.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_deferred_send.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_request_state.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_redistribution_extents.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_first_transpose_schedule.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_second_transpose_schedule.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_first_transpose_copy.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_second_transpose_copy.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_first_transpose_forward.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_second_transpose_forward.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_second_transpose_backward.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_first_transpose_backward.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_configuration_state.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_local_fft_state.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_layout_workspace_state.inc"

#include "mpi_transpose_3d_pencil_pencil_reference_owned_transport_state.inc"
};

#include "mpi_transpose_3d_pencil_pencil_reference_owned_y_diagnostics.h"

} // namespace detail
} // namespace fftm

#endif
