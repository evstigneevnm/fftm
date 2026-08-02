    template <
        class RealArray3, class Stage0Complex3, class Stage1Complex3, class XFastComplex3, class XFFTComplex3,
        class ComplexArray3>
	    void forward(
	        const RealArray3 &in, Stage0Complex3 &stage0, Stage1Complex3 &stage1, XFastComplex3 &stage1_xfast,
	        XFFTComplex3 &x_fft_stage, ComplexArray3 &out, mpi_transpose_3d_mode mode
	    )
    {
        ensure_inited_( mode );
        auto scope = optional_profiler_.scoped_tic(
	            reference_parity_enabled_ ? "fftm::forward_3d_pencil_pencil_reference_parity_pipeline"
	                                  : "fftm::forward_3d_pencil_pencil_reference_owned_pipeline"
	        );
	        exec_local_fft_( "reference_owned/forward_z_fft", "forward_z", in, stage0 );
	        SCFD_SAFE_CALL( forward_from_stage0(
	            stage0, stage1, stage1_xfast, x_fft_stage, out, mode
	        ) );
	    }

#include "mpi_transpose_3d_pencil_pencil_reference_owned_local_fft_plan_lifecycle.inc"

    template <class Stage0Complex3, class Stage1Complex3, class XFastComplex3, class XFFTComplex3, class ComplexArray3>
    void forward_from_stage0(
        Stage0Complex3 &stage0, Stage1Complex3 &stage1, XFastComplex3 &stage1_xfast, XFFTComplex3 &x_fft_stage,
        ComplexArray3 &out, mpi_transpose_3d_mode mode
    )
    {
        ensure_inited_( mode );
		        time_native_stage_( "reference_owned/forward_first_redistribution", [&]() {
		            auto phase = profile_scope_( "reference_owned/forward_first_redistribution" );
		            SCFD_SAFE_CALL( forward_first_( stage0, stage1 ) );
		        } );
        execute_with_deferred_send_overlap_( deferred_send_stage::forward_first, [&]() {
            exec_local_fft_( "reference_owned/forward_y_fft", "forward_y", stage1, stage1_xfast );
        } );
		        time_native_stage_( "reference_owned/forward_second_redistribution", [&]() {
		            auto phase = profile_scope_( "reference_owned/forward_second_redistribution" );
		            if ( native_forward_second_facade_active_() )
		            {
		                SCFD_SAFE_CALL( native_forward_second_xfast_( stage1_xfast, x_fft_stage ) );
		            }
		            else
		            {
		                SCFD_SAFE_CALL( forward_second_xfast_( stage1_xfast, x_fft_stage ) );
		            }
		        } );
        execute_with_deferred_send_overlap_( deferred_send_stage::forward_second, [&]() {
            exec_local_fft_( "reference_owned/forward_x_fft", "forward_x", x_fft_stage, out );
        } );
	    }

    template <
        class ComplexArray3, class XFFTComplex3, class XFastComplex3, class Stage1Complex3, class Stage0Complex3,
        class RealArray3>
    void backward(
        ComplexArray3 &in, XFFTComplex3 &x_fft_stage, XFastComplex3 &stage1_xfast, Stage1Complex3 &stage1,
        Stage0Complex3 &stage0, RealArray3 &out, mpi_transpose_3d_mode mode
    )
    {
        ensure_inited_( mode );
        auto scope = optional_profiler_.scoped_tic(
	            reference_parity_enabled_ ? "fftm::backward_3d_pencil_pencil_reference_parity_pipeline"
	                                  : "fftm::backward_3d_pencil_pencil_reference_owned_pipeline"
	        );
	        exec_local_fft_( "reference_owned/backward_x_fft", "inverse_x", in, x_fft_stage );
	        SCFD_SAFE_CALL( backward_to_stage0(
	            in, x_fft_stage, stage1_xfast, stage1, stage0, mode
	        ) );
        execute_with_deferred_send_overlap_( deferred_send_stage::backward_first, [&]() {
            exec_local_fft_( "reference_owned/backward_z_fft", "inverse_z", stage0, out );
        } );
	    }

    template <class ComplexArray3, class XFFTComplex3, class XFastComplex3, class Stage1Complex3, class Stage0Complex3>
    void backward_to_stage0(
        ComplexArray3 &in, XFFTComplex3 &x_fft_stage, XFastComplex3 &stage1_xfast, Stage1Complex3 &stage1,
        Stage0Complex3 &stage0, mpi_transpose_3d_mode mode
    )
    {
        ensure_inited_( mode );
		        time_native_stage_( "reference_owned/backward_second_redistribution", [&]() {
		            auto phase = profile_scope_( "reference_owned/backward_second_redistribution" );
		            if ( native_backward_second_facade_active_() )
		            {
		                SCFD_SAFE_CALL( native_backward_second_xfast_( x_fft_stage, stage1_xfast ) );
		            }
		            else
		            {
		                SCFD_SAFE_CALL( backward_second_xfast_( x_fft_stage, stage1_xfast ) );
		            }
		        } );
        execute_with_deferred_send_overlap_( deferred_send_stage::backward_second, [&]() {
            exec_local_fft_( "reference_owned/backward_y_fft", "inverse_y", stage1_xfast, stage1 );
        } );
		        time_native_stage_( "reference_owned/backward_first_redistribution", [&]() {
		            auto phase = profile_scope_( "reference_owned/backward_first_redistribution" );
		            SCFD_SAFE_CALL( backward_first_( stage1, stage0 ) );
		        } );
	    }

    template <class Stage0ZFast3, class Stage1ZFast3, class Stage1YFast3, class ComplexArray3>
    void forward_native_opt0_zfast(
        Stage0ZFast3 &stage0, Stage1ZFast3 &stage1, Stage1YFast3 &stage1_yfft, ComplexArray3 &out,
        const std::vector<std::string> &forward_y_plan_names, const std::vector<std::size_t> &y_plan_offsets,
        mpi_transpose_3d_mode mode
    )
    {
        ensure_inited_( mode );
        if ( !native_opt0_default_z_layout_ )
            throw std::logic_error( "forward_native_opt0_zfast called without native opt0 layout enabled" );
        auto scope = optional_profiler_.scoped_tic( "fftm::forward_3d_pencil_pencil_native_opt0_zfast_pipeline" );
        time_native_stage_( "reference_owned/forward_first_redistribution", [&]() {
            auto phase = profile_scope_( "reference_owned/forward_first_redistribution" );
            SCFD_SAFE_CALL( forward_first_( stage0, stage1 ) );
        } );
        execute_with_deferred_send_overlap_( deferred_send_stage::forward_first, [&]() {
            if ( native_opt0_y_plan_array_bundle_active_() )
            {
                exec_local_fft_plan_array_bundle_(
                    "reference_owned/forward_y_fft", "forward_y_native_opt0_zfast", stage1, stage1_yfft,
                    direction::C2CF
                );
            }
            else
            {
                exec_local_fft_many_offsets_(
                    "reference_owned/forward_y_fft", "forward_y_native_opt0_zfast", forward_y_plan_names,
                    y_plan_offsets, stage1, stage1_yfft, native_opt0_forward_y_plan_sequence_,
                    use_native_opt0_shared_y_plan_handles_, direction::C2CF
                );
            }
        } );
	        time_native_stage_( "reference_owned/forward_second_redistribution", [&]() {
	            auto phase = profile_scope_( "reference_owned/forward_second_redistribution" );
	            SCFD_SAFE_CALL( forward_second_xfast_( stage1_yfft, out ) );
	        } );
        execute_with_deferred_send_overlap_( deferred_send_stage::forward_second, [&]() {
            exec_local_fft_( "reference_owned/forward_x_fft", "forward_x", out, out );
        } );
    }

    template <class Stage0ZFast3, class Stage1ZFast3, class Stage1YFast3, class XStage3, class ComplexArray3>
    void forward_native_opt0_zfast_reference_buffers(
        Stage0ZFast3 &stage0, Stage1ZFast3 &stage1, Stage1YFast3 &stage1_yfft, XStage3 &x_stage,
        ComplexArray3 &out, const std::vector<std::string> &forward_y_plan_names,
        const std::vector<std::size_t> &y_plan_offsets, mpi_transpose_3d_mode mode
    )
    {
        ensure_inited_( mode );
        if ( !native_opt0_reference_y_buffer_topology_enabled_() )
            throw std::logic_error(
                "forward_native_opt0_zfast_reference_buffers called without native opt0 reference Y-buffer topology enabled"
            );
        auto scope = optional_profiler_.scoped_tic(
            "fftm::forward_3d_pencil_pencil_native_opt0_reference_y_buffer_topology"
        );
	        debug_native_opt0_topology_marker_( "forward:start" );
	        append_native_opt0_forward_layout_offsets_(
	            stage0, stage1, stage1_yfft, x_stage, out, y_plan_offsets
	        );
	        time_native_stage_( "reference_owned/forward_first_redistribution", [&]() {
	            debug_native_opt0_topology_marker_( "forward:first_transpose:begin" );
	            auto phase = profile_scope_( "reference_owned/forward_first_redistribution" );
	            SCFD_SAFE_CALL( forward_first_( stage0, stage1 ) );
	            debug_native_opt0_topology_marker_( "forward:first_transpose:end" );
        } );
        debug_native_opt0_topology_marker_( "forward:y_fft:begin" );
        execute_with_deferred_send_overlap_( deferred_send_stage::forward_first, [&]() {
            if ( native_opt0_y_plan_array_bundle_active_() )
            {
                exec_local_fft_plan_array_bundle_(
                    "reference_owned/forward_y_fft", "forward_y_native_opt0_zfast", stage1, stage1_yfft,
                    direction::C2CF
                );
            }
            else
            {
                exec_local_fft_many_offsets_(
                    "reference_owned/forward_y_fft", "forward_y_native_opt0_zfast", forward_y_plan_names,
                    y_plan_offsets, stage1, stage1_yfft, native_opt0_forward_y_plan_sequence_,
                    use_native_opt0_shared_y_plan_handles_, direction::C2CF
                );
            }
        } );
        debug_native_opt0_topology_marker_( "forward:y_fft:end" );
	        time_native_stage_( "reference_owned/forward_second_redistribution", [&]() {
	            debug_native_opt0_topology_marker_( "forward:second_transpose:begin" );
	            auto phase = profile_scope_( "reference_owned/forward_second_redistribution" );
	            SCFD_SAFE_CALL( forward_second_xfast_( stage1_yfft, x_stage ) );
	            debug_native_opt0_topology_marker_( "forward:second_transpose:end" );
	        } );
        debug_native_opt0_topology_marker_( "forward:x_fft:begin" );
        execute_with_deferred_send_overlap_( deferred_send_stage::forward_second, [&]() {
            exec_local_fft_( "reference_owned/forward_x_fft", "forward_x", x_stage, out );
        } );
        debug_native_opt0_topology_marker_( "forward:x_fft:end" );
    }

    template <class ComplexArray3, class XFFTZFast3, class Stage1YFast3, class Stage1ZFast3, class Stage0ZFast3>
    void backward_native_opt0_zfast(
        ComplexArray3 &in, XFFTZFast3 &x_fft_stage, Stage1YFast3 &stage1_yfft, Stage1ZFast3 &stage1,
        Stage0ZFast3 &stage0, const std::vector<std::string> &inverse_y_plan_names,
        const std::vector<std::size_t> &y_plan_offsets, mpi_transpose_3d_mode mode
    )
    {
        ensure_inited_( mode );
	        if ( !native_opt0_default_z_layout_ )
	            throw std::logic_error( "backward_native_opt0_zfast called without native opt0 layout enabled" );
	        auto scope = optional_profiler_.scoped_tic( "fftm::backward_3d_pencil_pencil_native_opt0_zfast_pipeline" );
	        append_native_opt0_backward_layout_offsets_(
	            in, x_fft_stage, stage1_yfft, stage1, stage0, y_plan_offsets
	        );
	        exec_local_fft_( "reference_owned/backward_x_fft", "inverse_x", in, x_fft_stage );
	        time_native_stage_( "reference_owned/backward_second_redistribution", [&]() {
	            auto phase = profile_scope_( "reference_owned/backward_second_redistribution" );
	            SCFD_SAFE_CALL( backward_second_xfast_( x_fft_stage, stage1_yfft ) );
	        } );
        execute_with_deferred_send_overlap_( deferred_send_stage::backward_second, [&]() {
            if ( native_opt0_y_plan_array_bundle_active_() )
            {
                exec_local_fft_plan_array_bundle_(
                    "reference_owned/backward_y_fft", "inverse_y_native_opt0_zfast", stage1_yfft, stage1,
                    direction::C2CB
                );
            }
            else
            {
                exec_local_fft_many_offsets_(
                    "reference_owned/backward_y_fft", "inverse_y_native_opt0_zfast", inverse_y_plan_names,
                    y_plan_offsets, stage1_yfft, stage1, native_opt0_inverse_y_plan_sequence_,
                    use_native_opt0_shared_y_plan_handles_, direction::C2CB
                );
            }
        } );
	        time_native_stage_( "reference_owned/backward_first_redistribution", [&]() {
	            auto phase = profile_scope_( "reference_owned/backward_first_redistribution" );
	            SCFD_SAFE_CALL( backward_first_( stage1, stage0 ) );
	        } );
    }
