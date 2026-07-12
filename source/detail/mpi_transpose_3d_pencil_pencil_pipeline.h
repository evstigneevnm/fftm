#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_PIPELINE_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_PIPELINE_H__

#include <stdexcept>

#include <scfd/utils/safe_call.h>

#include "../profiling.h"
#include "mpi_transpose_3d.h"

namespace fftm
{
namespace detail
{

template <class BaseFFT, class SameX, class SameZ, class OptionalProfiler>
class mpi_transpose_3d_pencil_pencil_pipeline
{
public:
    mpi_transpose_3d_pencil_pencil_pipeline(
        BaseFFT &base_fft, SameX &same_x, SameZ &same_z, OptionalProfiler &profiler
    )
        : base_fft_( base_fft ), same_x_( same_x ), same_z_( same_z ), profiler_( profiler )
    {
    }

    void init( mpi_transpose_3d_mode mode, bool use_persistent_p2p )
    {
        auto scope         = profiler_.scoped_tic( "mpi_transpose_3d_pencil_pencil_pipeline::init" );
        mode_              = mode;
        use_persistent_p2p_ = use_persistent_p2p;
        is_inited_         = true;
    }

    void reset()
    {
        is_inited_ = false;
    }

    template <
        class RealArray3, class Stage0Complex3, class Stage1Complex3, class XFastComplex3, class XFFTComplex3,
        class ComplexArray3>
    void forward(
        const RealArray3 &in, Stage0Complex3 &stage0, Stage1Complex3 &stage1, XFastComplex3 &stage1_xfast,
        XFFTComplex3 &x_fft_stage, ComplexArray3 &out, mpi_transpose_3d_mode transpose_mode
    )
    {
        /*
         * Layout contract for the fused pencil-pencil path:
         *   real input -> stage0: Z R2C output in Y-fast local layout;
         *   stage0 -> stage1: same-X redistribution to XZY optimized layout;
         *   stage1 -> stage1_xfast: Y FFT output in X-fast layout;
         *   stage1_xfast -> x_fft_stage: same-Z redistribution directly into X-FFT layout;
         *   x_fft_stage -> output: final X FFT output in public spectral layout.
         *
         * All backend-specific operations stay behind BaseFFT/SameX/SameZ.
         */
        ensure_inited_( transpose_mode );
        typename OptionalProfiler::scoped_ticker profile_scope =
            profiler_.scoped_tic( "fftm::forward_3d_pencil_pencil_fused_pipeline" );
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "pencil_pipeline/forward_z_fft" );
            SCFD_SAFE_CALL( base_fft_.template exec<RealArray3, Stage0Complex3>( "forward_z", in, stage0 ) );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "pencil_pipeline/forward_same_x" );
            SCFD_SAFE_CALL( same_x_.transpose_xyz_to_xzy_optimized_layout( stage0, stage1, transpose_mode ) );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "pencil_pipeline/forward_y_fft" );
            SCFD_SAFE_CALL(
                base_fft_.template exec<Stage1Complex3, XFastComplex3>( "forward_y", stage1, stage1_xfast )
            );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "pencil_pipeline/forward_same_z_xfast" );
            SCFD_SAFE_CALL( same_z_.transpose_x_to_y_xfast_layout( stage1_xfast, x_fft_stage, transpose_mode ) );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "pencil_pipeline/forward_x_fft" );
            SCFD_SAFE_CALL( base_fft_.template exec<XFFTComplex3, ComplexArray3>( "forward_x", x_fft_stage, out ) );
        }
    }

    template <
        class RealArray3, class Stage0Complex3, class Stage1Complex3, class XFastComplex3, class XFFTComplex3,
        class ComplexArray3>
    void forward_reference(
        const RealArray3 &in, Stage0Complex3 &stage0, Stage1Complex3 &stage1, XFastComplex3 &stage1_xfast,
        XFFTComplex3 &x_fft_stage, ComplexArray3 &out, mpi_transpose_3d_mode transpose_mode
    )
    {
        /*
         * Backend-neutral reference-style entry point. This keeps a distinct
         * profiler/test surface while the transport remains implemented through
         * SameX/SameZ/BaseFFT abstractions. The next optimization step can
         * replace these internal redistribution calls with an owned pencil
         * schedule without changing the public benchmark matrix again.
         */
        ensure_inited_( transpose_mode );
        typename OptionalProfiler::scoped_ticker profile_scope =
            profiler_.scoped_tic( "fftm::forward_3d_pencil_pencil_reference_pipeline" );
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "reference_pencil_pipeline/forward_z_fft" );
            SCFD_SAFE_CALL( base_fft_.template exec<RealArray3, Stage0Complex3>( "forward_z", in, stage0 ) );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "reference_pencil_pipeline/forward_same_x" );
            SCFD_SAFE_CALL( same_x_.transpose_xyz_to_xzy_optimized_layout( stage0, stage1, transpose_mode ) );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "reference_pencil_pipeline/forward_y_fft" );
            SCFD_SAFE_CALL(
                base_fft_.template exec<Stage1Complex3, XFastComplex3>( "forward_y", stage1, stage1_xfast )
            );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "reference_pencil_pipeline/forward_same_z_xfast" );
            SCFD_SAFE_CALL( same_z_.transpose_x_to_y_xfast_layout( stage1_xfast, x_fft_stage, transpose_mode ) );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "reference_pencil_pipeline/forward_x_fft" );
            SCFD_SAFE_CALL( base_fft_.template exec<XFFTComplex3, ComplexArray3>( "forward_x", x_fft_stage, out ) );
        }
    }

    template <
        class ComplexArray3, class XFFTComplex3, class XFastComplex3, class Stage1Complex3, class Stage0Complex3,
        class RealArray3>
    void backward(
        ComplexArray3 &in, XFFTComplex3 &x_fft_stage, XFastComplex3 &stage1_xfast, Stage1Complex3 &stage1,
        Stage0Complex3 &stage0, RealArray3 &out, mpi_transpose_3d_mode transpose_mode
    )
    {
        /*
         * Reverse of the forward contract. The caller binds stage1 to a valid
         * temporary/output buffer before entry when the optimized path aliases it.
         */
        ensure_inited_( transpose_mode );
        typename OptionalProfiler::scoped_ticker profile_scope =
            profiler_.scoped_tic( "fftm::backward_3d_pencil_pencil_fused_pipeline" );
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "pencil_pipeline/backward_x_fft" );
            SCFD_SAFE_CALL( base_fft_.template exec<ComplexArray3, XFFTComplex3>( "inverse_x", in, x_fft_stage ) );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "pencil_pipeline/backward_same_z_xfast" );
            SCFD_SAFE_CALL( same_z_.transpose_y_to_x_xfast_layout( x_fft_stage, stage1_xfast, transpose_mode ) );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "pencil_pipeline/backward_y_fft" );
            SCFD_SAFE_CALL(
                base_fft_.template exec<XFastComplex3, Stage1Complex3>( "inverse_y", stage1_xfast, stage1 )
            );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "pencil_pipeline/backward_same_x" );
            SCFD_SAFE_CALL( same_x_.transpose_xzy_to_xyz_optimized_layout( stage1, stage0, transpose_mode ) );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "pencil_pipeline/backward_z_fft" );
            SCFD_SAFE_CALL( base_fft_.template exec<Stage0Complex3, RealArray3>( "inverse_z", stage0, out ) );
        }
    }

    template <
        class ComplexArray3, class XFFTComplex3, class XFastComplex3, class Stage1Complex3, class Stage0Complex3,
        class RealArray3>
    void backward_reference(
        ComplexArray3 &in, XFFTComplex3 &x_fft_stage, XFastComplex3 &stage1_xfast, Stage1Complex3 &stage1,
        Stage0Complex3 &stage0, RealArray3 &out, mpi_transpose_3d_mode transpose_mode
    )
    {
        ensure_inited_( transpose_mode );
        typename OptionalProfiler::scoped_ticker profile_scope =
            profiler_.scoped_tic( "fftm::backward_3d_pencil_pencil_reference_pipeline" );
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "reference_pencil_pipeline/backward_x_fft" );
            SCFD_SAFE_CALL( base_fft_.template exec<ComplexArray3, XFFTComplex3>( "inverse_x", in, x_fft_stage ) );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "reference_pencil_pipeline/backward_same_z_xfast" );
            SCFD_SAFE_CALL( same_z_.transpose_y_to_x_xfast_layout( x_fft_stage, stage1_xfast, transpose_mode ) );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "reference_pencil_pipeline/backward_y_fft" );
            SCFD_SAFE_CALL(
                base_fft_.template exec<XFastComplex3, Stage1Complex3>( "inverse_y", stage1_xfast, stage1 )
            );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "reference_pencil_pipeline/backward_same_x" );
            SCFD_SAFE_CALL( same_x_.transpose_xzy_to_xyz_optimized_layout( stage1, stage0, transpose_mode ) );
        }
        {
            typename OptionalProfiler::scoped_ticker phase =
                profiler_.scoped_tic( "reference_pencil_pipeline/backward_z_fft" );
            SCFD_SAFE_CALL( base_fft_.template exec<Stage0Complex3, RealArray3>( "inverse_z", stage0, out ) );
        }
    }

private:
    void ensure_inited_( mpi_transpose_3d_mode mode ) const
    {
        if ( !is_inited_ )
        {
            throw std::logic_error( "mpi_transpose_3d_pencil_pencil_pipeline is not initialized" );
        }
        if ( mode_ != mode )
        {
            throw std::logic_error( "mpi_transpose_3d_pencil_pencil_pipeline mode mismatch" );
        }
        (void)use_persistent_p2p_;
    }

    BaseFFT          &base_fft_;
    SameX            &same_x_;
    SameZ            &same_z_;
    OptionalProfiler &profiler_;
    mpi_transpose_3d_mode mode_               = mpi_transpose_3d_mode::alltoallv;
    bool                  use_persistent_p2p_ = false;
    bool                  is_inited_          = false;
};

} // namespace detail
} // namespace fftm

#endif
