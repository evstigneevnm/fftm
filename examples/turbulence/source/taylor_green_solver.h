#ifndef FFTM_EXAMPLES_TURBULENCE_TAYLOR_GREEN_SOLVER_H
#define FFTM_EXAMPLES_TURBULENCE_TAYLOR_GREEN_SOLVER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include <scfd/functional/basic_ops.h>
#include <scfd/static_vec/rect.h>
#include <scfd/utils/system_timer_event.h>

#include "snapshot_writer.h"
#include "taylor_green_kernels.h"
#include "turbulence_options.h"

namespace fftm
{
namespace examples
{
namespace turbulence
{

template <class Array>
inline scfd::static_vec::rect<int, 3> array_range_3d( const Array &array )
{
    const auto size = array.size_nd();
    return scfd::static_vec::rect<int, 3>(
        index3_t( 0, 0, 0 ),
        index3_t( static_cast<int>( size[0] ), static_cast<int>( size[1] ), static_cast<int>( size[2] ) )
    );
}

template <class Real>
struct turbulence_diagnostics
{
    Real kinetic_energy = 0;
    Real enstrophy      = 0;
    Real max_speed      = 0;
    Real divergence_rms = 0;
    Real cfl            = 0;
};

template <class FFTM>
class taylor_green_solver
{
public:
    using fftm_t          = FFTM;
    using real_t          = typename fftm_t::real;
    using complex_t       = typename fftm_t::complex;
    using backend_t       = typename fftm_t::backend_type;
    using memory_t        = typename fftm_t::memory_t;
    using runtime_api_t   = typename fftm_t::runtime_api_t;
    using real_array_t    = typename fftm_t::template real_array_t<3>;
    using complex_array_t = typename fftm_t::template complex_array_t<3>;
    using for_each_t      = typename backend_t::template for_each_nd_type<3, int>;
    using reduce_t        = typename backend_t::reduce_type;
    using snapshot_writer_t = snapshot_writer<memory_t>;

    template <class Comm, class Log>
    taylor_green_solver(
        fftm_t &fft, const Comm &comm, Log &log, const turbulence_options &options, int input_start_x,
        int input_start_y, int spectral_start_y, int spectral_start_z
    )
        : fft_( fft ), options_( options ), input_start_x_( input_start_x ), input_start_y_( input_start_y )
    {
        const auto input_sizes  = fft_.get_local_input_sizes();
        const auto output_sizes = fft_.get_local_output_sizes();
        input_nx_ = std::get<0>( input_sizes );
        input_ny_ = std::get<1>( input_sizes );
        input_nz_ = std::get<2>( input_sizes );
        output_nx_ = std::get<0>( output_sizes );
        output_ny_ = std::get<1>( output_sizes );
        output_nz_ = std::get<2>( output_sizes );

        coordinates_.nx = static_cast<int>( options_.nx );
        coordinates_.ny = static_cast<int>( options_.ny );
        coordinates_.nz = static_cast<int>( options_.nz );
        coordinates_.y_start = spectral_start_y;
        coordinates_.z_start = spectral_start_z;
        coordinates_.xyz_layout =
            fftm_t::strategy_family_3d == transform_strategy_3d::pencil_pencil &&
            fftm_t::strategy_3d_optimized_layout;

        for_each_.block_size = 128;
        allocate_fields_( comm, log );

        if ( comm.myid == 0 )
            create_directories( options_.output_dir );
        comm.barrier();
        initialize_velocity_();

        if ( options_.snapshot_size != 0 )
        {
            snapshot_writer_.reset( new snapshot_writer_t(
                comm, options_.output_dir, options_.nx, options_.ny, options_.nz, options_.snapshot_size,
                static_cast<std::size_t>( input_start_x_ ), static_cast<std::size_t>( input_start_y_ ), input_nx_,
                input_ny_
            ) );
        }
        if ( options_.viz_snapshot_size != 0 && options_.viz_snapshot_size != options_.snapshot_size )
        {
            const std::string visualization_dir = options_.output_dir + "/visualization";
            if ( comm.myid == 0 )
                create_directories( visualization_dir );
            comm.barrier();
            visualization_writer_.reset( new snapshot_writer_t(
                comm, visualization_dir, options_.nx, options_.ny, options_.nz, options_.viz_snapshot_size,
                static_cast<std::size_t>( input_start_x_ ), static_cast<std::size_t>( input_start_y_ ), input_nx_,
                input_ny_
            ) );
        }

        if ( comm.myid == 0 )
        {
            diagnostics_stream_.open( ( options_.output_dir + "/diagnostics.csv" ).c_str() );
            if ( !diagnostics_stream_ )
                throw std::runtime_error( "Failed to create diagnostics.csv" );
            diagnostics_stream_ << "step,time,dt,kinetic_energy,enstrophy,max_speed,divergence_rms,cfl,step_wall_ms\n";

            if ( options_.target_cfl > 0.0 )
            {
                adaptive_timestep_stream_.open(
                    ( options_.output_dir + "/adaptive_timesteps.csv" ).c_str()
                );
                if ( !adaptive_timestep_stream_ )
                    throw std::runtime_error( "Failed to create adaptive_timesteps.csv" );
                adaptive_timestep_stream_
                    << "step,time,dt,max_speed_start,cfl_start,step_wall_ms\n";
            }
        }
    }

    template <class Comm, class Log>
    void run( const Comm &comm, Log &log )
    {
        const bool adaptive = options_.target_cfl > 0.0;
        double     time     = 0.0;
        std::size_t step    = 0;
        real_t previous_dt  = static_cast<real_t>( options_.dt );
        std::size_t next_snapshot_index = 1;
        std::size_t next_viz_index       = 1;

        auto initial = compute_diagnostics_( comm, real_t( 0 ) );
        validate_initial_state_( initial );
        write_diagnostics_( comm, 0, time, real_t( 0 ), initial, real_t( 0 ) );

        if ( options_.write_initial_snapshot )
        {
            if ( options_.snapshot_every != 0 || options_.snapshot_period > 0.0 )
                write_snapshot_( comm, "omega_z", 0, time );
            if ( options_.viz_every != 0 || options_.viz_period > 0.0 )
            {
                write_snapshot_( comm, "vorticity_magnitude", 0, time );
                write_snapshot_( comm, "q_criterion", 0, time );
            }
        }

        real_t total_wall_ms = real_t( 0 );
        while ( true )
        {
            if ( options_.steps != 0 )
            {
                if ( step >= options_.steps )
                    break;
            }
            else if ( time_reached_( time, options_.final_time ) )
                break;

            ++step;
            const double next_snapshot_time = options_.snapshot_period > 0.0
                ? static_cast<double>( next_snapshot_index ) * options_.snapshot_period
                : std::numeric_limits<double>::infinity();
            const double next_viz_time = options_.viz_period > 0.0
                ? static_cast<double>( next_viz_index ) * options_.viz_period
                : std::numeric_limits<double>::infinity();
            double event_limit = std::numeric_limits<double>::infinity();
            if ( options_.steps == 0 )
                event_limit = options_.final_time - time;
            event_limit = std::min( event_limit, next_snapshot_time - time );
            event_limit = std::min( event_limit, next_viz_time - time );
            if ( !std::isfinite( event_limit ) )
                event_limit = static_cast<double>( options_.dt );
            if ( event_limit <= 0.0 )
                throw std::runtime_error( "Taylor-Green output scheduler produced a nonpositive step" );

            runtime_api_t::device_synchronize();
            scfd::utils::system_timer_event start, finish;
            start.record();
            real_t controller_max_speed = real_t( 0 );
            real_t controller_cfl       = real_t( 0 );
            real_t step_dt              = real_t( 0 );
            if ( adaptive )
            {
                real_t controller_limit = step == 1
                    ? static_cast<real_t>( options_.dt )
                    : previous_dt * static_cast<real_t>( options_.dt_growth );
                if ( options_.dt_max > 0.0 )
                {
                    controller_limit =
                        std::min( controller_limit, static_cast<real_t>( options_.dt_max ) );
                }
                const auto adaptive_step = advance_adaptive_low_storage_rk3_(
                    comm, controller_limit, static_cast<real_t>( event_limit )
                );
                step_dt              = adaptive_step.dt;
                controller_max_speed = adaptive_step.max_speed;
                controller_cfl       = adaptive_step.cfl;
            }
            else
            {
                step_dt = std::min(
                    static_cast<real_t>( options_.dt ), static_cast<real_t>( event_limit )
                );
                advance_low_storage_rk3_( step_dt );
            }
            runtime_api_t::device_synchronize();
            finish.record();
            const real_t step_wall_ms = static_cast<real_t>( finish.elapsed_time( start ) );
            total_wall_ms += step_wall_ms;
            previous_dt = step_dt;
            time += static_cast<double>( step_dt );
            snap_time_to_target_( time, next_snapshot_time );
            snap_time_to_target_( time, next_viz_time );
            if ( options_.steps == 0 )
                snap_time_to_target_( time, options_.final_time );

            if ( adaptive )
            {
                write_adaptive_timestep_(
                    comm, step, time, step_dt, controller_max_speed, controller_cfl, step_wall_ms
                );
            }

            const bool final_step = options_.steps != 0
                ? step >= options_.steps
                : time_reached_( time, options_.final_time );
            const bool diagnostics_due =
                options_.diagnostics_every != 0 &&
                ( step % options_.diagnostics_every == 0 || final_step );
            if ( diagnostics_due )
            {
                const auto values = compute_diagnostics_( comm, step_dt );
                write_diagnostics_( comm, step, time, step_dt, values, step_wall_ms );
                validate_runtime_state_( values, step );
            }

            if ( options_.snapshot_every != 0 && step % options_.snapshot_every == 0 )
                write_snapshot_( comm, "omega_z", step, time );
            if ( options_.viz_every != 0 && step % options_.viz_every == 0 )
            {
                write_snapshot_( comm, "vorticity_magnitude", step, time );
                write_snapshot_( comm, "q_criterion", step, time );
            }

            if ( options_.snapshot_period > 0.0 && time_reached_( time, next_snapshot_time ) )
            {
                write_snapshot_( comm, "omega_z", step, next_snapshot_time );
                ++next_snapshot_index;
            }
            if ( options_.viz_period > 0.0 && time_reached_( time, next_viz_time ) )
            {
                write_snapshot_( comm, "vorticity_magnitude", step, next_viz_time );
                write_snapshot_( comm, "q_criterion", step, next_viz_time );
                ++next_viz_index;
            }
        }

        const real_t global_wall_ms = comm.all_reduce_max( total_wall_ms );
        if ( comm.myid == 0 )
        {
            log.info_f(
                "TAYLOR_GREEN_RESULT size=%zux%zux%zu ranks=%d reynolds=%.9g steps=%zu final_time=%.9g "
                "total_wall_ms=%.9g avg_step_ms=%.9g output=%s",
                options_.nx, options_.ny, options_.nz, comm.num_procs, options_.reynolds, step,
                time, static_cast<double>( global_wall_ms ),
                static_cast<double>( step == 0 ? real_t( 0 ) : global_wall_ms / step ),
                options_.output_dir.c_str()
            );
        }
    }

private:
    struct adaptive_step_result
    {
        real_t dt        = real_t( 0 );
        real_t max_speed = real_t( 0 );
        real_t cfl       = real_t( 0 );
    };

    static double time_tolerance_( double target )
    {
        return 64.0 * std::numeric_limits<double>::epsilon() *
            std::max( 1.0, std::abs( target ) );
    }

    static bool time_reached_( double time, double target )
    {
        return std::isfinite( target ) && time + time_tolerance_( target ) >= target;
    }

    static void snap_time_to_target_( double &time, double target )
    {
        if ( std::isfinite( target ) && std::abs( time - target ) <= time_tolerance_( target ) )
            time = target;
    }

    real_t min_grid_spacing_() const
    {
        return real_t( 6.283185307179586476925286766559 ) /
            static_cast<real_t>( std::max( options_.nx, std::max( options_.ny, options_.nz ) ) );
    }

    template <class Comm, class Log>
    void allocate_fields_( const Comm &comm, Log &log )
    {
        const std::size_t input_elements  = input_nx_ * input_ny_ * input_nz_;
        const std::size_t output_elements = output_nx_ * output_ny_ * output_nz_;
        if ( input_elements > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
        {
            throw std::logic_error(
                "Local physical array exceeds the current SCFD reduction index range; use more MPI ranks"
            );
        }

        const std::size_t application_bytes =
            7 * input_elements * sizeof( real_t ) + 9 * output_elements * sizeof( complex_t );
        const auto memory_info = backend_t::get_device_memory_info();
        if ( memory_info.free_bytes_known && application_bytes > memory_info.free_bytes * 9 / 10 )
        {
            throw std::runtime_error(
                "Taylor-Green application fields require " + std::to_string( application_bytes / ( 1024 * 1024 ) ) +
                " MiB per rank, exceeding 90% of currently free device memory (" +
                std::to_string( memory_info.free_bytes / ( 1024 * 1024 ) ) + " MiB)"
            );
        }

        for ( int component = 0; component < 3; ++component )
        {
            velocity_hat_[component].init( output_nx_, output_ny_, output_nz_ );
            residual_hat_[component].init( output_nx_, output_ny_, output_nz_ );
            nonlinear_hat_[component].init( output_nx_, output_ny_, output_nz_ );
            velocity_[component].init( input_nx_, input_ny_, input_nz_ );
            vorticity_[component].init( input_nx_, input_ny_, input_nz_ );
        }
        metric_.init( input_nx_, input_ny_, input_nz_ );

        if ( comm.myid == 0 )
        {
            log.info_f(
                "TAYLOR_GREEN_MEMORY application_fields_mib=%.3f local_real=%zux%zux%zu "
                "local_spectral=%zux%zux%zu",
                static_cast<double>( application_bytes ) / ( 1024.0 * 1024.0 ), input_nx_, input_ny_, input_nz_,
                output_nx_, output_ny_, output_nz_
            );
        }
    }

    void initialize_velocity_()
    {
        const real_t two_pi = real_t( 6.283185307179586476925286766559 );
        const real_t hx = two_pi / static_cast<real_t>( options_.nx );
        const real_t hy = two_pi / static_cast<real_t>( options_.ny );
        const real_t hz = two_pi / static_cast<real_t>( options_.nz );
        for_each_(
            fill_taylor_green_functor<real_t, real_array_t>{
                velocity_[0], velocity_[1], velocity_[2], hx, hy, hz, input_start_x_, input_start_y_ },
            array_range_3d( velocity_[0] )
        );
        for_each_.wait();

        for ( int component = 0; component < 3; ++component )
        {
            fft_.forward( velocity_[component], velocity_hat_[component] );
            for_each_(
                zero_complex_functor<complex_array_t>{ residual_hat_[component] },
                array_range_3d( residual_hat_[component] )
            );
            for_each_.wait();
        }
        for_each_(
            project_and_dealias_functor<complex_array_t>{
                velocity_hat_[0], velocity_hat_[1], velocity_hat_[2], coordinates_ },
            array_range_3d( velocity_hat_[0] )
        );
        for_each_.wait();
    }

    void copy_spectrum_( const complex_array_t &source, complex_array_t &destination )
    {
        memory_t::copy(
            source.size() * sizeof( complex_t ), static_cast<const void *>( source.raw_ptr() ),
            static_cast<void *>( destination.raw_ptr() )
        );
    }

    void inverse_preserving_input_( const complex_array_t &input, real_array_t &output )
    {
        copy_spectrum_( input, nonlinear_hat_[0] );
        fft_.backward( nonlinear_hat_[0], output );
        const real_t normalization =
            real_t( 1 ) / static_cast<real_t>( options_.nx * options_.ny * options_.nz );
        for_each_(
            scale_real_functor<real_t, real_array_t>{ output, normalization }, array_range_3d( output )
        );
        for_each_.wait();
    }

    void prepare_physical_fields_( int output_cutoff )
    {
        for ( int component = 0; component < 3; ++component )
            inverse_preserving_input_( velocity_hat_[component], velocity_[component] );

        const real_t normalization =
            real_t( 1 ) / static_cast<real_t>( options_.nx * options_.ny * options_.nz );
        for ( int component = 0; component < 3; ++component )
        {
            for_each_(
                vorticity_spectrum_functor<complex_array_t>{
                    velocity_hat_[0], velocity_hat_[1], velocity_hat_[2], nonlinear_hat_[0], coordinates_,
                    component, output_cutoff },
                array_range_3d( nonlinear_hat_[0] )
            );
            for_each_.wait();
            fft_.backward( nonlinear_hat_[0], vorticity_[component] );
            for_each_(
                scale_real_functor<real_t, real_array_t>{ vorticity_[component], normalization },
                array_range_3d( vorticity_[component] )
            );
            for_each_.wait();
        }
    }

    void finish_nonlinearity_from_physical_fields_()
    {
        for_each_(
            rotational_nonlinearity_functor<real_array_t>{
                velocity_[0], velocity_[1], velocity_[2], vorticity_[0], vorticity_[1], vorticity_[2] },
            array_range_3d( velocity_[0] )
        );
        for_each_.wait();

        for ( int component = 0; component < 3; ++component )
            fft_.forward( vorticity_[component], nonlinear_hat_[component] );

        for_each_(
            project_and_dealias_functor<complex_array_t>{
                nonlinear_hat_[0], nonlinear_hat_[1], nonlinear_hat_[2], coordinates_ },
            array_range_3d( nonlinear_hat_[0] )
        );
        for_each_.wait();
    }

    void evaluate_nonlinearity_()
    {
        prepare_physical_fields_( 0 );
        finish_nonlinearity_from_physical_fields_();
    }

    template <class Comm>
    real_t compute_physical_max_speed_( const Comm &comm )
    {
        for_each_(
            speed_functor<real_array_t>{ velocity_[0], velocity_[1], velocity_[2], metric_ },
            array_range_3d( metric_ )
        );
        for_each_.wait();
        const real_t local_max = reduce_(
            static_cast<int>( metric_.size() ), metric_.raw_ptr(), real_t( 0 ),
            scfd::functional::maximum<real_t>()
        );
        return comm.all_reduce_max( local_max );
    }

    void derivative_to_physical_( int velocity_component, int direction, real_array_t &output )
    {
        for_each_(
            spectral_derivative_functor<complex_array_t>{
                velocity_hat_[velocity_component], nonlinear_hat_[0], coordinates_, direction },
            array_range_3d( nonlinear_hat_[0] )
        );
        for_each_.wait();
        fft_.backward( nonlinear_hat_[0], output );
        const real_t normalization =
            real_t( 1 ) / static_cast<real_t>( options_.nx * options_.ny * options_.nz );
        for_each_(
            scale_real_functor<real_t, real_array_t>{ output, normalization }, array_range_3d( output )
        );
        for_each_.wait();
    }

    void compute_q_criterion_( int output_cutoff )
    {
        for_each_(
            fill_real_functor<real_array_t>{ metric_, real_t( 0 ) }, array_range_3d( metric_ )
        );
        for_each_.wait();

        for ( int direction = 0; direction < 3; ++direction )
        {
            derivative_to_physical_( direction, direction, vorticity_[0] );
            for_each_(
                accumulate_q_square_functor<real_array_t>{ metric_, vorticity_[0] },
                array_range_3d( metric_ )
            );
            for_each_.wait();
        }

        const int pairs[3][2] = { { 0, 1 }, { 0, 2 }, { 1, 2 } };
        for ( int pair = 0; pair < 3; ++pair )
        {
            const int i = pairs[pair][0];
            const int j = pairs[pair][1];
            derivative_to_physical_( i, j, vorticity_[0] );
            derivative_to_physical_( j, i, vorticity_[1] );
            for_each_(
                accumulate_q_pair_functor<real_array_t>{
                    metric_, vorticity_[0], vorticity_[1] },
                array_range_3d( metric_ )
            );
            for_each_.wait();
        }

        if ( output_cutoff > 0 )
        {
            fft_.forward( metric_, nonlinear_hat_[0] );
            for_each_(
                scalar_spectral_filter_functor<complex_array_t>{
                    nonlinear_hat_[0], coordinates_, output_cutoff },
                array_range_3d( nonlinear_hat_[0] )
            );
            for_each_.wait();
            fft_.backward( nonlinear_hat_[0], metric_ );
            const real_t normalization =
                real_t( 1 ) / static_cast<real_t>( options_.nx * options_.ny * options_.nz );
            for_each_(
                scale_real_functor<real_t, real_array_t>{ metric_, normalization },
                array_range_3d( metric_ )
            );
            for_each_.wait();
        }
    }

    void update_low_storage_rk3_stage_( int stage, real_t dt )
    {
        const real_t stage_a[3] = { real_t( 0 ), real_t( -5.0 / 9.0 ), real_t( -153.0 / 128.0 ) };
        const real_t stage_b[3] = { real_t( 1.0 / 3.0 ), real_t( 15.0 / 16.0 ), real_t( 8.0 / 15.0 ) };
        const real_t viscosity = real_t( 1 ) / static_cast<real_t>( options_.reynolds );

        for_each_(
            low_storage_rk3_update_functor<real_t, complex_array_t>{
                velocity_hat_[0], velocity_hat_[1], velocity_hat_[2], residual_hat_[0], residual_hat_[1],
                residual_hat_[2], nonlinear_hat_[0], nonlinear_hat_[1], nonlinear_hat_[2], coordinates_,
                viscosity, dt, stage_a[stage], stage_b[stage] },
            array_range_3d( velocity_hat_[0] )
        );
        for_each_.wait();
    }

    void advance_low_storage_rk3_( real_t dt )
    {
        for ( int stage = 0; stage < 3; ++stage )
        {
            evaluate_nonlinearity_();
            update_low_storage_rk3_stage_( stage, dt );
        }
    }

    template <class Comm>
    adaptive_step_result advance_adaptive_low_storage_rk3_(
        const Comm &comm, real_t controller_limit, real_t event_limit
    )
    {
        prepare_physical_fields_( 0 );
        const real_t max_speed = compute_physical_max_speed_( comm );
        if ( !std::isfinite( static_cast<double>( max_speed ) ) || max_speed < real_t( 0 ) )
            throw std::runtime_error( "Adaptive CFL controller measured an invalid maximum speed" );

        real_t cfl_dt = std::numeric_limits<real_t>::infinity();
        if ( max_speed > real_t( 0 ) )
        {
            cfl_dt = static_cast<real_t>( options_.target_cfl ) * min_grid_spacing_() / max_speed;
        }
        const real_t controller_dt = std::min( controller_limit, cfl_dt );
        if ( options_.dt_min > 0.0 &&
             controller_dt < static_cast<real_t>( options_.dt_min ) )
        {
            throw std::runtime_error(
                "Adaptive CFL controller requires dt=" +
                std::to_string( static_cast<double>( controller_dt ) ) +
                ", below --dt-min"
            );
        }

        const real_t dt = std::min( controller_dt, event_limit );
        if ( !std::isfinite( static_cast<double>( dt ) ) || dt <= real_t( 0 ) )
            throw std::runtime_error( "Adaptive CFL controller produced a nonpositive time step" );

        finish_nonlinearity_from_physical_fields_();
        update_low_storage_rk3_stage_( 0, dt );
        for ( int stage = 1; stage < 3; ++stage )
        {
            evaluate_nonlinearity_();
            update_low_storage_rk3_stage_( stage, dt );
        }

        adaptive_step_result result;
        result.dt        = dt;
        result.max_speed = max_speed;
        result.cfl       = max_speed * dt / min_grid_spacing_();
        const real_t tolerance = real_t( 32 ) * std::numeric_limits<real_t>::epsilon() *
            std::max( real_t( 1 ), static_cast<real_t>( options_.target_cfl ) );
        if ( result.cfl > static_cast<real_t>( options_.target_cfl ) + tolerance )
            throw std::runtime_error( "Adaptive CFL controller exceeded its target" );
        return result;
    }

    template <class Comm>
    turbulence_diagnostics<real_t> compute_diagnostics_( const Comm &comm, real_t dt )
    {
        prepare_physical_fields_( 0 );
        turbulence_diagnostics<real_t> result;
        const real_t point_count = static_cast<real_t>( options_.nx * options_.ny * options_.nz );

        for_each_(
            kinetic_energy_density_functor<real_array_t>{
                velocity_[0], velocity_[1], velocity_[2], metric_ },
            array_range_3d( metric_ )
        );
        for_each_.wait();
        result.kinetic_energy =
            comm.all_reduce_sum( reduce_( static_cast<int>( metric_.size() ), metric_.raw_ptr(), real_t( 0 ) ) ) /
            point_count;

        for_each_(
            enstrophy_density_functor<real_array_t>{
                vorticity_[0], vorticity_[1], vorticity_[2], metric_ },
            array_range_3d( metric_ )
        );
        for_each_.wait();
        result.enstrophy =
            comm.all_reduce_sum( reduce_( static_cast<int>( metric_.size() ), metric_.raw_ptr(), real_t( 0 ) ) ) /
            point_count;

        for_each_(
            speed_functor<real_array_t>{ velocity_[0], velocity_[1], velocity_[2], metric_ },
            array_range_3d( metric_ )
        );
        for_each_.wait();
        const real_t local_max = reduce_(
            static_cast<int>( metric_.size() ), metric_.raw_ptr(), real_t( 0 ),
            scfd::functional::maximum<real_t>()
        );
        result.max_speed = comm.all_reduce_max( local_max );

        for_each_(
            divergence_spectrum_functor<complex_array_t>{
                velocity_hat_[0], velocity_hat_[1], velocity_hat_[2], nonlinear_hat_[0], coordinates_ },
            array_range_3d( nonlinear_hat_[0] )
        );
        for_each_.wait();
        fft_.backward( nonlinear_hat_[0], metric_ );
        const real_t normalization = real_t( 1 ) / point_count;
        for_each_(
            scale_real_functor<real_t, real_array_t>{ metric_, normalization }, array_range_3d( metric_ )
        );
        for_each_.wait();
        for_each_( square_functor<real_array_t>{ metric_, metric_ }, array_range_3d( metric_ ) );
        for_each_.wait();
        const real_t divergence_square =
            comm.all_reduce_sum( reduce_( static_cast<int>( metric_.size() ), metric_.raw_ptr(), real_t( 0 ) ) );
        result.divergence_rms = sqrt( divergence_square / point_count );

        result.cfl = dt == real_t( 0 ) ? real_t( 0 ) : result.max_speed * dt / min_grid_spacing_();
        return result;
    }

    void validate_initial_state_( const turbulence_diagnostics<real_t> &values ) const
    {
        if ( !options_.verify_initial_energy )
            return;
        const real_t expected = real_t( 0.125 );
        const real_t relative = std::abs( values.kinetic_energy - expected ) / expected;
        const real_t tolerance = sizeof( real_t ) == sizeof( float ) ? real_t( 2.0e-5 ) : real_t( 1.0e-11 );
        if ( !std::isfinite( static_cast<double>( relative ) ) || relative > tolerance )
        {
            throw std::runtime_error(
                "Taylor-Green initial kinetic energy check failed: expected 0.125, measured " +
                std::to_string( static_cast<double>( values.kinetic_energy ) )
            );
        }
        if ( values.divergence_rms > tolerance )
        {
            throw std::runtime_error(
                "Taylor-Green initial divergence check failed: " +
                std::to_string( static_cast<double>( values.divergence_rms ) )
            );
        }
    }

    void validate_runtime_state_( const turbulence_diagnostics<real_t> &values, std::size_t step ) const
    {
        if ( !std::isfinite( static_cast<double>( values.kinetic_energy ) ) ||
             !std::isfinite( static_cast<double>( values.enstrophy ) ) ||
             !std::isfinite( static_cast<double>( values.max_speed ) ) )
        {
            throw std::runtime_error( "Non-finite Taylor-Green diagnostics at step " + std::to_string( step ) );
        }
        if ( values.cfl > static_cast<real_t>( options_.cfl_fail ) )
        {
            throw std::runtime_error(
                "Taylor-Green CFL " + std::to_string( static_cast<double>( values.cfl ) ) +
                " exceeds --cfl-fail at step " + std::to_string( step )
            );
        }
    }

    template <class Comm>
    void write_diagnostics_(
        const Comm &comm, std::size_t step, double time, real_t dt,
        const turbulence_diagnostics<real_t> &values, real_t step_wall_ms
    )
    {
        if ( comm.myid != 0 )
            return;
        diagnostics_stream_ << step << ',' << std::setprecision( 17 ) << static_cast<double>( time ) << ','
                            << static_cast<double>( dt ) << ',' << static_cast<double>( values.kinetic_energy )
                            << ',' << static_cast<double>( values.enstrophy ) << ','
                            << static_cast<double>( values.max_speed ) << ','
                            << static_cast<double>( values.divergence_rms ) << ','
                            << static_cast<double>( values.cfl ) << ',' << static_cast<double>( step_wall_ms )
                            << '\n';
        diagnostics_stream_.flush();
    }

    template <class Comm>
    void write_adaptive_timestep_(
        const Comm &comm, std::size_t step, double time, real_t dt, real_t max_speed, real_t cfl,
        real_t step_wall_ms
    )
    {
        if ( comm.myid != 0 )
            return;
        adaptive_timestep_stream_
            << step << ',' << std::setprecision( 17 ) << time << ',' << static_cast<double>( dt )
            << ',' << static_cast<double>( max_speed ) << ',' << static_cast<double>( cfl ) << ','
            << static_cast<double>( step_wall_ms ) << '\n';
        adaptive_timestep_stream_.flush();
    }

    template <class Comm>
    void write_snapshot_( const Comm &comm, const std::string &field, std::size_t step, double time )
    {
        snapshot_writer_t *writer =
            field == "omega_z" || !visualization_writer_ ? snapshot_writer_.get() : visualization_writer_.get();
        if ( writer == nullptr )
            throw std::logic_error( "Snapshot output requested without a snapshot writer" );

        const int output_cutoff =
            static_cast<int>( ( field == "omega_z" ? options_.snapshot_size : options_.viz_snapshot_size ) / 3 );
        auto &output = writer->device_array();
        if ( field == "omega_z" )
        {
            prepare_physical_fields_( output_cutoff );
            for_each_(
                snapshot_omega_z_functor<real_array_t, typename snapshot_writer_t::device_array_t>{
                    vorticity_[2], output, writer->source_offset_x(),
                    writer->source_offset_y(), writer->source_offset_z(),
                    writer->stride_x(), writer->stride_y(), writer->stride_z() },
                array_range_3d( output )
            );
        }
        else if ( field == "vorticity_magnitude" )
        {
            prepare_physical_fields_( output_cutoff );
            for_each_(
                snapshot_vorticity_magnitude_functor<real_array_t, typename snapshot_writer_t::device_array_t>{
                    vorticity_[0], vorticity_[1], vorticity_[2], output,
                    writer->source_offset_x(), writer->source_offset_y(),
                    writer->source_offset_z(), writer->stride_x(),
                    writer->stride_y(), writer->stride_z() },
                array_range_3d( output )
            );
        }
        else if ( field == "q_criterion" )
        {
            compute_q_criterion_( output_cutoff );
            for_each_(
                snapshot_scalar_functor<real_array_t, typename snapshot_writer_t::device_array_t>{
                    metric_, output, writer->source_offset_x(),
                    writer->source_offset_y(), writer->source_offset_z(),
                    writer->stride_x(), writer->stride_y(), writer->stride_z() },
                array_range_3d( output )
            );
        }
        else
            throw std::logic_error( "Unsupported snapshot field: " + field );
        for_each_.wait();
        writer->write( comm, field, step, time );
    }

    fftm_t            &fft_;
    turbulence_options options_;
    spectral_coordinates coordinates_;
    int input_start_x_ = 0;
    int input_start_y_ = 0;
    std::size_t input_nx_ = 0;
    std::size_t input_ny_ = 0;
    std::size_t input_nz_ = 0;
    std::size_t output_nx_ = 0;
    std::size_t output_ny_ = 0;
    std::size_t output_nz_ = 0;
    std::array<complex_array_t, 3> velocity_hat_;
    std::array<complex_array_t, 3> residual_hat_;
    std::array<complex_array_t, 3> nonlinear_hat_;
    std::array<real_array_t, 3>    velocity_;
    std::array<real_array_t, 3>    vorticity_;
    real_array_t                   metric_;
    for_each_t                     for_each_;
    reduce_t                       reduce_;
    std::unique_ptr<snapshot_writer_t> snapshot_writer_;
    std::unique_ptr<snapshot_writer_t> visualization_writer_;
    std::ofstream                      diagnostics_stream_;
    std::ofstream                      adaptive_timestep_stream_;
};

} // namespace turbulence
} // namespace examples
} // namespace fftm

#endif
