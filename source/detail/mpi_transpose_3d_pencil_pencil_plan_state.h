#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_PLAN_STATE_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_PLAN_STATE_H__

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../fft_partitioning.h"

namespace fftm
{
namespace detail
{

enum class pencil_pencil_plan_layout
{
    opt0,
    opt1
};

struct pencil_pencil_plan_record
{
    long long rank             = 0;
    long long pidx_i           = 0;
    long long pidx_j           = 0;
    long long input_start_x    = 0;
    long long input_size_x     = 0;
    long long input_start_y    = 0;
    long long input_size_y     = 0;
    long long input_size_z     = 0;
    long long stage1_start_x   = 0;
    long long stage1_size_x    = 0;
    long long stage1_size_y    = 0;
    long long stage1_start_z   = 0;
    long long stage1_size_z    = 0;
    long long output_size_x    = 0;
    long long output_start_y   = 0;
    long long output_size_y    = 0;
    long long output_start_z   = 0;
    long long output_size_z    = 0;
};

struct pencil_pencil_schedule_record
{
    long long rank         = 0;
    long long pidx_i       = 0;
    long long pidx_j       = 0;
    long long transpose    = 0;
    long long direction    = 0;
    long long operation    = 0;
    long long order_index  = 0;
    long long peer         = 0;
    long long offset_elems = 0;
    long long bytes        = 0;
    long long mpi_peer     = 0;
    long long mpi_tag      = 0;
};

inline std::vector<long long> flatten_pencil_plan_record( const pencil_pencil_plan_record &record )
{
    std::vector<long long> out;
    out.reserve( 18 );
    out.push_back( record.rank );
    out.push_back( record.pidx_i );
    out.push_back( record.pidx_j );
    out.push_back( record.input_start_x );
    out.push_back( record.input_size_x );
    out.push_back( record.input_start_y );
    out.push_back( record.input_size_y );
    out.push_back( record.input_size_z );
    out.push_back( record.stage1_start_x );
    out.push_back( record.stage1_size_x );
    out.push_back( record.stage1_size_y );
    out.push_back( record.stage1_start_z );
    out.push_back( record.stage1_size_z );
    out.push_back( record.output_size_x );
    out.push_back( record.output_start_y );
    out.push_back( record.output_size_y );
    out.push_back( record.output_start_z );
    out.push_back( record.output_size_z );
    return out;
}

inline std::vector<long long> flatten_pencil_schedule_records(
    const std::vector<pencil_pencil_schedule_record> &records
)
{
    std::vector<long long> out;
    out.reserve( records.size() * 12 );
    for ( std::size_t i = 0; i < records.size(); ++i )
    {
        const pencil_pencil_schedule_record &record = records[i];
        out.push_back( record.rank );
        out.push_back( record.pidx_i );
        out.push_back( record.pidx_j );
        out.push_back( record.transpose );
        out.push_back( record.direction );
        out.push_back( record.operation );
        out.push_back( record.order_index );
        out.push_back( record.peer );
        out.push_back( record.offset_elems );
        out.push_back( record.bytes );
        out.push_back( record.mpi_peer );
        out.push_back( record.mpi_tag );
    }
    return out;
}

class mpi_transpose_3d_pencil_pencil_plan_state
{
public:
    void init(
        const processor_grid &grid,
        const global_sizes &sizes,
        int world_rank,
        int world_size,
        pencil_pencil_plan_layout layout,
        std::size_t value_size_bytes
    )
    {
        if ( grid.p1 == 0 || grid.p2 == 0 )
            throw std::logic_error( "pencil-pencil plan state requires non-zero P1/P2" );
        if ( grid.p1 * grid.p2 != static_cast<std::size_t>( world_size ) )
            throw std::logic_error( "pencil-pencil plan state requires P1*P2 == world size" );
        if ( world_rank < 0 || world_rank >= world_size )
            throw std::logic_error( "pencil-pencil plan state rank is out of range" );
        if ( value_size_bytes == 0 )
            throw std::logic_error( "pencil-pencil plan state requires non-zero value size" );

        grid_             = grid;
        sizes_            = sizes;
        world_rank_       = world_rank;
        world_size_       = world_size;
        layout_           = layout;
        value_size_bytes_ = value_size_bytes;

        rank_comm_view comm;
        comm.myid      = world_rank_;
        comm.num_procs = world_size_;

        fft_partitioning<rank_comm_view> partitioning( comm );
        partitioning.init( grid_, sizes_ );
        std::tie( myid_i_, myid_j_, myid_k_ ) = partitioning.get_my_grid();
        std::tie( input_dim_, transpose1_dim_, output_dim_ ) = partitioning.get_partitioning_3D();

        half_input_dim_           = input_dim_;
        half_input_dim_.size_z[0] = sizes_.Nz / 2 + 1;
        half_input_dim_.compute_offsets( false );

        first_order_  = cyclic_peer_order_( myid_j_, static_cast<int>( grid_.p2 ) );
        second_order_ = cyclic_peer_order_( myid_i_, static_cast<int>( grid_.p1 ) );
        inited_       = true;
    }

    bool is_inited() const
    {
        return inited_;
    }

    int rank_i() const
    {
        ensure_inited_();
        return myid_i_;
    }

    int rank_j() const
    {
        ensure_inited_();
        return myid_j_;
    }

    const partition &real_input_partition() const
    {
        ensure_inited_();
        return input_dim_;
    }

    const partition &half_input_partition() const
    {
        ensure_inited_();
        return half_input_dim_;
    }

    const partition &stage1_partition() const
    {
        ensure_inited_();
        return transpose1_dim_;
    }

    const partition &output_partition() const
    {
        ensure_inited_();
        return output_dim_;
    }

    const std::vector<int> &first_transpose_order() const
    {
        ensure_inited_();
        return first_order_;
    }

    const std::vector<int> &second_transpose_order() const
    {
        ensure_inited_();
        return second_order_;
    }

    pencil_pencil_plan_record local_plan_record() const
    {
        ensure_inited_();
        pencil_pencil_plan_record record;
        record.rank           = world_rank_;
        record.pidx_i         = myid_i_;
        record.pidx_j         = myid_j_;
        record.input_start_x  = static_cast<long long>( half_input_dim_.start_x[myid_i_] );
        record.input_size_x   = static_cast<long long>( half_input_dim_.size_x[myid_i_] );
        record.input_start_y  = static_cast<long long>( half_input_dim_.start_y[myid_j_] );
        record.input_size_y   = static_cast<long long>( half_input_dim_.size_y[myid_j_] );
        record.input_size_z   = static_cast<long long>( half_input_dim_.size_z[0] );
        record.stage1_start_x = static_cast<long long>( transpose1_dim_.start_x[myid_i_] );
        record.stage1_size_x  = static_cast<long long>( transpose1_dim_.size_x[myid_i_] );
        record.stage1_size_y  = static_cast<long long>( transpose1_dim_.size_y[0] );
        record.stage1_start_z = static_cast<long long>( transpose1_dim_.start_z[myid_j_] );
        record.stage1_size_z  = static_cast<long long>( transpose1_dim_.size_z[myid_j_] );
        record.output_size_x  = static_cast<long long>( output_dim_.size_x[0] );
        record.output_start_y = static_cast<long long>( output_dim_.start_y[myid_i_] );
        record.output_size_y  = static_cast<long long>( output_dim_.size_y[myid_i_] );
        record.output_start_z = static_cast<long long>( output_dim_.start_z[myid_j_] );
        record.output_size_z  = static_cast<long long>( output_dim_.size_z[myid_j_] );
        return record;
    }

    std::vector<pencil_pencil_schedule_record> local_schedule_records() const
    {
        ensure_inited_();
        std::vector<pencil_pencil_schedule_record> records;
        records.reserve( ( first_order_.size() + second_order_.size() ) * 4 );
        append_first_transpose_records_( records );
        append_second_transpose_records_( records );
        return records;
    }

private:
    struct rank_comm_view
    {
        int myid      = 0;
        int num_procs = 1;
    };

    static std::vector<int> cyclic_peer_order_( int rank_index, int size )
    {
        std::vector<int> order;
        order.reserve( std::max( 0, size - 1 ) );
        for ( int step = 1; step < size; ++step )
            order.push_back( ( rank_index + step ) % size );
        return order;
    }

    bool opt1_layout_() const
    {
        return layout_ == pencil_pencil_plan_layout::opt1;
    }

    void ensure_inited_() const
    {
        if ( !inited_ )
            throw std::logic_error( "pencil-pencil plan state is not initialized" );
    }

    void append_record_(
        std::vector<pencil_pencil_schedule_record> &records,
        long long transpose,
        long long direction,
        long long operation,
        long long order_index,
        long long peer,
        std::size_t offset_elems,
        std::size_t elems,
        long long mpi_tag
    ) const
    {
        pencil_pencil_schedule_record record;
        record.rank         = world_rank_;
        record.pidx_i       = myid_i_;
        record.pidx_j       = myid_j_;
        record.transpose    = transpose;
        record.direction    = direction;
        record.operation    = operation;
        record.order_index  = order_index;
        record.peer         = peer;
        record.offset_elems = static_cast<long long>( offset_elems );
        record.bytes        = static_cast<long long>( elems * value_size_bytes_ );
        record.mpi_peer     = peer;
        record.mpi_tag      = mpi_tag;
        records.push_back( record );
    }

    void append_first_transpose_records_( std::vector<pencil_pencil_schedule_record> &records ) const
    {
        for ( std::size_t order_index = 0; order_index < first_order_.size(); ++order_index )
        {
            const std::size_t peer_j = static_cast<std::size_t>( first_order_[order_index] );

            std::size_t offset = transpose1_dim_.size_x[myid_i_] * half_input_dim_.start_y[peer_j] *
                                 transpose1_dim_.size_z[myid_j_];
            std::size_t elems = transpose1_dim_.size_x[myid_i_] * half_input_dim_.size_y[peer_j] *
                                transpose1_dim_.size_z[myid_j_];
            append_record_( records, 1, 0, 0, order_index, peer_j, offset, elems, peer_j );

            offset = opt1_layout_()
                         ? transpose1_dim_.start_z[peer_j] * half_input_dim_.size_y[myid_j_] *
                               half_input_dim_.size_x[myid_i_]
                         : half_input_dim_.size_x[myid_i_] * half_input_dim_.size_y[myid_j_] *
                               transpose1_dim_.start_z[peer_j];
            elems = half_input_dim_.size_x[myid_i_] * half_input_dim_.size_y[myid_j_] *
                    transpose1_dim_.size_z[peer_j];
            append_record_( records, 1, 0, 1, order_index, peer_j, offset, elems, myid_j_ );

            offset = opt1_layout_()
                         ? transpose1_dim_.start_z[peer_j] * half_input_dim_.size_y[myid_j_] *
                               half_input_dim_.size_x[myid_i_]
                         : half_input_dim_.size_x[myid_i_] * half_input_dim_.size_y[myid_j_] *
                               transpose1_dim_.start_z[peer_j];
            elems = transpose1_dim_.size_z[peer_j] * half_input_dim_.size_y[myid_j_] *
                    half_input_dim_.size_x[myid_i_];
            append_record_( records, 1, 1, 0, order_index, peer_j, offset, elems, peer_j );

            offset = transpose1_dim_.size_x[myid_i_] * half_input_dim_.start_y[peer_j] *
                     transpose1_dim_.size_z[myid_j_];
            elems = transpose1_dim_.size_x[myid_i_] * half_input_dim_.size_y[peer_j] *
                    transpose1_dim_.size_z[myid_j_];
            append_record_( records, 1, 1, 1, order_index, peer_j, offset, elems, myid_j_ );
        }
    }

    void append_second_transpose_records_( std::vector<pencil_pencil_schedule_record> &records ) const
    {
        for ( std::size_t order_index = 0; order_index < second_order_.size(); ++order_index )
        {
            const std::size_t peer_i = static_cast<std::size_t>( second_order_[order_index] );

            std::size_t offset = transpose1_dim_.start_x[peer_i] * output_dim_.size_y[myid_i_] *
                                 output_dim_.size_z[myid_j_];
            std::size_t elems = transpose1_dim_.size_x[peer_i] * output_dim_.size_y[myid_i_] *
                                output_dim_.size_z[myid_j_];
            append_record_( records, 2, 0, 0, order_index, peer_i, offset, elems, peer_i );

            offset = transpose1_dim_.size_x[myid_i_] * transpose1_dim_.size_z[myid_j_] *
                     output_dim_.start_y[peer_i];
            elems = transpose1_dim_.size_x[myid_i_] * output_dim_.size_y[peer_i] *
                    transpose1_dim_.size_z[myid_j_];
            append_record_( records, 2, 0, 1, order_index, peer_i, offset, elems, myid_i_ );

            offset = transpose1_dim_.size_x[myid_i_] * transpose1_dim_.size_z[myid_j_] *
                     output_dim_.start_y[peer_i];
            elems = transpose1_dim_.size_x[myid_i_] * transpose1_dim_.size_z[myid_j_] *
                    output_dim_.size_y[peer_i];
            append_record_( records, 2, 1, 0, order_index, peer_i, offset, elems, peer_i );

            offset = transpose1_dim_.start_x[peer_i] * output_dim_.size_y[myid_i_] *
                     output_dim_.size_z[myid_j_];
            elems = transpose1_dim_.size_x[peer_i] * output_dim_.size_y[myid_i_] *
                    output_dim_.size_z[myid_j_];
            append_record_( records, 2, 1, 1, order_index, peer_i, offset, elems, myid_i_ );
        }
    }

    processor_grid              grid_;
    global_sizes                sizes_;
    int                         world_rank_       = 0;
    int                         world_size_       = 1;
    int                         myid_i_           = 0;
    int                         myid_j_           = 0;
    int                         myid_k_           = 0;
    pencil_pencil_plan_layout   layout_           = pencil_pencil_plan_layout::opt0;
    std::size_t                 value_size_bytes_ = 0;
    partition                   input_dim_;
    partition                   half_input_dim_;
    partition                   transpose1_dim_;
    partition                   output_dim_;
    std::vector<int>            first_order_;
    std::vector<int>            second_order_;
    bool                        inited_ = false;
};

} // namespace detail
} // namespace fftm

#endif
