#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_COMMON_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_COMMON_H__

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/utils/log_mpi.h>

#include "../fftm_options.hpp"
#include "../fft_partitioning.h"
#include "../profiling.h"

namespace fftm
{

namespace detail
{

inline int mpi_int_cast( std::size_t value, const std::string &what )
{
    if ( value > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
    {
        throw std::logic_error( what + " exceeds MPI int range" );
    }
    return static_cast<int>( value );
}

template <class Request>
struct persistent_send_requests
{
    std::vector<Request>       requests;
    std::vector<const void *>  buffers;
    std::vector<int>           counts;
    std::vector<int>           destinations;
    std::vector<int>           tags;
    std::vector<MPI_Datatype>  datatypes;
};

template <class Request>
inline void free_persistent_send_requests( persistent_send_requests<Request> &state )
{
    for ( std::size_t i = 0; i < state.requests.size(); ++i )
    {
        if ( state.requests[i].request != MPI_REQUEST_NULL )
            SCFD_MPI_SAFE_CALL( MPI_Request_free( state.requests[i].native_ptr() ) );
    }
    state.requests.clear();
    state.buffers.clear();
    state.counts.clear();
    state.destinations.clear();
    state.tags.clear();
    state.datatypes.clear();
}

template <class Request>
inline void reset_persistent_send_requests( persistent_send_requests<Request> &state, int num_peers )
{
    if ( static_cast<int>( state.requests.size() ) == num_peers )
        return;

    free_persistent_send_requests( state );
    state.requests.assign( num_peers, Request() );
    state.buffers.assign( num_peers, nullptr );
    state.counts.assign( num_peers, 0 );
    state.destinations.assign( num_peers, MPI_PROC_NULL );
    state.tags.assign( num_peers, 0 );
    state.datatypes.assign( num_peers, MPI_DATATYPE_NULL );
}

template <class Request>
inline void start_persistent_send(
    persistent_send_requests<Request> &state, int peer, const void *buffer, int count, MPI_Datatype datatype,
    int destination, int tag, MPI_Comm comm
)
{
    if ( peer < 0 || peer >= static_cast<int>( state.requests.size() ) )
        throw std::logic_error( "persistent send peer index is out of range" );

    const bool needs_init =
        state.requests[peer].request == MPI_REQUEST_NULL || state.buffers[peer] != buffer || state.counts[peer] != count ||
        state.destinations[peer] != destination || state.tags[peer] != tag || state.datatypes[peer] != datatype;

    if ( needs_init )
    {
        if ( state.requests[peer].request != MPI_REQUEST_NULL )
            SCFD_MPI_SAFE_CALL( MPI_Request_free( state.requests[peer].native_ptr() ) );

        SCFD_MPI_SAFE_CALL( MPI_Send_init(
            buffer, count, datatype, destination, tag, comm, state.requests[peer].native_ptr()
        ) );
        state.buffers[peer]      = buffer;
        state.counts[peer]       = count;
        state.destinations[peer] = destination;
        state.tags[peer]         = tag;
        state.datatypes[peer]    = datatype;
    }

    SCFD_MPI_SAFE_CALL( MPI_Start( state.requests[peer].native_ptr() ) );
}

template <class Request>
struct persistent_recv_requests
{
    std::vector<Request>      requests;
    std::vector<void *>       buffers;
    std::vector<int>          counts;
    std::vector<int>          sources;
    std::vector<int>          tags;
    std::vector<MPI_Datatype> datatypes;
};

template <class Request>
inline void free_persistent_recv_requests( persistent_recv_requests<Request> &state )
{
    for ( std::size_t i = 0; i < state.requests.size(); ++i )
    {
        if ( state.requests[i].request != MPI_REQUEST_NULL )
            SCFD_MPI_SAFE_CALL( MPI_Request_free( state.requests[i].native_ptr() ) );
    }
    state.requests.clear();
    state.buffers.clear();
    state.counts.clear();
    state.sources.clear();
    state.tags.clear();
    state.datatypes.clear();
}

template <class Request>
inline void reset_persistent_recv_requests( persistent_recv_requests<Request> &state, int num_requests )
{
    if ( static_cast<int>( state.requests.size() ) == num_requests )
        return;

    free_persistent_recv_requests( state );
    state.requests.assign( num_requests, Request() );
    state.buffers.assign( num_requests, nullptr );
    state.counts.assign( num_requests, 0 );
    state.sources.assign( num_requests, MPI_PROC_NULL );
    state.tags.assign( num_requests, 0 );
    state.datatypes.assign( num_requests, MPI_DATATYPE_NULL );
}

template <class Request>
inline void start_persistent_recv(
    persistent_recv_requests<Request> &state, int index, void *buffer, int count, MPI_Datatype datatype, int source,
    int tag, MPI_Comm comm
)
{
    if ( index < 0 || index >= static_cast<int>( state.requests.size() ) )
        throw std::logic_error( "persistent recv request index is out of range" );

    const bool needs_init =
        state.requests[index].request == MPI_REQUEST_NULL || state.buffers[index] != buffer ||
        state.counts[index] != count || state.sources[index] != source || state.tags[index] != tag ||
        state.datatypes[index] != datatype;

    if ( needs_init )
    {
        if ( state.requests[index].request != MPI_REQUEST_NULL )
            SCFD_MPI_SAFE_CALL( MPI_Request_free( state.requests[index].native_ptr() ) );

        SCFD_MPI_SAFE_CALL( MPI_Recv_init(
            buffer, count, datatype, source, tag, comm, state.requests[index].native_ptr()
        ) );
        state.buffers[index]   = buffer;
        state.counts[index]    = count;
        state.sources[index]   = source;
        state.tags[index]      = tag;
        state.datatypes[index] = datatype;
    }

    SCFD_MPI_SAFE_CALL( MPI_Start( state.requests[index].native_ptr() ) );
}

} // namespace detail

} // namespace fftm

#endif // __FFTM_DETAIL_MPI_TRANSPOSE_3D_COMMON_H__
