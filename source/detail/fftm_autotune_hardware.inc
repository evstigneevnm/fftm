struct hardware_inventory
{
    int                      mpi_ranks = 1;
    std::vector<std::string> device_classes;
    std::vector<std::string> device_backends;
    std::vector<std::string> device_models;
    std::vector<std::string> device_architectures;
    std::vector<std::string> device_uuids;
    std::vector<std::string> device_pci_bus_ids;
    std::vector<std::string> runtime_versions;
    std::vector<std::string> driver_versions;
    std::vector<std::string> device_total_bytes;
    std::string              mpi_library_version = "unavailable";
    std::vector<std::string> rank_node_names;
    std::vector<std::string> node_names;
    std::vector<int>         ranks_per_node;
    std::vector<std::string> transport_environments;
    bool                     distributed_details_available = false;
};

struct hardware_signature_record
{
    std::string compatibility_signature;
    std::string strict_signature;
    std::string transport;
    std::string canonical_compatibility;
    std::string canonical_strict;
};

inline std::string normalize_hardware_token( const std::string &value )
{
    std::string result;
    bool        previous_separator = false;
    for ( unsigned char ch : value )
    {
        const bool separator = ch == '\0' || std::isspace( ch ) || ch == ';' || ch == '=' || ch == ',' || ch == '|';
        if ( separator )
        {
            if ( !result.empty() && !previous_separator )
                result.push_back( '_' );
            previous_separator = true;
            continue;
        }
        result.push_back( static_cast<char>( ch ) );
        previous_separator = false;
    }
    while ( !result.empty() && result.back() == '_' )
        result.pop_back();
    return result.empty() ? "unknown" : result;
}

inline std::vector<std::string> sorted_values( std::vector<std::string> values, bool unique )
{
    std::sort( values.begin(), values.end() );
    if ( unique )
        values.erase( std::unique( values.begin(), values.end() ), values.end() );
    return values;
}

inline std::string join_hardware_values( const std::vector<std::string> &values, bool unique = false )
{
    const std::vector<std::string> ordered = sorted_values( values, unique );
    std::ostringstream            out;
    for ( std::size_t i = 0; i < ordered.size(); ++i )
    {
        if ( i != 0 )
            out << ',';
        out << normalize_hardware_token( ordered[i] );
    }
    return ordered.empty() ? "unknown" : out.str();
}

inline std::string join_hardware_values( const std::vector<int> &values )
{
    std::vector<int> ordered = values;
    std::sort( ordered.begin(), ordered.end() );
    std::ostringstream out;
    for ( std::size_t i = 0; i < ordered.size(); ++i )
    {
        if ( i != 0 )
            out << ',';
        out << ordered[i];
    }
    return ordered.empty() ? "unknown" : out.str();
}

inline std::string stable_hardware_hash( const std::string &value )
{
    std::uint64_t hash = UINT64_C( 14695981039346656037 );
    for ( unsigned char ch : value )
    {
        hash ^= static_cast<std::uint64_t>( ch );
        hash *= UINT64_C( 1099511628211 );
    }
    std::ostringstream out;
    out << std::hex << std::setfill( '0' ) << std::setw( 16 ) << hash;
    return out.str();
}

inline std::string local_transport_environment()
{
    auto value = []( const char *name ) {
        const char *result = getenv_or_null( name );
        return normalize_hardware_token( result == nullptr ? "default" : result );
    };
    return "affinity=" + value( "FFTM_MPI_RANK_AFFINITY_MODE" ) +
           ":ucx_tls=" + value( "UCX_TLS" ) +
           ":ucx_net_devices=" + value( "UCX_NET_DEVICES" ) +
           ":mpi_pml=" + value( "OMPI_MCA_pml" ) +
           ":mpi_btl=" + value( "OMPI_MCA_btl" );
}

inline std::string runtime_device_class( const ::fftm::detail::runtime_hardware_identity &identity )
{
    std::ostringstream out;
    out << "backend=" << normalize_hardware_token( identity.backend )
        << ":model=" << normalize_hardware_token( identity.device_name )
        << ":arch=" << normalize_hardware_token( identity.architecture )
        << ":memory=";
    if ( identity.total_memory_known )
        out << identity.total_memory_bytes;
    else
        out << "unknown";
    out << ":runtime=" << identity.runtime_version << ":driver=" << identity.driver_version;
    return out.str();
}

template <class RuntimeApi>
inline auto query_runtime_hardware_identity_impl( int ) -> decltype( RuntimeApi::get_hardware_identity() )
{
    return RuntimeApi::get_hardware_identity();
}

template <class RuntimeApi>
inline ::fftm::detail::runtime_hardware_identity query_runtime_hardware_identity_impl( long )
{
    ::fftm::detail::runtime_hardware_identity result;
    result.backend = "runtime-api";
    const auto memory = RuntimeApi::get_device_memory_info();
    result.total_memory_known = memory.total_bytes_known;
    result.total_memory_bytes = memory.total_bytes_known ? memory.total_bytes : 0;
    return result;
}

template <class RuntimeApi>
inline ::fftm::detail::runtime_hardware_identity query_runtime_hardware_identity()
{
    const auto source = query_runtime_hardware_identity_impl<RuntimeApi>( 0 );
    ::fftm::detail::runtime_hardware_identity result;
    result.backend           = source.backend;
    result.device_name       = source.device_name;
    result.device_uuid       = source.device_uuid;
    result.pci_bus_id        = source.pci_bus_id;
    result.architecture      = source.architecture;
    result.runtime_version   = source.runtime_version;
    result.driver_version    = source.driver_version;
    result.total_memory_bytes = source.total_memory_bytes;
    result.total_memory_known = source.total_memory_known;
    return result;
}

inline void append_local_runtime_identity(
    hardware_inventory &inventory, const ::fftm::detail::runtime_hardware_identity &identity
)
{
    inventory.device_classes.push_back( runtime_device_class( identity ) );
    inventory.device_backends.push_back( identity.backend );
    inventory.device_models.push_back( identity.device_name );
    inventory.device_architectures.push_back( identity.architecture );
    inventory.device_uuids.push_back( identity.device_uuid.empty() ? "unavailable" : identity.device_uuid );
    inventory.device_pci_bus_ids.push_back( identity.pci_bus_id.empty() ? "unavailable" : identity.pci_bus_id );
    inventory.runtime_versions.push_back( std::to_string( identity.runtime_version ) );
    inventory.driver_versions.push_back( std::to_string( identity.driver_version ) );
    inventory.device_total_bytes.push_back(
        identity.total_memory_known ? std::to_string( identity.total_memory_bytes ) : "unknown"
    );
}

template <class RuntimeApi>
inline hardware_inventory query_local_hardware_inventory( int num_procs )
{
    hardware_inventory result;
    result.mpi_ranks = num_procs;
    append_local_runtime_identity( result, query_runtime_hardware_identity<RuntimeApi>() );
    result.transport_environments.push_back( local_transport_environment() );
    if ( num_procs == 1 )
    {
        result.rank_node_names.push_back( "local" );
        result.node_names.push_back( "local" );
        result.ranks_per_node.push_back( 1 );
    }
    return result;
}

template <class RuntimeApi, class MPIComm>
inline hardware_inventory query_distributed_hardware_inventory( const MPIComm &comm )
{
    const auto identity = query_runtime_hardware_identity<RuntimeApi>();
    hardware_inventory result;
    result.mpi_ranks = comm.num_procs;
    result.device_classes = ::fftm::wrap::all_gather_runtime_strings( comm, runtime_device_class( identity ) );
    result.device_backends = ::fftm::wrap::all_gather_runtime_strings( comm, identity.backend );
    result.device_models = ::fftm::wrap::all_gather_runtime_strings( comm, identity.device_name );
    result.device_architectures = ::fftm::wrap::all_gather_runtime_strings( comm, identity.architecture );
    result.device_uuids = ::fftm::wrap::all_gather_runtime_strings(
        comm, identity.device_uuid.empty() ? "unavailable" : identity.device_uuid
    );
    result.device_pci_bus_ids = ::fftm::wrap::all_gather_runtime_strings(
        comm, identity.pci_bus_id.empty() ? "unavailable" : identity.pci_bus_id
    );
    result.runtime_versions = ::fftm::wrap::all_gather_runtime_strings(
        comm, std::to_string( identity.runtime_version )
    );
    result.driver_versions = ::fftm::wrap::all_gather_runtime_strings(
        comm, std::to_string( identity.driver_version )
    );
    result.device_total_bytes = ::fftm::wrap::all_gather_runtime_strings(
        comm, identity.total_memory_known ? std::to_string( identity.total_memory_bytes ) : "unknown"
    );
    result.transport_environments = ::fftm::wrap::all_gather_runtime_strings( comm, local_transport_environment() );

    const auto mpi_identity = ::fftm::wrap::query_mpi_runtime_identity( comm );
    result.mpi_library_version = mpi_identity.library_version;
    result.rank_node_names = mpi_identity.rank_node_names;
    result.node_names = mpi_identity.unique_node_names;
    result.ranks_per_node = mpi_identity.ranks_per_node;
    result.distributed_details_available = mpi_identity.distributed_details_available;
    return result;
}

inline std::vector<int> distinct_devices_per_node( const hardware_inventory &inventory )
{
    if ( inventory.rank_node_names.size() != inventory.device_uuids.size() ||
         inventory.rank_node_names.empty() )
    {
        return {};
    }

    std::map<std::string, std::vector<std::string>> devices;
    for ( std::size_t rank = 0; rank < inventory.rank_node_names.size(); ++rank )
    {
        std::string identity = inventory.device_uuids[rank];
        if ( identity.empty() || identity == "unavailable" )
        {
            if ( rank < inventory.device_pci_bus_ids.size() &&
                 !inventory.device_pci_bus_ids[rank].empty() &&
                 inventory.device_pci_bus_ids[rank] != "unavailable" )
            {
                identity = inventory.device_pci_bus_ids[rank];
            }
            else
            {
                identity = "rank-" + std::to_string( rank );
            }
        }
        devices[inventory.rank_node_names[rank]].push_back( identity );
    }

    std::vector<int> result;
    for ( auto &entry : devices )
    {
        auto &identities = entry.second;
        std::sort( identities.begin(), identities.end() );
        identities.erase( std::unique( identities.begin(), identities.end() ), identities.end() );
        result.push_back( static_cast<int>( identities.size() ) );
    }
    std::sort( result.begin(), result.end() );
    return result;
}

inline std::string hardware_transport_policy( const config_map &config, int num_procs )
{
    if ( num_procs <= 1 )
        return "local";
    const auto direct = config.find( "FFTM_DIRECT_P2P_CUDA_AWARE" );
    const auto bytes  = config.find( "FFTM_USE_P2P_BYTE_TRANSFER" );
    const bool cuda_aware = direct != config.end() && truthy( direct->second );
    const bool byte_transfer = bytes != config.end() && truthy( bytes->second );
    return std::string( cuda_aware ? "cuda-aware" : "host-staged" ) +
           ( byte_transfer ? "-byte" : "-element" );
}

inline hardware_signature_record make_hardware_signature_record(
    const hardware_inventory &inventory, const std::string &transport
)
{
    std::ostringstream compatibility;
    compatibility << "version=2"
                  << ";mpi_ranks=" << inventory.mpi_ranks
                  << ";node_count=" << ( inventory.ranks_per_node.empty() ? 0 : inventory.ranks_per_node.size() )
                  << ";ranks_per_node=" << join_hardware_values( inventory.ranks_per_node )
                  << ";devices_per_node=" << join_hardware_values( distinct_devices_per_node( inventory ) )
                  << ";mpi_library=" << normalize_hardware_token( inventory.mpi_library_version )
                  << ";device_classes=" << join_hardware_values( inventory.device_classes )
                  << ";transport=" << normalize_hardware_token( transport )
                  << ";transport_env=" << join_hardware_values( inventory.transport_environments, true );

    hardware_signature_record result;
    result.transport = transport;
    result.canonical_compatibility = compatibility.str();
    result.canonical_strict = result.canonical_compatibility +
                              ";device_uuids=" + join_hardware_values( inventory.device_uuids ) +
                              ";pci_bus_ids=" + join_hardware_values( inventory.device_pci_bus_ids ) +
                              ";nodes=" + join_hardware_values( inventory.node_names );
    result.compatibility_signature = "v2:" + stable_hardware_hash( result.canonical_compatibility );
    result.strict_signature        = "v2:" + stable_hardware_hash( result.canonical_strict );
    return result;
}

inline void set_hardware_signature_metadata(
    config_map &config, const hardware_inventory &inventory, const hardware_signature_record &signature
)
{
    config["FFTM_AUTOTUNE_HARDWARE_SIGNATURE_VERSION"] = "2";
    config["FFTM_AUTOTUNE_HARDWARE_SIGNATURE"] = signature.compatibility_signature;
    config["FFTM_AUTOTUNE_HARDWARE_SIGNATURE_STRICT"] = signature.strict_signature;
    config["FFTM_AUTOTUNE_HARDWARE_BACKENDS"] = join_hardware_values( inventory.device_backends, true );
    config["FFTM_AUTOTUNE_GPU_MODELS"] = join_hardware_values( inventory.device_models, true );
    config["FFTM_AUTOTUNE_GPU_ARCHITECTURES"] = join_hardware_values( inventory.device_architectures, true );
    config["FFTM_AUTOTUNE_DEVICE_UUIDS"] = join_hardware_values( inventory.device_uuids, true );
    config["FFTM_AUTOTUNE_DEVICE_PCI_BUS_IDS"] = join_hardware_values( inventory.device_pci_bus_ids, true );
    config["FFTM_AUTOTUNE_DEVICE_TOTAL_BYTES"] = join_hardware_values( inventory.device_total_bytes, true );
    config["FFTM_AUTOTUNE_RUNTIME_VERSIONS"] = join_hardware_values( inventory.runtime_versions, true );
    config["FFTM_AUTOTUNE_DRIVER_VERSIONS"] = join_hardware_values( inventory.driver_versions, true );
    config["FFTM_AUTOTUNE_MPI_LIBRARY"] = normalize_hardware_token( inventory.mpi_library_version );
    config["FFTM_AUTOTUNE_NODE_COUNT"] =
        inventory.ranks_per_node.empty() ? "unknown" : std::to_string( inventory.ranks_per_node.size() );
    config["FFTM_AUTOTUNE_RANKS_PER_NODE"] = join_hardware_values( inventory.ranks_per_node );
    config["FFTM_AUTOTUNE_DEVICES_PER_NODE"] =
        join_hardware_values( distinct_devices_per_node( inventory ) );
    config["FFTM_AUTOTUNE_NODE_NAMES"] = join_hardware_values( inventory.node_names, true );
    config["FFTM_AUTOTUNE_TRANSPORT"] = normalize_hardware_token( signature.transport );
    config["FFTM_AUTOTUNE_TRANSPORT_ENVIRONMENT"] =
        join_hardware_values( inventory.transport_environments, true );
}

template <class RuntimeApi>
inline std::string legacy_hardware_signature( int num_procs )
{
    const auto mem_info = RuntimeApi::get_device_memory_info();
    std::ostringstream out;
    out << "mpi=" << num_procs << ";device_total=";
    if ( mem_info.total_bytes_known )
        out << mem_info.total_bytes;
    else
        out << "unknown";
    return out.str();
}

inline void validate_hardware_match(
    const config_map &config, const hardware_inventory &inventory, const autotune_options &options
)
{
    const std::string expected = value_or_empty( config, "FFTM_AUTOTUNE_HARDWARE_SIGNATURE" );
    if ( expected.empty() )
    {
        if ( options.allow_legacy_hardware_signature )
            return;
        throw std::logic_error(
            "FFTM autotune cache has no versioned hardware signature; regenerate it or explicitly allow legacy "
            "hardware signatures"
        );
    }

    if ( expected.compare( 0, 3, "v2:" ) != 0 )
    {
        if ( options.allow_legacy_hardware_signature )
            return;
        throw std::logic_error(
            "FFTM autotune cache uses a legacy hardware signature; regenerate it or explicitly allow legacy "
            "hardware signatures"
        );
    }

    const auto actual = make_hardware_signature_record(
        inventory, hardware_transport_policy( config, inventory.mpi_ranks )
    );
    if ( expected != actual.compatibility_signature )
    {
        throw std::logic_error(
            "FFTM autotune config hardware signature '" + expected + "' does not match current hardware '" +
            actual.compatibility_signature + "'"
        );
    }

    if ( options.strict_device_identity )
    {
        const std::string strict_expected =
            value_or_empty( config, "FFTM_AUTOTUNE_HARDWARE_SIGNATURE_STRICT" );
        if ( strict_expected.empty() || strict_expected != actual.strict_signature )
        {
            throw std::logic_error(
                "FFTM autotune config strict device signature '" + strict_expected +
                "' does not match current devices '" + actual.strict_signature + "'"
            );
        }
    }
}

template <class RuntimeApi>
inline std::string hardware_signature( int num_procs )
{
    config_map transport_config;
    if ( num_procs > 1 )
    {
        transport_config["FFTM_DIRECT_P2P_CUDA_AWARE"] = "1";
        transport_config["FFTM_USE_P2P_BYTE_TRANSFER"] = "1";
    }
    const auto inventory = query_local_hardware_inventory<RuntimeApi>( num_procs );
    return make_hardware_signature_record(
               inventory, hardware_transport_policy( transport_config, num_procs )
           )
        .compatibility_signature;
}

template <class RuntimeApi>
inline void validate_hardware_match(
    const config_map &config, int num_procs, const autotune_options &options = autotune_options{}
)
{
    const auto inventory = query_local_hardware_inventory<RuntimeApi>( num_procs );
    validate_hardware_match( config, inventory, options );
}
