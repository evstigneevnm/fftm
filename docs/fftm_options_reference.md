# FFTM execution and diagnostic option reference

This document covers every field in `detail::fftm_execution_options` and
`detail::fftm_diagnostic_options` from
[`source/fftm_options.hpp`](../source/fftm_options.hpp). Public initialization,
reporting, topology, strategy, and preset controls are documented in the
[README](../README.md#production-configuration).

These `detail` types are internal policy, not a stable low-level tuning API.
Applications should use production presets or measured selection. Benchmarks
may set these fields to reproduce experiments, but their presence is not an
endorsement of every combination, backend, or MPI stack. No generic environment
variable-to-field mapping exists in the library: example and benchmark adapters
provide their own mappings.

All defaults below describe the member initializers **before** preset expansion
and plan initialization. An enabled field may be inactive outside its applicable
strategy; incompatible combinations can be rejected. Some settings are
normalized rather than honored independently. Do not infer an active executor
from a requested Boolean alone.

## Execution options

Access these fields as `options.execution.<field>`. Byte sizes are per rank:
512 MiB = 536870912 bytes; 1 GiB = 1073741824 bytes. They are not total-job
memory limits. `same_xw` names the 4D redistribution preserving X/W ownership;
`same_zw` preserves Z/W ownership. WZ denotes the local transform over W and Z.

### Shared and 3D transport

| Field | Default | Effect and scope |
| --- | --- | --- |
| `use_direct_backward_receive` | `false` | Allows eligible 3D receives to land directly in their destination instead of staging/unpacking. Exact use is path-dependent (some value-transfer paths also use it for forward receives). Forced off by `reference_parity`. |
| `direct_p2p_cuda_aware` | `true` | Permits direct device-buffer peer operations on supported paths. The legacy name covers CUDA **and HIP**. This does not make MPI GPU-aware or change the binary's transport capability; the build and MPI installation must support device pointers. |
| `use_p2p_byte_transfer` | `false` | Enables eligible 3D device-aware byte-transfer paths. Message representation/chunking depends on `large_count_p2p_transport`; it does not always mean multiple messages per peer. Forced on by `reference_parity`. |
| `use_persistent_p2p` | `false` | Reuses persistent MPI request state where implemented. Requires stable buffers/request lifetimes. Forced off by `reference_parity`; not a generic way to accelerate all collectives. |
| `large_count_p2p_transport` | `fftm_3d_large_count_p2p_transport::hindexed` | Representation for eligible nonpersistent 3D byte payloads exceeding ordinary MPI byte counts; see [advanced enums](#advanced-enums). Does not control the separate 4D XW chunk scheduler. |

### Native 3D opt0 FFT and workspace

These fields target the optimized, reference-owned pencil executor, not every
3D strategy. The default-Z branch requires optimized pencil-pencil, an `opt0`
plan, `reference` or `reference_parity`, and `P1>1`. The dependent Y-topology
and executor options are active only when their parent branch is used.

| Field | Default | Effect and scope |
| --- | --- | --- |
| `use_native_opt0_default_z_layout` | `false` | Uses the Z-fast R2C output/Y-input arrangement for the native opt0 path. Parent switch for the specialized Y execution/topology. |
| `use_native_opt0_reference_y_buffer_topology` | `false` | Uses the reference-compatible Y workspace organization and associated stage-buffer lifetime aliases. Requires the default-Z path. |
| `use_native_opt0_compact_y_workarea` | `false` | Requests reduced-lifetime/compact Y workspace. Initialization may also select compact storage automatically when the full workspace does not fit but compact storage does. A requested `false` is not proof that effective storage is noncompact. |
| `use_native_opt0_tight_y_plan_sequence` | `false` | Selects the tight cached Y-plan sequence variant where reached by executor dispatch; higher-priority no-sync/raw/bundle paths can supersede it. |
| `use_native_opt0_shared_y_plan_handles` | `false` | Shares eligible C2C Y handles between forward and backward directions instead of separate direction-specific handles. |
| `use_native_opt0_y_group_device_sync` | `false` | Uses a device synchronization at the Y-group boundary instead of individual stream waits. `false` does not eliminate completion synchronization. |
| `use_native_opt0_y_no_sync_exec` | `true` | Uses the non-synchronizing FFT execution check inside the Y launch group, with completion synchronization retained after the group. Distinct from the unsafe generic diagnostic flag below. |
| `use_native_opt0_raw_y_plan_array_executor` | `false` | Selects an abstraction-owned opaque/raw Y-plan sequence variant where reached by dispatch. Does not expose vendor handles to the application. The no-sync path takes precedence in the ordinary sequence branch. |
| `use_native_opt0_memory_feasibility_guard` | `true` | Checks the native opt0 allocation requirement before large allocation and reports an infeasible configuration. Does not guarantee that unrelated caller allocations will fit. |
| `native_opt0_memory_feasibility_reserve_bytes` | `536870912` (512 MiB) | Extra free-device-memory reserve for the opt0 estimate. This is additional to estimated FFTM storage, not a cap on total memory consumption. |

### Native 4D redistribution, FFT, and workspace

| Field | Default | Effect and scope |
| --- | --- | --- |
| `use_4d_slab_native_xw_transpose` | `true` | Uses the native slab XW redistribution instead of the staged conversion path. Required by the slab `native_xzwy` spectral interface. |
| `use_4d_native_xw_direct_layout` | `false` | Receives forward data directly into native spectral-layout slices and sends backward data directly from them. Requires `native_xzwy`, device-aware MPI, `p2p_waitany`, and an eligible slab or specialized pencil path. |
| `use_4d_native_xw_chunked_transport` | `false` | Splits direct-XW peer transfers into bounded-size messages. Requires direct layout and a nonzero `native_xw_chunk_bytes`. |
| `native_xw_chunk_bytes` | `536870912` (512 MiB) | Requested maximum XW chunk bytes. Runtime caps the effective size to the MPI integer count limit and aligns it to the element size; a zero effective size is rejected. |
| `native_xw_chunk_window` | `0` | Chunks in flight per peer: `0` retains all-chunks scheduling; a positive value selects bounded scheduling. Requires chunking when nonzero; checked against integer and portable MPI tag limits. |
| `use_4d_native_xw_compact_staging` | `false` | Uses chunk-window-sized staging rather than a complete domain workspace on eligible direct-XW paths. Requires direct layout, chunking, and a positive window. |
| `use_4d_slab_native_work_area_alias` | `false` | Aliases the shared FFT workspace with an idle slab stage buffer when its lifetime/capacity permits. Restricted to native slab execution; incompatible with the separately owned workspace required by node-aligned pencil WZ. |
| `use_4d_slab_native_wz_communication_layout` | `false` | Produces WZ FFT output in communication-native order to avoid full-size pack/unpack work. Requires slab-slab, native XW, native spectral layout, direct layout, chunking, and a positive window. |
| `slab_native_wz_plan_concurrency` | `1` | Number of concurrent WZ FFT lanes; positive and capped by the local Y extent at execution setup. Values other than 1 require the slab communication-native or node-aligned pencil WZ path. Despite its name, also used by the latter. |
| `use_4d_slab_native_wz_ready_pipeline` | `false` | Posts plane communication as the corresponding WZ work becomes ready and schedules inverse work when data are ready. Requires a communication-native slab or node-aligned pencil WZ path; sends are owned independently of FFT lanes. |
| `slab_native_wz_backward_plane_credit_window` | `0` | Maximum backward planes in flight in the credit scheduler, independent of FFT-lane count. `0` retains the unbounded fallback. A positive value requires native slab-slab, `p2p_waitany`, device-aware MPI, WZ communication layout, and the ready pipeline. |
| `use_4d_slab_native_wz_backward_cyclic_peer_order` | `false` | Uses cyclic rather than the existing peer order in the backward plane-credit scheduler. Requires a positive plane-credit window. |
| `use_4d_pencil_same_zw_native_layout` | `true` | Uses the native Y-output/same-ZW communication layout for eligible pencil `native_xzwy` execution. It is not a request to change a public-layout plan's output contract. |
| `use_4d_pencil_node_aligned_wz_pipeline` | `false` | Uses the specialized node-aligned pencil WZ pipeline. Requires native same-ZW layout, direct/chunked XW, positive chunk window, ready pipeline, separately owned workspace, and a `P1>1, P2>1, P3=1` grid. Production policy additionally requires a node-aligned allocation; use the public topology policy rather than forcing this Boolean. |
| `use_4d_pencil_degenerate_local_transposes` | `false` | Replaces degenerate same-XY/same-ZW communication with local transformations on a `1 x P2 x 1` pencil grid. Requires `native_xzwy`. |
| `use_4d_pencil_degenerate_same_xw_native` | `false` | Uses native same-XW redistribution consuming the pencil post-Z layout. Requires the degenerate local-transpose path. |
| `use_4d_pencil_degenerate_wz_sliced_z_fft` | `true` | Uses sliced-Z FFT plans in the eligible degenerate local path, reducing the W/local-reorder/Z cost. This is a conditional preference, not activation of the parent degenerate path. |

The WZ ready pipeline changes which work is ready to send/execute; the chunk
window bounds message chunks; the backward plane-credit window bounds planes.
These are different units and must not be substituted for one another.

## Diagnostic options

Access these fields as `options.diagnostics.<field>`. All diagnostic Boolean
defaults are `false`. These are experimental, measurement-only, or compatibility
controls, and do not participate as arbitrary production autotune candidates.
Some experiments were slower or incorrect on tested stacks. Enabling an
allow flag does not make an experiment safe.

### 3D communication experiments

| Field | Default | Effect and scope |
| --- | --- | --- |
| `use_p2p_send_thread` | `false` | Experiments with MPI posting from a sender thread. Requires appropriate `MPI_THREAD_MULTIPLE` support when active. Normalized off by `reference_parity`. |
| `use_ready_p2p_send` | `false` | Experiments with ready-send semantics on eligible paths; matching receives must already be posted. Normalized off by `reference_parity`. |
| `print_pencil_schedule` | `false` | Requests the owned pencil communication schedule dump. This is diagnostic output, not a change to the mathematical transform. |
| `use_direct_forward_byte_receive` | `false` | Experiments with direct forward receives on the reference opt1 byte path, avoiding staging where its destination layout allows it. Not active on default-Z opt0. |
| `use_stable_forward_byte_send_buffer` | `false` | Keeps a stable forward byte-send staging buffer for the corresponding communication experiments. |
| `use_ready_stable_forward_byte_send_buffer` | `false` | Uses the ready-buffer variant of stable forward staging; requires `use_stable_forward_byte_send_buffer`. |
| `use_contiguous_forward_byte_send` | `false` | Sends from the contiguous stable forward buffer on the device-aware byte path; requires stable forward staging. |
| `use_physical_forward_peer_exchange` | `false` | Experimental peer-contiguous physical forward exchange with explicit final reshaping; restricted to the eligible opt1 reference-parity byte path. |
| `contiguous_forward_send_mode` | `fftm_3d_contiguous_forward_send_mode::single` | Single-payload versus chunked scheduling for the contiguous-forward experiment; see [advanced enums](#advanced-enums). |
| `contiguous_forward_send_chunk_bytes` | `1073741824` (1 GiB) | Requested chunk size for that experiment, not the 4D XW chunk size. Effective chunks must respect element alignment and MPI count limits. |
| `use_large_count_datatype_cache` | `false` | Caches reusable large byte datatypes in eligible hindexed paths. Does not select the transport itself. |
| `use_fft_exec_no_sync` | `false` | Removes generic hot FFT execution synchronization. Requires `allow_native_opt0_diagnostic_variants=true`; can race subsequent MPI/copy consumers. Never substitute this for the production Y-group no-sync option. |
| `use_native_backward_second_peer_loop` | `false` | Selects the alternative native backward second-transpose peer loop for comparison. |
| `use_3d_deferred_send_completion` | `false` | Returns from eligible transposes before all sends finish, retaining requests and draining them at safe later boundaries. Requires optimized pencil-pencil, `reference_parity`, `p2p_waitany`, device-aware byte transfers, and nonpersistent requests. |

### Y plan and timing experiments

The six lifecycle/bundle/context flags below require
`allow_native_opt0_diagnostic_variants=true`. They apply to the native opt0 Y
path and are not six independent optimizations that should all be enabled.

| Field | Default | Effect and scope |
| --- | --- | --- |
| `use_native_opt0_reference_y_plan_lifecycle` | `false` | Tests reference-style Y plan creation and binding order. |
| `use_native_opt0_reference_y_plan_bundle` | `false` | Tests an abstraction-owned Y plan bundle instead of the ordinary sequence path. |
| `use_native_opt0_raw_y_plan_bundle` | `false` | Activates a Y bundle with raw/reference-style creation inside the FFT abstraction; does not require the separate reference-bundle flag. |
| `use_native_opt0_y_plan_bundle_stream_first` | `false` | Changes bundle construction to create/bind streams first. Requires the bundle path. |
| `use_native_opt0_raw_y_plan_bundle_reference_streams` | `false` | Tests reference-style stream ownership/construction; requires the raw Y bundle. |
| `use_native_opt0_reference_local_plan_context` | `false` | Tests combined reference-style Z/Y/X plan-context creation and lifecycle. |
| `allow_native_opt0_diagnostic_variants` | `false` | Explicitly permits guarded Y lifecycle/bundle/context variants and generic FFT no-sync. Does not enable them by itself and is not a universal diagnostic gate. |
| `enable_local_fft_diagnostics` | `false` | Enables detailed local FFT diagnostic records on supported paths. Additional synchronization and I/O can perturb performance. |
| `local_fft_diagnostics_directory` | Empty string | Output directory passed to the local FFT diagnostics; consulted only when diagnostics are enabled. |
| `local_fft_diagnostics_label` | Empty string | Label attached to local FFT diagnostic records for run identification. |
| `enable_native_stage_timers` | `false` | Enables finer native stage/substage timing. Its synchronization overhead makes results unsuitable for an unqualified wall-only performance comparison. |

### 4D kernel and scheduling experiments

| Field | Default | Effect and scope |
| --- | --- | --- |
| `use_4d_slab_native_xw_batched_peer_kernels` | `false` | Compares batched peer pack/unpack kernels with the direct native baseline. |
| `use_4d_slab_native_xw_tensor_coalesced_kernels` | `false` | Changes indexing/order to favor tensor-side coalescing in native XW kernels. |
| `use_4d_slab_native_xw_vector4_kernels` | `false` | Tests four-element vectorized native XW pack/unpack work. |
| `use_4d_slab_native_xw_tiled_kernels` | `false` | Tests tiled native XW pack/unpack kernels. |
| `use_4d_slab_native_xw_layout_stage` | `false` | Unpacks into an intermediate layout and converts locally. Mutually exclusive with `spectral_layout_4d=native_xzwy`. |
| `use_4d_slab_native_wz_send_overlap` | `false` | Experiments with forward XY FFT execution before WZ sends complete. Requires native slab-slab, `p2p_waitany`, device-aware MPI, WZ communication layout, and the ready pipeline. |
| `use_4d_slab_native_wz_send_overlap_nonblocking_stream` | `false` | Binds the overlapped XY plan to an abstraction-owned nonblocking stream. Requires `use_4d_slab_native_wz_send_overlap`. |
| `use_4d_pencil_same_zw_peer_paired` | `false` | Tests peer-paired scheduling for the standard pencil same-ZW exchange. This does not activate a native layout. |
| `use_4d_pencil_p3_degenerate_wz_pipeline` | `false` | Deprecated alias: normalization sets `execution.use_4d_pencil_node_aligned_wz_pipeline=true`. Prefer the public node-aligned topology policy. |
| `use_4d_pencil_degenerate_xw_slab_path` | `false` | Tests a slab-like local FFT/transpose path on degenerate pencil grids; distinct from the production local-transpose/sliced-Z specialization and incompatible with node-aligned pencil WZ. |
| `use_4d_slab_native_xw_native_spectral_layout` | `false` | Deprecated alias: normalization sets `spectral_layout_4d=native_xzwy`, overriding a requested public layout. Prefer the public layout field. |

## Advanced enums

`fftm_3d_large_count_p2p_transport` belongs to
`execution.large_count_p2p_transport`:

| Value | Printable name | Mechanism |
| --- | --- | --- |
| `hindexed` | `hindexed` | Represents large byte payloads using derived hindexed MPI datatypes with bounded blocks. Normal preset choice on the reference-parity byte path. |
| `mpi_count` | `mpi-count` | Uses MPI-4 `MPI_Isend_c`/`MPI_Irecv_c` count interfaces. The implementation checks MPI header support; this is not portable to older MPI builds. |
| `element_count` | `element-count` | Counts complex elements instead of bytes where alignment and `int` element counts permit; retains fallback handling for ineligible payloads. |
| `chunked` | `chunked` | Uses multiple ordinary byte-count messages below MPI's integer count limit instead of the large-count request representation. |

`fftm_3d_contiguous_forward_send_mode` belongs to
`diagnostics.contiguous_forward_send_mode`:

| Value | Printable name | Mechanism |
| --- | --- | --- |
| `single` | `single` | Attempts one contiguous peer payload using the eligible request representation/count. |
| `chunked` | `chunked` | Splits the experimental contiguous forward payload using `contiguous_forward_send_chunk_bytes` and runtime limits. |

The corresponding `*_name` helpers format values; they do not validate a
transport stack or parse environment variables.

## Preset expansion and normalization

`production_options_3d` and `production_options_4d` are ordinary C++ functions
returning `fftm_init_options`; they do not create plans or measure candidates.
Their selection predicates are in the [README topology/preset tables](../README.md#topology-and-preset-behavior).
Unlisted fields keep their member defaults. Reporting stays silent and all
diagnostic Booleans remain false.

### 3D preset expansion

The preset sets `execution.direct_p2p_cuda_aware` from its `device_aware_mpi`
argument. For multi-rank pencil-pencil it also sets the selected pencil layout,
pipeline, `use_p2p_byte_transfer=device_aware_mpi`, and
`large_count_p2p_transport=hindexed`. When it selects opt0, it sets all of:

- `use_native_opt0_default_z_layout=true`
- `use_native_opt0_reference_y_buffer_topology=true`
- `use_native_opt0_tight_y_plan_sequence=true`
- `use_native_opt0_shared_y_plan_handles=true`
- `use_native_opt0_y_group_device_sync=true`
- `use_native_opt0_y_no_sync_exec=true`
- `use_native_opt0_raw_y_plan_array_executor=true`

This is an execution bundle with dispatch precedence, not seven separately
measured winners. It does not turn off memory feasibility checks or force
compact storage.

### 4D preset expansion

Every call sets `spectral_layout_4d` and the device-aware capability field.
Native/device-aware slab-slab additionally selects:

- `use_4d_slab_native_xw_transpose=true`
- `use_4d_native_xw_direct_layout=true`
- `use_4d_native_xw_chunked_transport=true`
- `native_xw_chunk_window=1` (the default chunk bytes remain 512 MiB)
- `use_4d_native_xw_compact_staging=true`
- `use_4d_slab_native_work_area_alias=true`
- `use_4d_slab_native_wz_communication_layout=true`
- `slab_native_wz_plan_concurrency=4`
- `use_4d_slab_native_wz_ready_pipeline=true`

The topology helper supplies `slab_native_wz_backward_plane_credit_window` and
`use_4d_slab_native_wz_backward_cyclic_peer_order`. With default topology or
outside its eligible counts they remain `0` and `false`, respectively.

Native node-aligned pencil uses direct XW, chunking/window 1, compact staging,
concurrency 4, and the ready pipeline, together with
`use_4d_pencil_same_zw_native_layout=true` and
`use_4d_pencil_node_aligned_wz_pipeline=true`. It does **not** enable the slab
work-area alias, slab WZ-layout flag, or slab backward credits.

Native standard pencil on `1 x P2 x 1` enables the three
`use_4d_pencil_degenerate_local_transposes`,
`use_4d_pencil_degenerate_same_xw_native`, and
`use_4d_pencil_degenerate_wz_sliced_z_fft` flags. Other standard pencil grids do
not receive that degenerate bundle. None of these presets changes a plan's
compile-time mode: match the direct-layout paths to `p2p_waitany`.

### Initialization overrides and safety

For `pencil_pipeline_3d=reference_parity`, initialization forces:

| Field | Effective normalized value |
| --- | --- |
| `execution.use_direct_backward_receive` | `false` |
| `execution.direct_p2p_cuda_aware` | `true` |
| `execution.use_p2p_byte_transfer` | `true` |
| `execution.use_persistent_p2p` | `false` |
| `diagnostics.use_p2p_send_thread` | `false` |
| `diagnostics.use_ready_p2p_send` | `false` |

This path requires device-aware MPI; do not use the normalized `true` to bypass
a host-staged build. Use the `reference` preset for the supported host-staged
alternative. A cache/CLI record of requested values can differ from the
effective normalized or memory-selected configuration.

The narrow opt0 Y no-sync executor retains group-boundary synchronization.
The generic diagnostic `use_fft_exec_no_sync` removes checks on other FFT
execution paths whose consumers may rely on completion: it is not a performance
preset and must remain off in production. A group-device-sync flag of `false`
selects per-stream completion, not unsynchronized consumption.

Validation also enforces the direct/chunked/window dependencies, native
spectral contracts, ready-pipeline/credit prerequisites, workspace ownership,
and the legacy alias conversions documented above. No single allow flag
disables all these guards.

## Source and validation pointers

- [Option declarations and presets](../source/fftm_options.hpp).
- [Plan initialization](../source/fftm.hpp): `normalize_init_options_`,
  `native_pencil_pencil_plan_layout_`, memory feasibility/compact selection,
  native-layout checks, and strategy initialization.
- [3D transport predicates and count handling](../source/detail/mpi_transpose_3d_pencil_pencil_reference_owned_transport_policy_datatypes.inc).
- [3D Y executor dispatch](../source/detail/mpi_transpose_3d_pencil_pencil_reference_owned_local_fft_execution.inc)
  and [group synchronization](../source/detail/mpi_transpose_3d_pencil_pencil_reference_owned_local_fft_synchronization.inc).
- [4D XW executor and chunk validation](../source/detail/mpi_transpose_4d_same_xw.h).
- [CPU-only option/preset regression](../source/tests/test_fftm_options.cpp).

The CPU-only regression verifies preset values and topology-policy errors. It
does not validate every experimental FFT/MPI executor. Changes to execution or
synchronization require backend correctness tests and wall-only performance
confirmation; documentation changes alone do not require a cluster run.
