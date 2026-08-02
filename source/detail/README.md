# FFTM Implementation Ownership

The public configuration surface lives in `../fftm_options.hpp`. Production
applications should construct it through `production_options_3d()` or
`production_options_4d()`; benchmark-only variants belong in
`fftm_init_options::diagnostics`.

The generic 3D and 4D transpose implementations are split by communication
stage. The native pencil executor is likewise decomposed into private
class-body fragments while retaining one assembly header so plan, buffer, and
execution lifetimes remain explicit and performance-regressed together:

| Header | Current responsibility | Planned extraction boundary |
| --- | --- | --- |
| `mpi_transpose_3d.h` | Compatibility umbrella for generic 3D transposes | Keep include compatibility only |
| `mpi_transpose_3d_common.h` | MPI integer checks and persistent request ownership | Stable transport primitives |
| `mpi_transpose_3d_same_x.h` | Generic same-X transpose | Pack, transport, and unpack execution |
| `mpi_transpose_3d_same_z.h` | Generic same-Z transpose | Pack, transport, and unpack execution |
| `mpi_transpose_3d_pencil_pencil_reference_owned.h` | Native pencil class declaration, ownership types, and fragment assembly | Keep as the auditable compatibility/assembly boundary |
| `mpi_transpose_3d_pencil_pencil_reference_owned_configuration_api.inc` | Internal profiler, transport, layout, native-opt0, and reference-parity controls | Public class-body configuration fragment; preserves cache-reset side effects |
| `mpi_transpose_3d_pencil_pencil_reference_owned_diagnostics_api.inc` | Public controls for local FFT diagnostics and native stage timers | Class-body API fragment; diagnostics only |
| `mpi_transpose_3d_pencil_pencil_reference_owned_local_fft_diagnostics.inc` | Inline CSV, layout-offset, and sampled-signature diagnostics | Class-body fragment; no production scheduling or device API ownership |
| `mpi_transpose_3d_pencil_pencil_reference_owned_plan_schedule.inc` | Native plan-state validation, schedule construction, and schedule-slot accessors | Class-body fragment; no communication-loop or buffer ownership |
| `mpi_transpose_3d_pencil_pencil_reference_owned_first_transpose_schedule.inc` | First-transpose count, offset, tag, and buffer-pointer accessors | Class-body fragment; consumes the verified native schedule without owning transport |
| `mpi_transpose_3d_pencil_pencil_reference_owned_first_transpose_copy.inc` | First-transpose runtime copy and packing helpers | Class-body fragment; runtime abstraction only, with no raw CUDA/HIP calls |
| `mpi_transpose_3d_pencil_pencil_reference_owned_first_transpose_forward.inc` | Forward first-transpose peer posting, completion, and unpack execution | Class-body fragment; preserves production row-communicator ordering |
| `mpi_transpose_3d_pencil_pencil_reference_owned_first_transpose_backward.inc` | Backward first-transpose peer posting, completion, and unpack execution | Class-body fragment; preserves production row-communicator ordering |
| `mpi_transpose_3d_pencil_pencil_reference_owned_second_transpose_schedule.inc` | Second-transpose count, offset, extent, tag, and buffer-pointer accessors | Class-body fragment; consumes the verified native schedule without owning transport |
| `mpi_transpose_3d_pencil_pencil_reference_owned_second_transpose_copy.inc` | Second-transpose runtime copy and packing helpers | Class-body fragment; runtime abstraction only, with no raw CUDA/HIP calls |
| `mpi_transpose_3d_pencil_pencil_reference_owned_second_transpose_forward.inc` | Forward second-transpose legacy and native peer execution | Class-body fragment; preserves production line-communicator ordering |
| `mpi_transpose_3d_pencil_pencil_reference_owned_second_transpose_backward.inc` | Backward second-transpose legacy and native peer execution | Class-body fragment; preserves production line-communicator ordering |
| `mpi_transpose_3d_pencil_pencil_reference_owned_transport_policy_datatypes.inc` | Transport policy, large-count selection, datatype caches, and persistent-request teardown | Class-body fragment; transport lifecycle without transpose execution ownership |
| `mpi_transpose_3d_pencil_pencil_reference_owned_transport_posting.inc` | Direct receive datatype setup and value/byte MPI request posting helpers | Class-body fragment; SCFD communication path plus isolated MPI-4 count fallback |
| `mpi_transpose_3d_pencil_pencil_reference_owned_deferred_send.inc` | Deferred-send ownership, progress, draining, and send-wait helpers | Class-body fragment; no transpose layout ownership |
| `mpi_transpose_3d_pencil_pencil_reference_owned_request_state.inc` | Per-communicator request counts, stream synchronization, reset, and peer-order construction | Class-body fragment shared by both transposes |
| `mpi_transpose_3d_pencil_pencil_reference_owned_workspace_api.inc` | Public external device/host workspace controls and size queries | Class-body API fragment; no allocation or execution ownership |
| `mpi_transpose_3d_pencil_pencil_reference_owned_initialization.inc` | Plan binding, communicator construction, buffer/request allocation, and stream initialization | Class-body fragment preserving the production initialization order |
| `mpi_transpose_3d_pencil_pencil_reference_owned_initialization_validation.inc` | Partition and initialized-state validation | Private class-body fragment shared by initialization and execution entry points |
| `mpi_transpose_3d_pencil_pencil_reference_owned_workspace_binding.inc` | Workspace byte arithmetic, native slot mapping, external-buffer binding, and memory accounting | Private class-body fragment using SCFD array ownership and raw-data views |
| `mpi_transpose_3d_pencil_pencil_reference_owned_local_fft_plan_lifecycle.inc` | Local FFT entry points, Y-plan sequence/bundle binding, stream/work-area association, and workspace queries | Public class-body fragment consuming only FFT/runtime wrapper APIs |
| `mpi_transpose_3d_pencil_pencil_reference_owned_local_fft_execution.inc` | Single-plan, offset-sequence, and plan-array execution dispatch | Private class-body fragment preserving production no-sync and diagnostic timing branches |
| `mpi_transpose_3d_pencil_pencil_reference_owned_local_fft_synchronization.inc` | Native-opt0 Y-plan stream/device synchronization and receive-layout query | Private class-body fragment using only FFT/runtime abstractions |
| `mpi_transpose_3d_pencil_pencil_reference_owned_pipeline_execution.inc` | Generic and native-opt0 forward/backward pipeline orchestration | Public class-body fragment preserving FFT, transpose, timer, and deferred-send ordering |
| `mpi_transpose_3d_pencil_pencil_reference_owned_transport_synchronization.inc` | Byte-request profiling and CUDA-aware send/FFT boundary synchronization | Private class-body fragment preserving reference-parity synchronization semantics |
| `mpi_transpose_3d_pencil_pencil_reference_owned_schedule_reporting.inc` | Per-stage schedule summaries, per-peer diagnostics, and selected transport reporting | Private class-body fragment fully gated by schedule-print controls |
| `mpi_transpose_3d_pencil_pencil_reference_owned_redistribution_extents.inc` | Maximum first/second transpose send, receive, and shared redistribution extents | Private class-body sizing fragment; no allocation or execution ownership |
| `mpi_transpose_3d_pencil_pencil_reference_owned_configuration_state.inc` | Backend references and validated production/diagnostic configuration state | Private member-state fragment preserving declaration order |
| `mpi_transpose_3d_pencil_pencil_reference_owned_local_fft_state.inc` | Local FFT diagnostics, timing, and native-opt0 plan identifiers | Private member-state fragment using opaque FFT abstraction identifiers |
| `mpi_transpose_3d_pencil_pencil_reference_owned_layout_workspace_state.inc` | Partition, communicator, external workspace, and redistribution buffer state | Private member-state fragment using SCFD storage and communication types |
| `mpi_transpose_3d_pencil_pencil_reference_owned_transport_state.inc` | MPI request, datatype, schedule, stream, and bound native-plan state | Private member-state fragment preserving transport ownership |
| `mpi_transpose_3d_pencil_pencil_reference_owned_stage_diagnostics.inc` | Native stage timing, topology markers, and Y work-area metadata | Private class-body fragment; no production executor ownership |
| `mpi_transpose_3d_pencil_pencil_reference_owned_y_diagnostics.h` | Diagnostic Y-executor microbenchmark implementation | Excluded from the production execution path; retained for performance investigations |
| `mpi_transpose_4d.h` | Compatibility umbrella for all 4D transpose classes | Keep include compatibility only |
| `mpi_transpose_4d_common.h` | Shared 4D indexing and stage-timing helpers | Stable common primitives |
| `mpi_transpose_4d_same_xy.h` | Same-XY transpose | Pack, transport, and unpack execution |
| `mpi_transpose_4d_same_xw.h` | Same-XW transpose and native slab scheduler | Direct/chunked XW production execution |
| `mpi_transpose_4d_same_zw.h` | Same-ZW transpose | Pencil transport and native-layout execution |

CUDA, HIP, cuFFT, and hipFFT operations must remain in `../external_wrap` or
the SCFD backend. Transpose and plan code consumes only wrapper/runtime types.
The unused historical raw-CUDA distributor was removed after confirming that
no active library, example, or test target included it.
