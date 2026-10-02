"""Correctness tests for flashinfer.comm.trtllm_allgather (all-gather on the Lamport one-shot allreduce workspace).

What is covered, and why:
  1. workspace replacement without graph re-capture: the feature's main claim. A graph recorded against workspace A
     must replay correctly after A's pointer table is rewritten in place with new peer buffers and control flags;
  2. graph lengths that are not a multiple of the 3-slot Lamport rotation (1, 2, 4, 5 calls), captured after 0/1/2
     eager calls and interleaved with eager calls between replays;
  3. back-to-back calls with no intermediate synchronization, irregular size sequences, per-rank skew, and
     alternation with the one-shot allreduce on the same workspace (both advance the slot rotation the same way);
  4. boundary sizes (empty, one vector, around block/grid multiples, exact capacity, one vector beyond) and
     alignment (16-byte aligned slices accepted, odd storage offsets and non-vector sizes rejected before launch);
  5. bit-exact value checks (integer views) on inputs that encode rank, position and call number plus +0/-0/inf/nan/
     subnormal/extreme values; the documented -0.0 -> +0.0 normalization is checked explicitly;
  6. execution variants: fp32 workspace, launch_with_pdl, trigger_completion_at_end, metadata mismatch rejection,
     calls without metadata; world_size 2/4/8/16 (skipped when the host has fewer GPUs);
  7. a bounded harness: shared deadline, prompt failure detection, remaining workers killed; two self-tests prove it
     reports a failing and a stalled worker instead of hanging.

Run as: pytest tests/comm/test_trtllm_allgather.py -rA   (needs >= 2 GPUs; CUDA_VISIBLE_DEVICES picks them)
"""

import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Callable, Sequence

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.trtllm_ar import AllReduceFusionPattern
from flashinfer.utils import get_compute_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_compute_capability(torch.device("cuda:0"))[0] not in (9, 10, 12),
    reason="trtllm_comm kernels support SM90/SM100/SM12x only",
)

# Workspace sized for 256 tokens x a 38,720-wide vocab shard (GLM-5.3-Flash lm_head at TP=4): 19 MiB per rank.
MAX_TOKEN_NUM = 256
HIDDEN_DIM = 38720
VEC = {torch.float16: 8, torch.bfloat16: 8, torch.float32: 4}  # elements per 16-byte vector
INT_VIEW = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32}
WORLD_SIZES = [2, 4, 8, 16]
GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0


@pytest.fixture(scope="module", autouse=True)
def _build_jit_module_once():
    """Build/load the trtllm_comm JIT module in the parent before any ranks are spawned: with a cold JIT cache
    every rank otherwise compiles the module concurrently (a rank was SIGKILLed once on a 224-core host)."""
    if GPUS:
        from flashinfer.comm.trtllm_ar import get_trtllm_comm_module

        get_trtllm_comm_module()


# ----------------------------------------------------------------------------------------------------------------
# data: every rank can compute every other rank's input locally from (rank, call); values encode rank, position and
# call so a swapped rank, a stale slot or a shifted element is visible; bit-exact comparison through integer views
# ----------------------------------------------------------------------------------------------------------------
_SPECIAL = [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 65504.0, -65504.0, 6.1035e-5, 5.9605e-8, 1.0,
            -1.0, 0.5, 3.140625, -2.71875, 1e-3, -1e-3]  # fp16 max, min normal, smallest subnormal, odd values


def make_input(rank: int, call: int, shape: Sequence[int], dtype: torch.dtype, device) -> torch.Tensor:
    """Generated on the GPU: CPU-side generation with the default OMP thread count made every call take seconds when
    several ranks share one host (224 cores x 4 processes of spinning OpenMP threads)."""
    numel = 1
    for d in shape:
        numel *= d
    i = torch.arange(numel, dtype=torch.float32, device=device)
    structured = (i % 997) * 0.25 + rank * 1000.0 + (call % 50) * 0.125
    g = torch.Generator(device=device).manual_seed(100003 * call + 7919 * rank + numel)
    noise = torch.randn(numel, generator=g, device=device) * 3.0
    x = torch.where(i % 2 == 0, structured, noise).to(dtype)
    if numel >= 64:
        sp = torch.tensor(_SPECIAL, dtype=torch.float32, device=device).to(dtype)
        x[:16] = sp
        x[16:32] = -sp  # flips the sign bit of every special value, including NaN's
        x[32:48] = torch.full((16,), -0.0, dtype=dtype, device=device)  # a full sentinel vector
    return x.view(*shape)


def expected_gather(world_size: int, call: int, shape, dtype, device) -> torch.Tensor:
    return torch.stack([make_input(r, call, shape, dtype, device) for r in range(world_size)])


def normalize_ref(ref: torch.Tensor) -> torch.Tensor:
    """The kernel uses -0.0 as the Lamport sentinel and emits +0.0 for it; everything else is passed bit-for-bit."""
    ref = ref.clone()
    ref[ref == 0] = 0.0  # -0.0 == 0 -> written as +0.0
    return ref


def check_bits(out: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    o = out.view(INT_VIEW[out.dtype])
    r = normalize_ref(ref).view(INT_VIEW[ref.dtype])
    if torch.equal(o, r):  # integer views: NaN payloads and the sign of zero are compared bit for bit
        return
    bad = o != r
    idx = bad.flatten().nonzero().flatten()[:6].tolist()
    raise AssertionError(
        f"{what}: {int(bad.sum())}/{o.numel()} elements differ, first flat indices {idx}; "
        f"got {out.flatten()[idx].tolist()} want {ref.flatten()[idx].tolist()}"
    )


# ----------------------------------------------------------------------------------------------------------------
# bounded multi-process harness
# ----------------------------------------------------------------------------------------------------------------
class HarnessFailure(RuntimeError):
    pass


def _worker_main(fn: Callable, world_size: int, rank: int, dtype: torch.dtype, port: int, kwargs: dict) -> None:
    try:
        torch.set_num_threads(4)  # several ranks share the host; default = all cores -> OpenMP spin contention
        torch.cuda.set_device(rank)
        # the parent owns the rendezvous socket (TCPStore master), so no port can be stolen between spawn and init
        store = dist.TCPStore("127.0.0.1", port, world_size, is_master=False, timeout=__import__("datetime").timedelta(seconds=120))
        dist.init_process_group("nccl", store=store, rank=rank, world_size=world_size)
        fn(world_size, rank, dtype, **kwargs)
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()
        sys.stdout.flush()
        os._exit(0)
    except BaseException:  # noqa: BLE001 - report anything, then exit non-zero without touching the collective
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)


def run_workers(fn: Callable, world_size: int, dtype: torch.dtype, timeout_s: float = 600.0, _procs=None, **kwargs) -> None:
    """Spawn one process per rank; fail promptly on any non-zero exit; kill the rest on failure or deadline.
    Exit codes are re-checked after joining: a worker that dies between the failure scan and the all-stopped
    check must not turn into a pass. `_procs` injects process stand-ins for the harness's own tests."""
    ctx = mp.get_context("spawn")
    master = dist.TCPStore("127.0.0.1", 0, world_size, is_master=True, wait_for_workers=False)  # held until the end
    port = master.port
    procs = _procs if _procs is not None else [
        ctx.Process(target=_worker_main, args=(fn, world_size, r, dtype, port, kwargs), name=f"rank{r}")
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + timeout_s
    failure = None
    while True:
        bad = [p for p in procs if not p.is_alive() and p.exitcode not in (0, None)]
        if bad:
            failure = f"{bad[0].name} exited with code {bad[0].exitcode}"
            break
        if not any(p.is_alive() for p in procs):
            break
        if time.monotonic() > deadline:
            failure = f"timeout after {timeout_s:.0f}s; still running: {[p.name for p in procs if p.is_alive()]}"
            break
        time.sleep(0.1)
    killed = [p for p in procs if p.is_alive()]
    for p in killed:
        p.kill()
    for p in procs:
        p.join(10)
    del master
    if failure is None:  # unconditional final verdict from the exit codes
        bad = [p for p in procs if p.exitcode != 0]
        if bad:
            failure = f"{bad[0].name} exited with code {bad[0].exitcode} (noticed after join)"
    if failure:
        raise HarnessFailure(f"{fn.__name__} (world_size={world_size}, {dtype}): {failure}")


def _need_gpus(world_size: int) -> None:
    if world_size > GPUS:
        pytest.skip(f"world_size {world_size} needs {world_size} GPUs, host exposes {GPUS}")


# ----------------------------------------------------------------------------------------------------------------
# worker-side helpers
# ----------------------------------------------------------------------------------------------------------------
def create_workspace(rank: int, world_size: int, dtype: torch.dtype, max_token_num: int = MAX_TOKEN_NUM,
                     hidden_dim: int = HIDDEN_DIM):
    return comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
        rank, world_size, max_token_num, hidden_dim, use_fp32_lamport=(dtype == torch.float32),
        group=dist.group.WORLD, create_metadata=True,
    )


def capacity_elems(metadata: dict, world_size: int, dtype: torch.dtype) -> int:
    return metadata["lamport_comm_size"] // (world_size * torch.tensor([], dtype=dtype).element_size())


VERBOSE = os.environ.get("AG_VERBOSE", "0") == "1"


def log(rank: int, msg: str) -> None:
    if VERBOSE:
        print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Calls:
    """Monotonic call counter so every collective in a worker has a unique data pattern (identical on all ranks)."""

    def __init__(self) -> None:
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return self.n


def gather_checked(rank, world_size, ws, meta, dtype, device, shape, calls: Calls, sync=True, **kw):
    call = calls.next()
    log(rank, f"eager call {call} shape {tuple(shape)} {kw}")
    x = make_input(rank, call, shape, dtype, device)
    out = torch.empty((world_size,) + tuple(shape), dtype=dtype, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    comm.trtllm_allgather(x, out, world_size, rank, ws, metadata=meta, **kw)
    if sync:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        check_bits(out, expected_gather(world_size, call, shape, dtype, device), f"eager call {call} shape {tuple(shape)}")
        log(rank, f"  collective {1e3*(t1-t0):.2f} ms, check {1e3*(time.perf_counter()-t1):.1f} ms")
    return call, out


def _read_i32(ptr: int, n: int) -> list[int]:
    """Read n int32 from device address ptr (control block: counter, non-lamport flag, lamport flag, stride, clear)."""
    from ctypes import c_int32, c_void_p, cast

    from flashinfer.comm.cuda_ipc import cudart

    torch.cuda.synchronize()
    buf = (c_int32 * n)()
    cudart.cudaMemcpy(cast(buf, c_void_p), c_void_p(ptr), 4 * n)
    return list(buf)


def _read_bytes(ptr: int, n: int) -> bytes:
    from ctypes import c_void_p, cast, create_string_buffer

    from flashinfer.comm.cuda_ipc import cudart

    torch.cuda.synchronize()
    buf = create_string_buffer(n)
    cudart.cudaMemcpy(cast(buf, c_void_p), c_void_p(ptr), n)
    return buf.raw


def _memset(ptr: int, value: int, n: int) -> None:
    from ctypes import c_void_p

    from flashinfer.comm.cuda_ipc import cudart

    torch.cuda.synchronize()
    cudart.cudaMemset(c_void_p(ptr), value, n)
    torch.cuda.synchronize()


def _table_ptrs(ws: torch.Tensor, n: int, rank: int) -> tuple[int, int]:
    """(own Lamport data buffer, control block) addresses from a workspace pointer table."""
    t = ws.tolist()
    return t[2 * n + rank], t[3 * n]


# ----------------------------------------------------------------------------------------------------------------
# scenario 5 + 6 (partly): eager, bit-exact, several shapes, three passes over the slot rotation, with/without metadata
# ----------------------------------------------------------------------------------------------------------------
def s_eager(world_size, rank, dtype):
    device = torch.device("cuda", rank)
    handles, ws, meta = create_workspace(rank, world_size, dtype)
    calls = Calls()
    shapes = [(256, HIDDEN_DIM), (7, HIDDEN_DIM), (1, 4096), (64, HIDDEN_DIM), (3, 5, 8), (8, 3)]
    if dtype == torch.float32:
        shapes = [(128, HIDDEN_DIM), (7, HIDDEN_DIM), (1, 4096), (3, 5, 8)]
    for _pass in range(3):
        for shape in shapes:
            gather_checked(rank, world_size, ws, meta, dtype, device, shape, calls)
    for shape in shapes[:2]:  # the wrapper's own checks still apply when no metadata is passed
        gather_checked(rank, world_size, ws, None, dtype, device, shape, calls)
    dist.barrier()
    comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(handles)


# ----------------------------------------------------------------------------------------------------------------
# scenario 2: graph lengths 1/2/4/5 captured after 0/1/2 eager calls, replays interleaved with eager calls
# ----------------------------------------------------------------------------------------------------------------
GRAPH_SHAPES = [(64, HIDDEN_DIM), (3, 4096), (16, HIDDEN_DIM), (1, 4096), (5, HIDDEN_DIM)]


def _capture(rank, world_size, ws, meta, dtype, device, shapes, calls: Calls, **kw):
    """Record one trtllm_allgather per shape into a CUDA graph; returns (graph, xs, outs). Warm-up on the side
    stream first (executes the ops once; the data of that pass is also checked)."""
    xs = [torch.empty(s, dtype=dtype, device=device) for s in shapes]
    outs = [torch.empty((world_size,) + tuple(s), dtype=dtype, device=device) for s in shapes]
    stream = torch.cuda.Stream()
    warm = [calls.next() for _ in shapes]
    log(rank, f"capture: warm-up calls {warm} shapes {shapes} {kw}")
    with torch.cuda.stream(stream):
        for i, s in enumerate(shapes):
            xs[i].copy_(make_input(rank, warm[i], s, dtype, device))
            comm.trtllm_allgather(xs[i], outs[i], world_size, rank, ws, metadata=meta, **kw)
    torch.cuda.synchronize()
    for i, s in enumerate(shapes):
        check_bits(outs[i], expected_gather(world_size, warm[i], s, dtype, device), f"warm-up {i} shape {s}")
    dist.barrier()
    log(rank, "capture: recording")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for i, s in enumerate(shapes):
            comm.trtllm_allgather(xs[i], outs[i], world_size, rank, ws, metadata=meta, **kw)
    torch.cuda.synchronize()
    dist.barrier()
    log(rank, "capture: done")
    return graph, xs, outs


def _replay_checked(rank, world_size, graph, xs, outs, shapes, dtype, device, calls: Calls, label: str):
    ids = [calls.next() for _ in shapes]
    log(rank, f"{label}: replay calls {ids}")
    for i, s in enumerate(shapes):
        xs[i].copy_(make_input(rank, ids[i], s, dtype, device))
    graph.replay()
    torch.cuda.synchronize()
    for i, s in enumerate(shapes):
        check_bits(outs[i], expected_gather(world_size, ids[i], s, dtype, device), f"{label} replay call {i} shape {s}")


def s_graph_phases(world_size, rank, dtype):
    device = torch.device("cuda", rank)
    handles, ws, meta = create_workspace(rank, world_size, dtype)
    calls = Calls()
    for offset in (0, 1, 2):
        for ncalls in (1, 2, 4, 5):
            # every collective is one call id, so calls.n % 3 is the slot the next call will use;
            # advance explicitly to the intended phase (a fixed number of extra calls would not,
            # since each case below consumes 6*ncalls + 3 calls and keeps the phase it started with)
            while calls.n % 3 != offset:
                gather_checked(rank, world_size, ws, meta, dtype, device, (2, 4096), calls)
            log(rank, f"case offset={offset} ncalls={ncalls} starts at phase {calls.n % 3}")
            shapes = GRAPH_SHAPES[:ncalls]
            graph, xs, outs = _capture(rank, world_size, ws, meta, dtype, device, shapes, calls)
            for rep in range(3):
                _replay_checked(rank, world_size, graph, xs, outs, shapes, dtype, device, calls,
                                f"offset {offset} ncalls {ncalls} rep {rep}")
                gather_checked(rank, world_size, ws, meta, dtype, device, (9, 4096), calls)  # eager between replays
            graph.replay()  # two replays back to back, checked after the second
            _replay_checked(rank, world_size, graph, xs, outs, shapes, dtype, device, calls,
                            f"offset {offset} ncalls {ncalls} double")
            del graph
    dist.barrier()
    comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(handles)


# ----------------------------------------------------------------------------------------------------------------
# scenario 1: replace the workspace's peer buffers and control flags in place; replay the original graph
# ----------------------------------------------------------------------------------------------------------------
def s_workspace_replacement(world_size, rank, dtype, rounds: int = 3):
    """A graph recorded against workspace A must run on whatever the table points to after an in-place rewrite.
    Verified by state, not only by output: the retired workspace's control block and Lamport slots must stay
    untouched (poisoned slots keep their poison, flag does not advance) while the new workspace's flag advances by
    exactly the number of replayed calls and its own slot receives the pushed data."""
    device = torch.device("cuda", rank)
    handles_a, ws, meta = create_workspace(rank, world_size, dtype)
    keep = [handles_a]  # retired buffers stay mapped so a stale pointer would read/write them, not fault
    table_ptr = ws.data_ptr()
    calls = Calls()
    n = world_size
    stride = meta["lamport_comm_size"]
    shapes = [(64, HIDDEN_DIM), (7, 4096)]
    graph, xs, outs = _capture(rank, world_size, ws, meta, dtype, device, shapes, calls)
    _replay_checked(rank, world_size, graph, xs, outs, shapes, dtype, device, calls, "before replacement")
    cur_data, cur_ctrl = _table_ptrs(ws, n, rank)
    for rnd in range(1, rounds + 1):
        handles_b, ws_b, meta_b = create_workspace(rank, world_size, dtype)  # new buffers, new control block
        keep.append(handles_b)
        assert meta_b["lamport_comm_size"] == stride
        old_data, old_ctrl = cur_data, cur_ctrl
        new_data, new_ctrl = _table_ptrs(ws_b, n, rank)
        assert new_data != old_data and new_ctrl != old_ctrl
        old_flag_before = _read_i32(old_ctrl, 5)
        # retire A: poison every slot of my A buffer so any stale push from any rank's graph shows
        dist.barrier()  # nobody is still using A
        _memset(old_data, 0x55, 3 * stride)
        ws.copy_(ws_b)  # rewrite the pointer table in place: the graph baked only the table's address
        torch.cuda.synchronize()
        assert ws.data_ptr() == table_ptr, "pointer table moved"
        dist.barrier()  # all ranks switched before anyone replays
        ncalls = 0

        def check_controls(where: str) -> None:
            # own control blocks are written by this rank's kernels only, so after the stream sync inside the
            # replay/eager helpers they are deterministic: the retired one must not move, the new one must have
            # advanced by exactly the calls issued so far. Checked after EVERY step: a graph that still used the
            # retired control pointer leaves the new flag at 0 after the first 2-call replay, where the end-of-round
            # total (7 calls -> flag 1) would have masked it through the mod-3 wrap.
            old_now = _read_i32(old_ctrl, 5)
            new_now = _read_i32(new_ctrl, 5)
            assert old_now == old_flag_before, f"round {rnd} {where}: retired control block changed {old_flag_before} -> {old_now}"
            assert new_now[2] == ncalls % 3 and new_now[0] == 0, (
                f"round {rnd} {where}: new control block {new_now}, expected flag {ncalls % 3} after {ncalls} calls"
            )

        for rep in range(2):
            _replay_checked(rank, world_size, graph, xs, outs, shapes, dtype, device, calls, f"round {rnd} rep {rep}")
            ncalls += len(shapes)
            check_controls(f"after replay {rep}")  # first replay: 2 calls -> new flag must read 2
        gather_checked(rank, world_size, ws, meta_b, dtype, device, (5, HIDDEN_DIM), calls)  # eager on the new table
        ncalls += 1
        check_controls("after eager")
        _replay_checked(rank, world_size, graph, xs, outs, shapes, dtype, device, calls, f"round {rnd} after eager")
        ncalls += len(shapes)
        check_controls("after final replay")
        torch.cuda.synchronize()
        dist.barrier()  # every rank's kernels for this round have finished before inspecting shared buffers
        for slot in range(3):
            head = _read_bytes(old_data + slot * stride, 256)
            assert head == b"\x55" * 256, f"round {rnd}: retired slot {slot} was written after retirement"
        # the last replay's last call pushed every rank's slice into my new buffer at slot (ncalls-1) % 3
        last_shape = shapes[-1]
        numel = last_shape[0] * last_shape[1]
        slot = (ncalls - 1) % 3
        es = torch.tensor([], dtype=dtype).element_size()
        got = _read_bytes(new_data + slot * stride, numel * es * n)
        want = normalize_ref(expected_gather(world_size, calls.n, last_shape, dtype, device)).contiguous().cpu().view(torch.uint8).numpy().tobytes()
        assert got == want, f"round {rnd}: new workspace slot {slot} does not hold the last gathered data"
        cur_data, cur_ctrl = new_data, new_ctrl
    dist.barrier()
    del graph
    for h in keep:
        comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(h)


# ----------------------------------------------------------------------------------------------------------------
# scenario 3: back-to-back calls, no intermediate sync, irregular sizes, per-rank skew, alternation with allreduce
# ----------------------------------------------------------------------------------------------------------------
ASYNC_SHAPES = [(256, HIDDEN_DIM), (1, 4096), (200, HIDDEN_DIM), (1, 4096), (1, 4096), (1, 4096), (1, 4096),
                (3, HIDDEN_DIM), (130, HIDDEN_DIM), (2, 4096), (256, HIDDEN_DIM), (9, HIDDEN_DIM), (1, 8), (255, HIDDEN_DIM)]


def s_async_sequences(world_size, rank, dtype):
    device = torch.device("cuda", rank)
    handles, ws, meta = create_workspace(rank, world_size, dtype)
    calls = Calls()
    for skew in (False, True):
        ids = [calls.next() for _ in ASYNC_SHAPES]
        xs = [make_input(rank, c, s, dtype, device) for c, s in zip(ids, ASYNC_SHAPES)]
        outs = [torch.empty((world_size,) + s, dtype=dtype, device=device) for s in ASYNC_SHAPES]
        torch.cuda.synchronize()
        dist.barrier()
        for k, s in enumerate(ASYNC_SHAPES):  # nothing but kernel launches from here to the final sync
            if skew and k % 3 == rank % 3:
                torch.cuda._sleep(int(2e6) * (rank + 1))  # delay this rank's next call, keep collective order
            comm.trtllm_allgather(xs[k], outs[k], world_size, rank, ws, metadata=meta)
        torch.cuda.synchronize()
        for k, s in enumerate(ASYNC_SHAPES):
            check_bits(outs[k], expected_gather(world_size, ids[k], s, dtype, device), f"async skew={skew} call {k} shape {s}")
    # alternate with the one-shot allreduce on the same workspace: both advance the slot rotation by update(size*N)
    shape = (16, 4096)
    ar_in, ar_out, ar_ref, ag_out, ag_ids = [], [], [], [], []
    for _ in range(6):
        c = calls.next()
        ag_ids.append(c)
        ag_out.append(torch.empty((world_size,) + shape, dtype=dtype, device=device))
        c2 = calls.next()
        xin = torch.stack([make_input(r, c2, shape, dtype, device) for r in range(world_size)])
        xin = xin.nan_to_num(0.0, 0.0, 0.0).clamp(-64, 64)  # finite, small: the reduction must not overflow
        ar_in.append(xin[rank].contiguous())
        ar_ref.append(xin.float().sum(0))
        ar_out.append(torch.empty(shape, dtype=dtype, device=device))
    ag_in = [make_input(rank, c, shape, dtype, device) for c in ag_ids]
    torch.cuda.synchronize()
    dist.barrier()
    for i in range(6):
        comm.trtllm_allgather(ag_in[i], ag_out[i], world_size, rank, ws, metadata=meta)
        comm.trtllm_allreduce_fusion(
            allreduce_in=ar_in[i], world_size=world_size, world_rank=rank, token_num=shape[0], hidden_dim=shape[1],
            workspace_ptrs=ws, launch_with_pdl=False, trigger_completion_at_end=True, fp32_acc=True,
            pattern_code=AllReduceFusionPattern.kAllReduce, use_oneshot=True, allreduce_out=ar_out[i],
            residual_in=None, residual_out=None, norm_out=None, quant_out=None, scale_out=None, rms_gamma=None,
            rms_eps=None, scale_factor=None, layout_code=None,
        )
    torch.cuda.synchronize()
    for i in range(6):
        check_bits(ag_out[i], expected_gather(world_size, ag_ids[i], shape, dtype, device), f"alternation gather {i}")
        torch.testing.assert_close(ar_out[i].float(), ar_ref[i], rtol=2e-2, atol=2e-1,
                                   msg=lambda m, i=i: f"alternation allreduce {i}: {m}")
    dist.barrier()
    comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(handles)


# ----------------------------------------------------------------------------------------------------------------
# scenario 4: boundary sizes and alignment
# ----------------------------------------------------------------------------------------------------------------
def s_boundaries(world_size, rank, dtype):
    device = torch.device("cuda", rank)
    handles, ws, meta = create_workspace(rank, world_size, dtype)
    calls = Calls()
    vec = VEC[dtype]
    cap = capacity_elems(meta, world_size, dtype)
    sizes = sorted({vec, 2 * vec, 127 * vec, 128 * vec, 129 * vec, 1023 * vec, 1024 * vec, 1025 * vec,
                    4095 * vec, 4096 * vec, 4097 * vec, (148 * 1024 - 1) * vec, 148 * 1024 * vec,
                    (148 * 1024 + 1) * vec, (148 * 4096 - 1) * vec, 148 * 4096 * vec, (148 * 4096 + 1) * vec,
                    cap - vec, cap})
    for n in sizes:
        assert n <= cap
        gather_checked(rank, world_size, ws, meta, dtype, device, (n,), calls)
    # empty input: a documented no-op; the output is left untouched and the rotation is unaffected
    out = torch.full((world_size, 0), 0.0, dtype=dtype, device=device)
    comm.trtllm_allgather(torch.empty(0, dtype=dtype, device=device), out, world_size, rank, ws, metadata=meta)
    gather_checked(rank, world_size, ws, meta, dtype, device, (3, vec), calls)
    # aligned slices with a non-zero storage offset are fine
    base = torch.empty(16 * vec + 2 * vec, dtype=dtype, device=device)
    obase = torch.empty(world_size * 16 * vec + 2 * vec, dtype=dtype, device=device)
    c = calls.next()
    x = base[2 * vec:2 * vec + 16 * vec]
    x.copy_(make_input(rank, c, (16 * vec,), dtype, device))
    o = obase[2 * vec:2 * vec + world_size * 16 * vec].view(world_size, 16 * vec)
    comm.trtllm_allgather(x, o, world_size, rank, ws, metadata=meta)
    torch.cuda.synchronize()
    check_bits(o, expected_gather(world_size, c, (16 * vec,), dtype, device), "aligned slice")
    # rejected before any launch: odd storage offset (input or output), non-vector size, over capacity,
    # wrong output shape, wrong dtype for the workspace, wrong world_size against metadata
    good_out = torch.empty((world_size, 16 * vec), dtype=dtype, device=device)
    bad_cases = {
        "input offset 1 element": (base[1:1 + 16 * vec], good_out, meta, world_size),
        "output offset 1 element": (base[:16 * vec], obase[1:1 + world_size * 16 * vec].view(world_size, 16 * vec), meta, world_size),
        "numel not a vector multiple": (base[:vec + 1], torch.empty((world_size, vec + 1), dtype=dtype, device=device), meta, world_size),
        "one vector over capacity": (torch.empty(cap + vec, dtype=dtype, device=device),
                                     torch.empty((world_size, cap + vec), dtype=dtype, device=device), meta, world_size),
        "output shape mismatch": (base[:16 * vec], torch.empty((world_size, 8 * vec), dtype=dtype, device=device), meta, world_size),
        "world_size vs metadata": (base[:16 * vec], torch.empty((world_size + 1, 16 * vec), dtype=dtype, device=device), meta, world_size + 1),
        "dtype vs workspace": (torch.empty(16 * vec, dtype=torch.float32 if dtype != torch.float32 else torch.float16, device=device),
                               torch.empty((world_size, 16 * vec), dtype=torch.float32 if dtype != torch.float32 else torch.float16, device=device),
                               meta, world_size),
        "non-contiguous input": (torch.empty((16, 2 * vec), dtype=dtype, device=device)[:, ::2],
                                 torch.empty((world_size, 16, vec), dtype=dtype, device=device), meta, world_size),
    }
    for name, (xi, oi, m, wsz) in bad_cases.items():
        with pytest.raises(ValueError):
            comm.trtllm_allgather(xi, oi, wsz, rank, ws, metadata=m)
    torch.cuda.synchronize()  # nothing was launched: the device-side error state must be clean
    gather_checked(rank, world_size, ws, meta, dtype, device, (5, HIDDEN_DIM), calls)  # still healthy afterwards
    dist.barrier()
    comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(handles)


# ----------------------------------------------------------------------------------------------------------------
# scenario 6: execution variants
# ----------------------------------------------------------------------------------------------------------------
def s_variants(world_size, rank, dtype):
    device = torch.device("cuda", rank)
    handles, ws, meta = create_workspace(rank, world_size, dtype)
    calls = Calls()
    for pdl in (False, True):
        for trigger in (True, False):
            for shape in ((64, HIDDEN_DIM), (1, 4096), (7, HIDDEN_DIM)):
                gather_checked(rank, world_size, ws, meta, dtype, device, shape, calls,
                               launch_with_pdl=pdl, trigger_completion_at_end=trigger)
            graph, xs, outs = _capture(rank, world_size, ws, meta, dtype, device, GRAPH_SHAPES[:2], calls,
                                       launch_with_pdl=pdl, trigger_completion_at_end=trigger)
            _replay_checked(rank, world_size, graph, xs, outs, GRAPH_SHAPES[:2], dtype, device, calls,
                            f"pdl={pdl} trigger={trigger}")
            del graph
    dist.barrier()
    comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(handles)


# ----------------------------------------------------------------------------------------------------------------
# slot stride: a workspace whose tp*max_token*hidden*itemsize is not a multiple of 16 must still work call after call
# ----------------------------------------------------------------------------------------------------------------
def s_small_workspace(world_size, rank, dtype):
    device = torch.device("cuda", rank)
    handles, ws, meta = create_workspace(rank, world_size, dtype, max_token_num=1, hidden_dim=9)
    stride = meta["lamport_comm_size"]
    raw = world_size * 1 * 9 * torch.tensor([], dtype=dtype).element_size()
    assert stride % 16 == 0 and stride >= raw, f"slot stride {stride} (raw {raw})"
    calls = Calls()
    vec = VEC[dtype]
    cap = capacity_elems(meta, world_size, dtype)
    for _ in range(7):  # more than two full slot rotations; slots 1 and 2 were the misaligned ones
        gather_checked(rank, world_size, ws, meta, dtype, device, (vec,), calls)
    if cap >= 2 * vec:
        for _ in range(3):
            gather_checked(rank, world_size, ws, meta, dtype, device, (2 * vec,), calls)
    dist.barrier()
    comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(handles)


# ----------------------------------------------------------------------------------------------------------------
# resource lifecycle: the unified-API workspace must really release on destroy(); TP=2 -> destroy -> TP=4 in one process
# ----------------------------------------------------------------------------------------------------------------
def _unified_workspace(world_size, rank, dtype, group, max_token_num=64, hidden_dim=1024):
    from flashinfer.comm.mnnvl import TorchDistBackend

    return comm.create_allreduce_fusion_workspace(
        backend="trtllm", world_size=world_size, rank=rank, max_token_num=max_token_num, hidden_dim=hidden_dim,
        dtype=dtype, comm_backend=TorchDistBackend(group=group), group=group,
    )


def _gather_unified(ws_obj, world_size, rank, dtype, device, shape, calls: Calls, what: str):
    call = calls.next()
    x = make_input(rank, call, shape, dtype, device)
    out = torch.empty((world_size,) + tuple(shape), dtype=dtype, device=device)
    comm.trtllm_allgather(x, out, world_size, rank, ws_obj.workspace_tensor, metadata=ws_obj.metadata)
    torch.cuda.synchronize()
    check_bits(out, expected_gather(world_size, call, shape, dtype, device), what)


def s_resource_release(world_size, rank, dtype, iterations: int = 6):
    import gc
    import weakref

    from flashinfer.comm import trtllm_ar

    device = torch.device("cuda", rank)
    calls = Calls()
    frees = []
    for it in range(iterations):
        ws_obj = _unified_workspace(world_size, rank, dtype, dist.group.WORLD)
        key = id(ws_obj.ipc_handles)
        assert key in trtllm_ar._symm_workspace_refs, "registry entry missing after creation"
        blocks = [r for r in trtllm_ar._symm_workspace_refs[key] if isinstance(r, trtllm_ar._ControlBlock)]
        assert len(blocks) == 1 and blocks[0].ptr == ws_obj.workspace_tensor.tolist()[3 * world_size], (
            "the registry entry must own exactly the control block the pointer table points to"
        )
        control = blocks[0]
        handles_ref = [weakref.ref(h) for h in ws_obj.mem_handles]
        assert handles_ref, "unified trtllm workspace without symmetric-memory handles"
        _gather_unified(ws_obj, world_size, rank, dtype, device, (16, 1024), calls, f"iteration {it}")
        dist.barrier()
        ws_obj.destroy()
        assert key not in trtllm_ar._symm_workspace_refs, f"iteration {it}: registry still holds the workspace"
        assert ws_obj._internal_workspace is None
        assert control.freed, f"iteration {it}: control block not freed by destroy()"
        ws_obj.destroy()  # idempotent: no second free, no error
        del ws_obj
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier()
        alive = [r for r in handles_ref if r() is not None]
        assert not alive, f"iteration {it}: {len(alive)} SymmDeviceMemory handle(s) still alive after destroy()"
        frees.append(torch.cuda.mem_get_info(device)[0])
    # after warm-up the free device memory must not keep going down (64 MiB tolerance for allocator noise)
    assert frees[-1] >= frees[1] - (64 << 20), f"device memory not recovered across cycles: {[f >> 20 for f in frees]} MiB"


def s_reinit_tp2_then_tp4(world_size, rank, dtype):
    assert world_size == 4
    device = torch.device("cuda", rank)
    calls = Calls()
    sub = dist.new_group([0, 1])
    if rank < 2:
        ws2 = _unified_workspace(2, rank, dtype, sub)
        for _ in range(3):
            _gather_unified(ws2, 2, rank, dtype, device, (16, 1024), calls, "tp2 phase")
        dist.barrier(group=sub)
        ws2.destroy()
    else:
        for _ in range(3):
            calls.next()  # keep the call ids (= data patterns) identical on all ranks for the TP=4 phase
    dist.barrier()
    ws4 = _unified_workspace(4, rank, dtype, dist.group.WORLD)
    for _ in range(3):
        _gather_unified(ws4, 4, rank, dtype, device, (16, 1024), calls, "tp4 phase")
    dist.barrier()
    ws4.destroy()


# ----------------------------------------------------------------------------------------------------------------
# slot offsets beyond 2 GiB: clear slot 2 of a > 1 GiB slot (clear_offset * comm_size overflowed int32)
# ----------------------------------------------------------------------------------------------------------------
def s_large_slot(world_size, rank, dtype):
    device = torch.device("cuda", rank)
    # comm_size = world_size * max_token_num * hidden_dim * itemsize, just above 1 GiB so slot 2 starts above 2 GiB
    hidden = 8
    max_token = (1 << 30) // (world_size * hidden * 2) + 1
    handles, ws, meta = create_workspace(rank, world_size, dtype, max_token_num=max_token, hidden_dim=hidden)
    stride = meta["lamport_comm_size"]
    assert stride > (1 << 30) and 2 * stride > (1 << 31), f"slot stride {stride}"
    calls = Calls()
    vec = VEC[dtype]
    for shape in [(vec,), (64, hidden), (1024, hidden), (vec,), (4096, hidden), (vec,), (64, hidden)]:
        gather_checked(rank, world_size, ws, meta, dtype, device, shape, calls)  # 7 calls: every slot cleared twice
    dist.barrier()
    comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(handles)


# ----------------------------------------------------------------------------------------------------------------
# scenario 7: harness self-tests
# ----------------------------------------------------------------------------------------------------------------
def s_one_rank_raises(world_size, rank, dtype):
    if rank == 1:
        raise RuntimeError("deliberate failure on rank 1")
    dist.barrier()  # the healthy rank blocks here; the harness must still report and kill it


def s_one_rank_stalls(world_size, rank, dtype):
    if rank == 1:
        time.sleep(3600)
    dist.barrier()


# ----------------------------------------------------------------------------------------------------------------
# pytest entry points
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_eager_bit_exact(world_size, dtype):
    _need_gpus(world_size)
    run_workers(s_eager, world_size, dtype)


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_graph_lengths_and_phases(world_size, dtype):
    _need_gpus(world_size)
    run_workers(s_graph_phases, world_size, dtype)


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_workspace_replacement_without_recapture(world_size, dtype):
    _need_gpus(world_size)
    run_workers(s_workspace_replacement, world_size, dtype)


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_async_sequences_and_allreduce_alternation(world_size, dtype):
    _need_gpus(world_size)
    run_workers(s_async_sequences, world_size, dtype)


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_boundaries_and_alignment(world_size, dtype):
    _need_gpus(world_size)
    run_workers(s_boundaries, world_size, dtype)


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_execution_variants(world_size, dtype):
    _need_gpus(world_size)
    run_workers(s_variants, world_size, dtype)


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_small_workspace_slot_stride(world_size, dtype):
    _need_gpus(world_size)
    run_workers(s_small_workspace, world_size, dtype)


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_unified_workspace_destroy_releases_resources(world_size, dtype):
    _need_gpus(world_size)
    run_workers(s_resource_release, world_size, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_reinit_tp2_then_tp4_in_one_process(dtype):
    _need_gpus(4)
    run_workers(s_reinit_tp2_then_tp4, 4, dtype)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_slot_offsets_beyond_2gib(world_size, dtype):
    """Each rank allocates ~4 GiB (three > 1 GiB slots + the fusion buffer)."""
    _need_gpus(world_size)
    run_workers(s_large_slot, world_size, dtype, timeout_s=900)


def test_control_block_freed_exactly_once(monkeypatch):
    """The workspace's cudaMalloc'ed control block is owned by its registry entry and freed exactly once by the
    destroy function (CPU-only: the CUDA runtime is stubbed)."""
    from types import SimpleNamespace

    from flashinfer.comm import trtllm_ar

    freed: list[int] = []
    monkeypatch.setattr(trtllm_ar, "cudart", SimpleNamespace(cudaFree=lambda p: freed.append(p.value)))
    handles = [[1, 2], [3, 4], [5, 6]]
    block = trtllm_ar._ControlBlock(0xC0DE)
    monkeypatch.setitem(trtllm_ar._symm_workspace_refs, id(handles), [object(), block])
    comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(handles)
    comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(handles)  # second call: nothing left to free
    block.free()  # direct second free is also a no-op
    assert freed == [0xC0DE] and block.freed


def test_wrapper_rejects_misaligned_slot_stride():
    """A workspace whose slot stride is not a multiple of 16 B is refused before launch (CPU-only check)."""
    x = torch.zeros(8, dtype=torch.float16)
    out = torch.zeros((2, 8), dtype=torch.float16)
    meta = {"lamport_comm_size": 36, "tp_size": 2, "use_fp32_lamport": False}
    with pytest.raises(ValueError, match="slot stride"):
        comm.trtllm_allgather(x, out, 2, 0, torch.zeros(7, dtype=torch.int64), metadata=meta)


class _FakeProc:
    """Process stand-in: alive on the first scan, then exited with the given code (the race the harness must catch)."""

    def __init__(self, name: str, exitcode: int, alive_scans: int = 1):
        self.name, self._exit, self._alive, self.killed = name, exitcode, alive_scans, False

    def start(self):
        pass

    def is_alive(self):
        if self._alive > 0:
            self._alive -= 1
            return True
        return False

    @property
    def exitcode(self):
        return None if self._alive > 0 else self._exit

    def kill(self):
        self.killed = True

    def join(self, timeout=None):
        pass


def test_harness_catches_exit_between_scans():
    """Workers that die between the failed-worker scan and the all-stopped check must still fail the run."""
    with pytest.raises(HarnessFailure, match="noticed after join"):
        run_workers(s_eager, 2, torch.bfloat16, timeout_s=5, _procs=[_FakeProc("rank0", 1), _FakeProc("rank1", 1)])
    run_workers(s_eager, 2, torch.bfloat16, timeout_s=5, _procs=[_FakeProc("rank0", 0), _FakeProc("rank1", 0)])


def test_harness_reports_worker_failure():
    _need_gpus(2)
    t = time.monotonic()
    with pytest.raises(HarnessFailure, match="rank1 exited"):
        run_workers(s_one_rank_raises, 2, torch.bfloat16, timeout_s=120)
    assert time.monotonic() - t < 90, "a failing worker must be reported promptly, not at the deadline"


def test_harness_reports_stalled_worker():
    _need_gpus(2)
    with pytest.raises(HarnessFailure, match="timeout"):
        run_workers(s_one_rank_stalls, 2, torch.bfloat16, timeout_s=25)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-rA", "-v"]))
