import multiprocessing as mp
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.utils import get_compute_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda:0"))[0] not in (9, 10, 12),
    reason="trtllm_comm kernels support SM90/SM100/SM12x only",
)

# Largest shape: 256 tokens x a 38,720-wide vocab shard (GLM-5.3-Flash lm_head at TP=4) -> 19 MiB per rank.
MAX_TOKEN_NUM = 256
HIDDEN_DIM = 38720
SHAPES = [(256, 38720), (7, 38720), (1, 4096), (64, 38720)]


def _make_input(rank: int, call: int, shape, dtype, device) -> torch.Tensor:
    """Deterministic per-(rank, call) input so every rank can build the expected gather locally."""
    g = torch.Generator(device="cpu").manual_seed(1000 * call + rank)
    x = torch.randn(shape, generator=g, dtype=torch.float32) * 3.0
    x.view(-1)[:16] = -0.0  # the Lamport sentinel value must come out as +0.0
    return x.to(dtype).to(device)


def _expected(world_size: int, call: int, shape, dtype, device) -> torch.Tensor:
    return torch.stack([_make_input(r, call, shape, dtype, device) for r in range(world_size)])


def _run_allgather_worker(world_size, rank, dtype, hidden_dim, distributed_init_port, gpu_offset=0):
    device = torch.device(f"cuda:{rank + gpu_offset}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    try:
        ipc_handles, workspace, metadata = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
            rank,
            world_size,
            MAX_TOKEN_NUM,
            hidden_dim,
            use_fp32_lamport=(dtype == torch.float32),
            group=group,
            create_metadata=True,
        )
        call = 0
        # eager: 3 passes over the shapes exercise the mod-3 Lamport slot rotation and the clear path
        for _ in range(3):
            for shape in SHAPES:
                x = _make_input(rank, call, shape, dtype, device)
                out = torch.empty((world_size,) + shape, dtype=dtype, device=device)
                comm.trtllm_allgather(x, out, world_size, rank, workspace, metadata=metadata)
                torch.cuda.synchronize()
                torch.testing.assert_close(out, _expected(world_size, call, shape, dtype, device), rtol=0, atol=0)
                call += 1
        # CUDA graph: three calls recorded once, replayed with changing inputs
        shape = SHAPES[0]
        xs = [torch.empty(shape, dtype=dtype, device=device) for _ in range(3)]
        outs = [torch.empty((world_size,) + shape, dtype=dtype, device=device) for _ in range(3)]
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for i in range(3):
                xs[i].copy_(_make_input(rank, call + i, shape, dtype, device))
                comm.trtllm_allgather(xs[i], outs[i], world_size, rank, workspace, metadata=metadata)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for i in range(3):
                comm.trtllm_allgather(xs[i], outs[i], world_size, rank, workspace, metadata=metadata)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        for replay in range(5):
            base = call + 100 + 3 * replay
            for i in range(3):
                xs[i].copy_(_make_input(rank, base + i, shape, dtype, device))
            graph.replay()
            torch.cuda.synchronize()
            for i in range(3):
                torch.testing.assert_close(outs[i], _expected(world_size, base + i, shape, dtype, device), rtol=0, atol=0)
        # capacity check must refuse a message larger than the Lamport slot
        too_big = torch.empty((MAX_TOKEN_NUM + 1, hidden_dim), dtype=dtype, device=device)
        with pytest.raises(ValueError):
            comm.trtllm_allgather(
                too_big,
                torch.empty((world_size,) + too_big.shape, dtype=dtype, device=device),
                world_size,
                rank,
                workspace,
                metadata=metadata,
            )
        dist.barrier(group=group)
        comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(ipc_handles, group=group)
    finally:
        dist.barrier(group=group)
        dist.destroy_process_group(group=group)


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(world_size: int, dtype: torch.dtype, hidden_dim: int, test_target: Any) -> None:
    mp.set_start_method("spawn", force=True)
    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc = mp.Process(
            target=test_target,
            args=(world_size, i, dtype, hidden_dim, distributed_init_port),
            name=f"Worker-{i}",
        )
        proc.start()
        procs.append(proc)
    for i in range(world_size):
        procs[i].join()
        assert procs[i].exitcode == 0, f"Process {i} failed with exit code {procs[i].exitcode}"


# Run as: python tests/comm/test_trtllm_allgather.py
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trtllm_allgather(world_size, dtype):
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(f"world_size {world_size} is greater than available_gpus {available_gpus}")
    multi_process_parallel(world_size, dtype, HIDDEN_DIM, _run_allgather_worker)
    print(f"trtllm_allgather tp = {world_size} ({dtype}): OK")


if __name__ == "__main__":
    for ws in (2, 4):
        for dt in (torch.float16, torch.bfloat16):
            test_trtllm_allgather(ws, dt)
