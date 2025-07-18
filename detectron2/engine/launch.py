# Copyright (c) Facebook, Inc. and its affiliates.
import os
import logging
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from detectron2.utils import comm

__all__ = ["DEFAULT_TIMEOUT", "launch"]

DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def _detect_backend():
    """
    Detect whether to use Accelerate or standard PyTorch distributed backend.
    """
    # Check if we're being launched by accelerate
    use_accelerate = (
        "ACCELERATE_USE_CPU" in os.environ or 
        "LOCAL_RANK" in os.environ or
        "ACCELERATE_MIXED_PRECISION" in os.environ or
        "ACCELERATE_GRADIENT_ACCUMULATION_STEPS" in os.environ
    )
    
    if use_accelerate:
        return "Accelerate"
    else:
        return "Torch"


def launch(
    main_func,
    # Should be num_processes_per_machine, but kept for compatibility.
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
    use_accelerate=None,
):
    """
    Launch multi-process or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of processes per machine. When
            using GPUs, this should be the number of GPUs.
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
        use_accelerate (bool): whether to use HuggingFace Accelerate for launching.
                              If None, will auto-detect based on environment variables.
    """
    # Auto-detect backend if not specified
    if use_accelerate is None:
        backend = _detect_backend()
        use_accelerate = (backend == "Accelerate")
    
    world_size = num_machines * num_gpus_per_machine
    
    if use_accelerate:
        # When using accelerate, we don't handle multi-process spawning ourselves
        # The user should use `accelerate launch` command
        logger = logging.getLogger(__name__)
        if world_size > 1:
            logger.info(
                "Accelerate backend detected. Make sure you're using 'accelerate launch' "
                "command to start training. Detectron2's launch() will not spawn processes."
            )
        # Just call the main function directly when using accelerate
        main_func(*args)
    elif world_size > 1:
        # Standard PyTorch distributed training
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )

        mp.start_processes(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                args,
                timeout,
            ),
            daemon=False,
        )
    else:
        main_func(*args)


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        assert num_gpus_per_machine <= torch.cuda.device_count()
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL" if has_gpu else "GLOO",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    # Setup the local process group.
    comm.create_local_process_group(num_gpus_per_machine)
    if has_gpu:
        torch.cuda.set_device(local_rank)

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    main_func(*args)
