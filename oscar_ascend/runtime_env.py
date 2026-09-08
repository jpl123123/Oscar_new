"""Process policy applied before importing TorchNPU or vLLM."""

import os


def configure_process_environment(environment=None):
    """Pin owned devices and avoid inheriting accelerator/OpenMP state via fork.

    vLLM's EngineCore and tensor-parallel workers both use get_mp_context(),
    which reads VLLM_WORKER_MULTIPROC_METHOD. Set it before envs are cached.
    """
    environment = os.environ if environment is None else environment
    environment["ASCEND_RT_VISIBLE_DEVICES"] = "4,5,6,7"
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return environment
