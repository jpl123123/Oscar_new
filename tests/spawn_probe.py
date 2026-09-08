"""CPU-only two-level process probe: not an Ascend/NPU acceptance test."""

import json
import multiprocessing as mp
import os
import threading

from oscar_ascend.runtime_env import configure_process_environment

INHERITED_MARKER = False


def warm_runtime():
    global INHERITED_MARKER
    import torch

    INHERITED_MARKER = True
    torch.set_num_threads(2)
    value = torch.arange(16.0, requires_grad=True)
    (value * value).sum().backward()
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, daemon=True)
    thread.start()
    return stop, thread, value.grad.sum().item()


def context():
    ctx = mp.get_context(os.environ["VLLM_WORKER_MULTIPROC_METHOD"])
    # Never exercise the unsafe fork path intentionally in this test.
    assert ctx.get_start_method() == "spawn"
    return ctx


def worker_level(writer):
    fresh = not INHERITED_MARKER
    stop, thread, value = warm_runtime()
    try:
        writer.send(
            {
                "fresh": fresh,
                "method": mp.get_start_method(),
                "value": value,
                "devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                "pid": os.getpid(),
            }
        )
    finally:
        stop.set()
        thread.join()
        writer.close()


def engine_level(writer):
    fresh = not INHERITED_MARKER
    stop, thread, _ = warm_runtime()
    reader, child_writer = context().Pipe(duplex=False)
    worker = context().Process(target=worker_level, args=(child_writer,))
    try:
        worker.start()
        child_writer.close()
        assert reader.poll(30), "Spawned worker did not respond"
        result = reader.recv()
        worker.join(10)
        assert worker.exitcode == 0
        writer.send(
            {
                "fresh_engine": fresh,
                "engine_method": mp.get_start_method(),
                "engine_pid": os.getpid(),
                "worker": result,
            }
        )
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
        stop.set()
        thread.join()
        reader.close()
        child_writer.close()
        writer.close()


def main():
    configure_process_environment()
    stop, thread, _ = warm_runtime()
    reader, writer = context().Pipe(duplex=False)
    engine = context().Process(target=engine_level, args=(writer,))
    try:
        engine.start()
        writer.close()
        assert reader.poll(45), "Spawned engine did not respond"
        result = reader.recv()
        engine.join(10)
        assert engine.exitcode == 0
        print(json.dumps(result))
    finally:
        if engine.is_alive():
            engine.terminate()
            engine.join(5)
        stop.set()
        thread.join()
        reader.close()
        writer.close()


if __name__ == "__main__":
    main()
