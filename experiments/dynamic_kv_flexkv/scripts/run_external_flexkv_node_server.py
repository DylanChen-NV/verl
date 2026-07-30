#!/usr/bin/env python3
"""Run one externally owned FlexKV server for all vLLM instances on a node."""

import argparse
import os
import subprocess

import torch

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.server.server import KVServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-recv-port", required=True)
    parser.add_argument("--gpu-register-port", required=True)
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--instance-num", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-size", type=int, required=True)
    parser.add_argument("--tokens-per-block", type=int, default=16)
    parser.add_argument("--num-cpu-blocks", type=int, required=True)
    parser.add_argument("--packed-kv", action="store_true")
    parser.add_argument("--start-mps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if args.start_mps:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
        print("external_flexkv_server mps_started=true", flush=True)

    device_count = torch.cuda.device_count()
    print(
        f"external_flexkv_server visible_devices={visible_devices!r} "
        f"device_count={device_count} expected_gpus={args.expected_gpus}",
        flush=True,
    )
    if device_count != args.expected_gpus:
        raise RuntimeError(
            f"external FlexKV server needs {args.expected_gpus} GPUs, "
            f"but torch reports {device_count}"
        )

    torch.cuda.init()
    for device_id in range(args.expected_gpus):
        print(
            f"external_flexkv_server gpu={device_id} "
            f"name={torch.cuda.get_device_name(device_id)}",
            flush=True,
        )

    model_config = ModelConfig(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        packed_kv=args.packed_kv,
        dtype=torch.bfloat16,
        tp_size=args.tp_size,
        pp_size=1,
        dp_size=1,
        nnodes=1,
        instance_num=args.instance_num,
    )
    model_config.freeze()
    cache_config = CacheConfig(
        tokens_per_block=args.tokens_per_block,
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=False,
        num_cpu_blocks=args.num_cpu_blocks,
    )

    print(f"external_flexkv_server model_config={model_config}", flush=True)
    print(f"external_flexkv_server cache_config={cache_config}", flush=True)
    server = KVServer(
        model_config=model_config,
        cache_config=cache_config,
        gpu_register_port=args.gpu_register_port,
        server_recv_port=args.server_recv_port,
    )
    server.run()


if __name__ == "__main__":
    main()
