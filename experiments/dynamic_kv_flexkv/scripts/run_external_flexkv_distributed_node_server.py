#!/usr/bin/env python3
"""Run one externally owned distributed FlexKV server per node."""

import argparse
import os
import subprocess

import torch

from flexkv.common.config import (
    CacheConfig,
    ModelConfig,
    RankInfo,
    load_user_config_from_file,
    update_default_config_from_user_config,
)
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
    parser.add_argument(
        "--config-path",
        default=os.getenv("FLEXKV_CONFIG_PATH"),
        help="FlexKV YAML/JSON config; defaults to FLEXKV_CONFIG_PATH",
    )
    parser.add_argument("--packed-kv", action="store_true")
    parser.add_argument("--start-mps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config_path:
        raise ValueError("--config-path or FLEXKV_CONFIG_PATH is required")

    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if args.start_mps:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
        print("external_flexkv_distributed_server mps_started=true", flush=True)

    device_count = torch.cuda.device_count()
    print(
        f"external_flexkv_distributed_server visible_devices={visible_devices!r} "
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
            f"external_flexkv_distributed_server gpu={device_id} "
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
    cache_config = CacheConfig(tokens_per_block=args.tokens_per_block)
    user_config = load_user_config_from_file(args.config_path)
    update_default_config_from_user_config(
        RankInfo(model_config=model_config),
        cache_config,
        user_config,
    )

    if not cache_config.enable_kv_sharing:
        raise RuntimeError(
            "distributed FlexKV server requires enable_p2p_cpu, "
            "enable_p2p_ssd, or enable_3rd_remote"
        )

    print(
        f"external_flexkv_distributed_server config_path={args.config_path}",
        flush=True,
    )
    print(
        f"external_flexkv_distributed_server model_config={model_config}",
        flush=True,
    )
    print(
        f"external_flexkv_distributed_server cache_config={cache_config}",
        flush=True,
    )
    server = KVServer(
        model_config=model_config,
        cache_config=cache_config,
        gpu_register_port=args.gpu_register_port,
        server_recv_port=args.server_recv_port,
    )
    server.run()


if __name__ == "__main__":
    main()
