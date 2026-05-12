"""
Environment Capture Script
===========================
Comprehensive utility to log system specifications, driver versions,
and library dependencies for reproducible environments.

Generates env_snapshot.json with:
  - System: platform, hostname, OS, CPU, RAM, Python version
  - GPU: model, architecture, VRAM, driver version, CUDA toolkit
  - Software: Rust/cargo version, git commit/branch/dirty, package versions
  - Build: compiler toolchain, maturin version

Usage:
    python scripts/capture_env.py [--output env_snapshot.json]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, timeout=10):
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            shell=isinstance(cmd, str),
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def capture_system():
    info = {
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
        info["ram_available_gb"] = round(mem.available / (1024 ** 3), 2)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_freq_mhz"] = psutil.cpu_freq().current if psutil.cpu_freq() else None
    except ImportError:
        info["ram_total_gb"] = None
    return info


def capture_gpu():
    gpu_info = {"driver_version": None, "cuda_version": None, "gpus": []}

    nvidia_smi = run_cmd("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,"
                         "pci.bus_id,compute_cap --format=csv,noheader,nounits")
    if nvidia_smi:
        for line in nvidia_smi.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpu_info["gpus"].append({
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mb": int(float(parts[2])),
                    "memory_free_mb": int(float(parts[3])),
                    "pci_bus_id": parts[4],
                    "compute_capability": parts[5],
                })

    driver_ver = run_cmd(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader"
    )
    if driver_ver:
        gpu_info["driver_version"] = driver_ver.split("\n")[0].strip()

    cuda_ver = run_cmd("nvcc --version")
    if cuda_ver:
        for line in cuda_ver.split("\n"):
            if "release" in line.lower():
                import re
                m = re.search(r"release\s+([\d.]+)", line)
                if m:
                    gpu_info["cuda_version"] = m.group(1)
                    break

    if not gpu_info["cuda_version"]:
        cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_path:
            version_file = os.path.join(cuda_path, "version.txt")
            if os.path.exists(version_file):
                with open(version_file) as f:
                    gpu_info["cuda_version"] = f.read().strip()

    return gpu_info


def capture_software():
    info = {}

    cargo_ver = run_cmd("cargo --version")
    info["cargo_version"] = cargo_ver

    rustc_ver = run_cmd("rustc --version")
    info["rustc_version"] = rustc_ver

    rustup_show = run_cmd("rustup show")
    if rustup_show:
        for line in rustup_show.split("\n"):
            line = line.strip()
            if "default" in line.lower() and ("x86" in line or "aarch" in line):
                info["rustup_host_triple"] = line.split()[-1]
                break

    maturin_ver = run_cmd("maturin --version")
    info["maturin_version"] = maturin_ver

    git_info = {}
    project_root = str(Path(__file__).resolve().parent.parent)
    git_info["commit"] = run_cmd("git rev-parse HEAD", timeout=5)
    git_info["commit_short"] = run_cmd("git rev-parse --short HEAD", timeout=5)
    git_info["branch"] = run_cmd("git rev-parse --abbrev-ref HEAD", timeout=5)
    git_info["describe"] = run_cmd("git describe --tags --always --dirty", timeout=5)
    dirty = run_cmd("git status --porcelain", timeout=5)
    git_info["dirty"] = bool(dirty) if dirty is not None else None
    git_info["remote_url"] = run_cmd("git remote get-url origin", timeout=5)
    info["git"] = git_info

    info["packages"] = {}
    pkg_list = [
        "numpy", "torch", "cupy", "pycuda", "psutil",
        "ternary_zero", "ternary_zero_core",
    ]
    for pkg in pkg_list:
        ver = run_cmd(f'{sys.executable} -c "import {pkg}; print({pkg}.__version__)"')
        if ver:
            info["packages"][pkg] = ver

    info["pip_freeze"] = []
    freeze = run_cmd(f"{sys.executable} -m pip freeze")
    if freeze:
        info["pip_freeze"] = freeze.split("\n")

    return info


def capture_build_env():
    info = {}
    cc = run_cmd("cc --version") or run_cmd("cl")
    info["c_compiler"] = cc.split("\n")[0] if cc else None

    try:
        import torch
        info["torch_cuda_arch_list"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
        info["torch_cuda_version"] = str(torch.version.cuda) if torch.cuda.is_available() else None
    except ImportError:
        pass

    info["environment_variables"] = {}
    env_keys = [
        "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
        "CUDNN_VERSION", "NCCL_VERSION",
        "TORCH_CUDA_ARCH_LIST", "CARGO_TARGET_DIR",
        "PYO3_PYTHON", "RUST_BACKTRACE",
    ]
    for key in env_keys:
        val = os.environ.get(key)
        if val is not None:
            info["environment_variables"][key] = val

    return info


def capture_cuda_device_props():
    props = {}
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                d = torch.cuda.get_device_properties(i)
                props[f"gpu_{i}"] = {
                    "name": d.name,
                    "total_memory_mb": d.total_mem // (1024 * 1024),
                    "major": d.major,
                    "minor": d.minor,
                    "multi_processor_count": d.multi_processor_count,
                    "max_threads_per_multi_processor": d.max_threads_per_multi_processor,
                    "max_threads_per_block": d.max_threads_per_block,
                    "warp_size": d.warp_size,
                    "max_shared_memory_per_block": d.max_shared_memory_per_block,
                    "max_registers_per_block": d.regs_per_block,
                }
    except Exception:
        pass

    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            for i in range(cupy.cuda.runtime.getDeviceCount()):
                with cupy.cuda.Device(i):
                    p = cupy.cuda.runtime.getDeviceProperties(i)
                    key = f"gpu_{i}_cupy"
                    props[key] = {
                        "name": p["name"].decode() if isinstance(p["name"], bytes) else p["name"],
                        "compute_capability": f"{p['major']}.{p['minor']}",
                        "multi_processor_count": p["multiProcessorCount"],
                        "total_global_memory_mb": p["totalGlobalMem"] // (1024 * 1024),
                        "shared_mem_per_block": p["sharedMemPerBlock"],
                        "regs_per_block": p["regsPerBlock"],
                        "warp_size": p["warpSize"],
                        "max_threads_per_block": p["maxThreadsPerBlock"],
                        "max_threads_per_sm": p["maxThreadsPerMultiProcessor"],
                        "clock_rate_khz": p["clockRate"],
                        "memory_clock_rate_khz": p["memoryClockRate"],
                        "memory_bus_width_bits": p["memoryBusWidth"],
                        "l2_cache_size_kb": p["l2CacheSize"] // 1024,
                    }
    except Exception:
        pass

    return props


def capture_all():
    return {
        "schema_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": capture_system(),
        "gpu": capture_gpu(),
        "cuda_device_properties": capture_cuda_device_props(),
        "software": capture_software(),
        "build_environment": capture_build_env(),
    }


def format_report(snapshot):
    lines = [
        "=" * 70,
        "Environment Snapshot",
        "=" * 70,
        f"Timestamp: {snapshot['timestamp']}",
        "",
        "--- System ---",
    ]
    for k, v in snapshot["system"].items():
        lines.append(f"  {k}: {v}")

    lines += ["", "--- GPU ---"]
    gpu = snapshot["gpu"]
    lines.append(f"  Driver: {gpu.get('driver_version', 'N/A')}")
    lines.append(f"  CUDA Toolkit: {gpu.get('cuda_version', 'N/A')}")
    for i, g in enumerate(gpu.get("gpus", [])):
        lines.append(f"  GPU {i}: {g['name']} ({g['memory_total_mb']} MB, "
                      f"CC {g['compute_capability']})")

    lines += ["", "--- Software ---"]
    sw = snapshot["software"]
    lines.append(f"  Rust: {sw.get('rustc_version', 'N/A')}")
    lines.append(f"  Cargo: {sw.get('cargo_version', 'N/A')}")
    lines.append(f"  Maturin: {sw.get('maturin_version', 'N/A')}")
    git = sw.get("git", {})
    lines.append(f"  Git Branch: {git.get('branch', 'N/A')}")
    lines.append(f"  Git Commit: {git.get('commit_short', 'N/A')}")
    lines.append(f"  Git Dirty: {git.get('dirty', 'N/A')}")
    for pkg, ver in sw.get("packages", {}).items():
        lines.append(f"  {pkg}: {ver}")

    lines += ["", "--- Build Environment ---"]
    be = snapshot["build_environment"]
    lines.append(f"  C Compiler: {be.get('c_compiler', 'N/A')}")
    for k, v in be.get("environment_variables", {}).items():
        lines.append(f"  {k}: {v}")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Capture environment snapshot for reproducible benchmarks"
    )
    parser.add_argument(
        "--output", "-o",
        default="env_snapshot.json",
        help="Output JSON file path (default: env_snapshot.json)",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root directory (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output",
    )
    args = parser.parse_args()

    if args.project_root:
        os.chdir(args.project_root)

    snapshot = capture_all()

    if not args.quiet:
        print(format_report(snapshot))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)

    if not args.quiet:
        print(f"\nSnapshot saved to: {output_path.resolve()}")

    return snapshot


if __name__ == "__main__":
    main()
