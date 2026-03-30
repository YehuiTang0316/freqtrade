#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
USER_DATA_DIR = ROOT / "user_data"
OPS_DIR = ROOT / "ops" / "multi_dryrun"
REGISTRY_PATH = OPS_DIR / "registry.json"
GENERATED_DIR = OPS_DIR / "generated"
COMPOSE_PATH = GENERATED_DIR / "compose.yml"
INSTANCE_ROOT = USER_DATA_DIR / "dryrun_instances"


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    if not path.is_file():
        fail(f"registry file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(data: dict[str, Any], path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        fail(f"invalid instance name: {name!r}")
    return slug


def normalize_repo_relative(raw_path: str) -> str:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(ROOT).as_posix()
        except ValueError as exc:
            fail(f"path must live inside repo: {raw_path} ({exc})")
    return candidate.as_posix().lstrip("./")


def ensure_user_data_relative(repo_rel: str) -> PurePosixPath:
    absolute = (ROOT / repo_rel).resolve()
    if not absolute.exists():
        fail(f"path does not exist: {repo_rel}")
    try:
        user_rel = absolute.relative_to(USER_DATA_DIR.resolve())
    except ValueError:
        fail(f"path must live under user_data/: {repo_rel}")
    return PurePosixPath("/freqtrade/user_data") / PurePosixPath(user_rel.as_posix())


def runtime_rel(name: str) -> str:
    return f"user_data/dryrun_instances/{slugify(name)}"


def runtime_abs(name: str) -> Path:
    return INSTANCE_ROOT / slugify(name)


def service_name(name: str) -> str:
    return f"dryrun-{slugify(name)}"


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def yaml_scalar(value: Any) -> str:
    return json.dumps(value)


def next_host_port(registry: dict[str, Any]) -> int:
    start = int(registry.get("host_port_start", 18080))
    used = {
        int(instance["host_port"])
        for instance in registry.get("instances", [])
        if "host_port" in instance
    }
    port = start
    while port in used:
        port += 1
    return port


def validate_registry(registry: dict[str, Any]) -> None:
    names: set[str] = set()
    ports: set[int] = set()

    default_config = registry.get("default_config")
    if default_config:
        ensure_user_data_relative(normalize_repo_relative(default_config))

    for instance in registry.get("instances", []):
        name = instance.get("name")
        strategy = instance.get("strategy")
        host_port = instance.get("host_port")
        configs = instance.get("configs") or []

        if not name or not strategy:
            fail(f"instance entries require name and strategy: {instance}")

        slug = slugify(name)
        if slug in names:
            fail(f"duplicate instance name: {name}")
        names.add(slug)

        if host_port is None:
            fail(f"instance {name} is missing host_port")
        host_port = int(host_port)
        if host_port in ports:
            fail(f"duplicate host_port: {host_port}")
        ports.add(host_port)

        if not configs and not default_config:
            fail(f"instance {name} has no configs and registry has no default_config")

        for config in configs:
            ensure_user_data_relative(normalize_repo_relative(config))


def build_overlay(registry: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    internal_port = int(registry.get("internal_api_port", 8080))
    overlay = {
        "dry_run": True,
        "bot_name": instance.get("bot_name", f"dryrun-{slugify(instance['name'])}"),
        "initial_state": instance.get("initial_state", "running"),
        "api_server": {
            "enabled": True,
            "listen_ip_address": "0.0.0.0",
            "listen_port": internal_port,
            "verbosity": "error",
            "jwt_secret_key": registry.get(
                "jwt_secret_key", "freqtrade-local-dev-key-change-me"
            ),
            "CORS_origins": [],
            "username": registry.get("username", "freqtrader"),
            "password": registry.get("password", "freqtrader"),
        },
    }
    overrides = instance.get("overrides") or {}
    if overrides:
        overlay = deep_merge(overlay, overrides)
    return overlay


def render_instance_files(registry: dict[str, Any]) -> None:
    INSTANCE_ROOT.mkdir(parents=True, exist_ok=True)
    for instance in registry.get("instances", []):
        name = instance["name"]
        runtime_dir = runtime_abs(name)
        logs_dir = runtime_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = runtime_dir / "instance.json"
        overlay = build_overlay(registry, instance)
        overlay_path.write_text(json.dumps(overlay, indent=2) + "\n", encoding="utf-8")


def build_command(registry: dict[str, Any], instance: dict[str, Any]) -> list[str]:
    name = instance["name"]
    command = [
        "trade",
        "--logfile",
        f"/freqtrade/{runtime_rel(name)}/logs/freqtrade.log",
        "--db-url",
        f"sqlite:////freqtrade/{runtime_rel(name)}/tradesv3.sqlite",
    ]

    config_paths = [normalize_repo_relative(path) for path in instance.get("configs", [])]
    if not config_paths:
        config_paths = [normalize_repo_relative(registry["default_config"])]

    for config in config_paths:
        command.extend(["--config", str(ensure_user_data_relative(config))])

    command.extend(
        [
            "--config",
            f"/freqtrade/{runtime_rel(name)}/instance.json",
            "--strategy",
            instance["strategy"],
        ]
    )
    command.extend(instance.get("extra_args", []))
    return command


def render_compose(registry: dict[str, Any], compose_path: Path = COMPOSE_PATH) -> Path:
    validate_registry(registry)
    render_instance_files(registry)
    compose_path.parent.mkdir(parents=True, exist_ok=True)

    project_name = registry.get("project_name", "freqtrade-dryrun")
    image = registry.get("image", "freqtradeorg/freqtrade:stable")
    internal_port = int(registry.get("internal_api_port", 8080))
    bind_address = registry.get("bind_address", "127.0.0.1")
    user_data_source = USER_DATA_DIR.resolve().as_posix()

    lines = [f"name: {yaml_scalar(project_name)}", "services:"]
    for instance in registry.get("instances", []):
        host_port = int(instance["host_port"])
        name = instance["name"]
        lines.extend(
            [
                f"  {service_name(name)}:",
                f"    image: {yaml_scalar(image)}",
                "    restart: unless-stopped",
                f"    container_name: {yaml_scalar(f'{project_name}-{slugify(name)}')}",
                "    volumes:",
                "      - type: bind",
                f"        source: {yaml_scalar(user_data_source)}",
                f"        target: {yaml_scalar('/freqtrade/user_data')}",
                "    ports:",
                f"      - {yaml_scalar(f'{bind_address}:{host_port}:{internal_port}')}",
                "    command:",
            ]
        )
        for arg in build_command(registry, instance):
            lines.append(f"      - {yaml_scalar(arg)}")

    compose_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return compose_path


def require_instance_names(
    registry: dict[str, Any], names: list[str] | None, allow_empty: bool = False
) -> list[str]:
    available = {slugify(instance["name"]): instance["name"] for instance in registry["instances"]}
    if not names:
        if allow_empty:
            return []
        return [instance["name"] for instance in registry["instances"]]

    selected: list[str] = []
    for raw_name in names:
        slug = slugify(raw_name)
        if slug not in available:
            fail(f"unknown instance: {raw_name}")
        selected.append(available[slug])
    return selected


def run_compose(compose_path: Path, args: list[str]) -> int:
    command = ["docker", "compose", "-f", str(compose_path), *args]
    completed = subprocess.run(command, cwd=ROOT)
    return completed.returncode


def cmd_render(_: argparse.Namespace) -> int:
    compose_path = render_compose(load_registry())
    print(compose_path.relative_to(ROOT).as_posix())
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    registry = load_registry()
    validate_registry(registry)
    instances = registry.get("instances", [])
    if not instances:
        print("No instances configured.")
        return 0

    print("NAME\tSTRATEGY\tPORT\tURL\tCONFIGS")
    for instance in instances:
        configs = instance.get("configs") or [registry.get("default_config", "")]
        url = f"http://127.0.0.1:{int(instance['host_port'])}"
        print(
            "\t".join(
                [
                    instance["name"],
                    instance["strategy"],
                    str(instance["host_port"]),
                    url,
                    ",".join(configs),
                ]
            )
        )
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    compose_path = render_compose(load_registry())
    return run_compose(compose_path, ["ps"])


def cmd_up(args: argparse.Namespace) -> int:
    registry = load_registry()
    compose_path = render_compose(registry)
    names = require_instance_names(registry, args.names)
    return run_compose(compose_path, ["up", "-d", *[service_name(name) for name in names]])


def cmd_stop(args: argparse.Namespace) -> int:
    registry = load_registry()
    compose_path = render_compose(registry)
    names = require_instance_names(registry, args.names)
    return run_compose(compose_path, ["stop", *[service_name(name) for name in names]])


def cmd_logs(args: argparse.Namespace) -> int:
    registry = load_registry()
    compose_path = render_compose(registry)
    names = require_instance_names(registry, [args.name])
    compose_args = ["logs"]
    if args.follow:
        compose_args.append("-f")
    compose_args.append(service_name(names[0]))
    return run_compose(compose_path, compose_args)


def cmd_remove(args: argparse.Namespace) -> int:
    registry = load_registry()
    names = require_instance_names(registry, [args.name])
    instance_name = names[0]
    if not args.registry_only:
        compose_path = render_compose(registry)
        service = service_name(instance_name)
        stop_code = run_compose(compose_path, ["stop", service])
        rm_code = run_compose(compose_path, ["rm", "-sf", service])
        if stop_code != 0 or rm_code != 0:
            fail(
                "docker compose stop/rm failed; rerun with --registry-only if you only want "
                "to edit the registry"
            )

    registry["instances"] = [
        instance
        for instance in registry["instances"]
        if slugify(instance["name"]) != slugify(instance_name)
    ]
    save_registry(registry)
    render_compose(registry)

    if args.purge_data:
        shutil.rmtree(runtime_abs(instance_name), ignore_errors=True)

    print(f"Removed instance: {instance_name}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    registry = load_registry()
    name = slugify(args.name)
    existing = {slugify(instance["name"]) for instance in registry.get("instances", [])}
    if name in existing:
        fail(f"instance already exists: {args.name}")

    port = args.port or next_host_port(registry)
    configs = [normalize_repo_relative(path) for path in (args.config or [])]
    instance: dict[str, Any] = {
        "name": name,
        "strategy": args.strategy,
        "host_port": port,
    }
    if configs:
        instance["configs"] = configs

    overrides: dict[str, Any] = {}
    if args.wallet is not None:
        overrides["dry_run_wallet"] = args.wallet
    if args.stake_amount is not None:
        overrides["stake_amount"] = args.stake_amount
    if args.max_open_trades is not None:
        overrides["max_open_trades"] = args.max_open_trades
    if overrides:
        instance["overrides"] = overrides

    registry.setdefault("instances", []).append(instance)
    save_registry(registry)
    compose_path = render_compose(registry)
    print(f"Added instance: {name} -> http://127.0.0.1:{port}")
    if args.start:
        return run_compose(compose_path, ["up", "-d", service_name(name)])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage independent freqtrade dry-run instances for parallel strategy validation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="Render instance configs and compose file.")
    render_parser.set_defaults(func=cmd_render)

    list_parser = subparsers.add_parser("list", help="List configured instances.")
    list_parser.set_defaults(func=cmd_list)

    status_parser = subparsers.add_parser("status", help="Show docker compose status.")
    status_parser.set_defaults(func=cmd_status)

    up_parser = subparsers.add_parser("up", help="Start all or selected instances.")
    up_parser.add_argument("names", nargs="*", help="Instance names to start. Default: all.")
    up_parser.set_defaults(func=cmd_up)

    stop_parser = subparsers.add_parser("stop", help="Stop all or selected instances.")
    stop_parser.add_argument("names", nargs="*", help="Instance names to stop. Default: all.")
    stop_parser.set_defaults(func=cmd_stop)

    logs_parser = subparsers.add_parser("logs", help="Show logs for one instance.")
    logs_parser.add_argument("name", help="Instance name.")
    logs_parser.add_argument("-f", "--follow", action="store_true", help="Follow logs.")
    logs_parser.set_defaults(func=cmd_logs)

    add_parser = subparsers.add_parser("add", help="Add a new dry-run instance.")
    add_parser.add_argument("--name", required=True, help="Instance name, used for service/db/log paths.")
    add_parser.add_argument("--strategy", required=True, help="Freqtrade strategy class name.")
    add_parser.add_argument(
        "--config",
        action="append",
        help="Repo-relative config path under user_data/. Repeat to stack configs.",
    )
    add_parser.add_argument("--port", type=int, help="Host port for the instance UI/API.")
    add_parser.add_argument("--wallet", type=float, help="Override dry_run_wallet.")
    add_parser.add_argument("--stake-amount", type=float, help="Override stake_amount.")
    add_parser.add_argument("--max-open-trades", type=int, help="Override max_open_trades.")
    add_parser.add_argument("--start", action="store_true", help="Start the instance after adding it.")
    add_parser.set_defaults(func=cmd_add)

    remove_parser = subparsers.add_parser("remove", help="Remove one instance from the registry.")
    remove_parser.add_argument("name", help="Instance name.")
    remove_parser.add_argument(
        "--registry-only",
        action="store_true",
        help="Only update the registry and generated files. Do not stop or remove containers.",
    )
    remove_parser.add_argument(
        "--purge-data",
        action="store_true",
        help="Also delete user_data/dryrun_instances/<name>/ after stopping.",
    )
    remove_parser.set_defaults(func=cmd_remove)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
