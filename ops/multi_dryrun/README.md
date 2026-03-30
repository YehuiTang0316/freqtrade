# Multi-Instance Dry-Run

This setup runs one independent `freqtrade trade` container per strategy so you can validate several strategies in parallel without sharing runtime state.

## What stays isolated

- One container per strategy instance
- One SQLite database per instance under `user_data/dryrun_instances/<name>/`
- One log file per instance under `user_data/dryrun_instances/<name>/logs/`
- One localhost port per instance

Your existing root `docker-compose.yml` is not used by this flow.

## Registry

Edit [registry.json](./registry.json) to define instances. Each instance needs:

- `name`
- `strategy`
- `host_port`
- `configs`

`configs` must point to files under `user_data/` because only `user_data` is mounted into the container.

## Commands

List instances:

```bash
python3 scripts/multi_dryrun.py list
```

Render the generated compose file:

```bash
python3 scripts/multi_dryrun.py render
```

Start all configured instances:

```bash
python3 scripts/multi_dryrun.py up
```

Start only one instance:

```bash
python3 scripts/multi_dryrun.py up trendpulse
```

Add a new instance without touching the running ones:

```bash
python3 scripts/multi_dryrun.py add \
  --name donchian \
  --strategy DonchianPulse \
  --config user_data/config_nfix.json \
  --port 18084 \
  --start
```

Stop one instance:

```bash
python3 scripts/multi_dryrun.py stop trendpulse
```

Remove one instance from the registry and stop only that service:

```bash
python3 scripts/multi_dryrun.py remove trendpulse
```

If you only want to edit the registry without touching containers:

```bash
python3 scripts/multi_dryrun.py remove trendpulse --registry-only
```

Show container status:

```bash
python3 scripts/multi_dryrun.py status
```

Show one instance log stream:

```bash
python3 scripts/multi_dryrun.py logs -f mlpulse
```

## Port Mapping

- `breakpulse` -> `http://127.0.0.1:18081`
- `trendpulse` -> `http://127.0.0.1:18082`
- `mlpulse` -> `http://127.0.0.1:18083`

All containers still listen on `8080` internally. Only the host ports differ.
