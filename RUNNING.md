# Running

## Setup
Requires [uv](https://docs.astral.sh/uv/). Python is pinned to 3.12 via `.python-version`
(uv will download it if missing).

```sh
uv sync          # creates .venv and installs numpy + pygame from uv.lock
```

## Run
```sh
uv run main.py   # opens the pygame window
```

Controls: `W`/`S` throttle, `A`/`D` steering. Close the window to quit.

## Adding dependencies
```sh
uv add <package>   # updates pyproject.toml and uv.lock
```

## Gotchas
- System `python3` on macOS is 3.9 — don't use it directly; always go through `uv run`
  (or `source .venv/bin/activate`).
- The project is not an installable package (no build backend); uv treats it as a
  plain application, so `uv sync` only installs dependencies.
