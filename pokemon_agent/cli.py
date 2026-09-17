"""
Pokemon Agent — CLI entry point.

Usage:
    pokemon-agent serve --rom path/to/rom.gba [--port 8765] [--data-dir ~/.pokemon-agent]
    pokemon-agent info  --rom path/to/rom.gba
    pokemon-agent --version
"""

import argparse
import hashlib
import sys
from pathlib import Path

__version__ = "0.1.0"

BANNER = r"""
  ____       _                              _                    _
 |  _ \ ___ | | _____ _ __ ___   ___  _ __ / \   __ _  ___ _ __ | |_
 | |_) / _ \| |/ / _ \ '_ ` _ \ / _ \| '_ / _ \ / _` |/ _ \ '_ \| __|
 |  __/ (_) |   <  __/ | | | | | (_) | | / ___ \ (_| |  __/ | | | |_
 |_|   \___/|_|\_\___|_| |_| |_|\___/|_|/_/   \_\__, |___|_| |_|\__|
                                                  |___/  v{version}
"""


def _detect_game_type(rom_path: str) -> str:
    ext = Path(rom_path).suffix.lower()
    if ext in (".gb", ".gbc"):
        return "red"
    elif ext == ".gba":
        return "firered"
    return "unknown"


def cmd_serve(args):
    """Start the FastAPI game server."""
    rom = Path(args.rom).expanduser().resolve()
    if not rom.exists():
        print(f"ERROR: ROM file not found: {rom}", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "saves").mkdir(exist_ok=True)

    game_type = _detect_game_type(str(rom))

    if args.load_state:
        sp = data_dir / "saves" / f"{args.load_state}.state"
        if not sp.exists():
            print(f"ERROR: save state not found: {sp}", file=sys.stderr)
            sys.exit(1)

    print(BANNER.format(version=__version__))
    print(f"  ROM:       {rom}")
    print(f"  Game type: {game_type}")
    print(f"  Port:      {args.port}")
    print(f"  Data dir:  {data_dir}")
    print()

    # Configure the server before uvicorn imports the app
    from pokemon_agent.server import GameConfig, configure, app  # noqa: F811

    configure(GameConfig(
        rom_path=str(rom),
        game_type=game_type,
        port=args.port,
        data_dir=str(data_dir),
        load_state=getattr(args, 'load_state', None),
        no_dashboard=args.no_dashboard,
    ))

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_info(args):
    """Display ROM information."""
    rom = Path(args.rom).expanduser().resolve()
    if not rom.exists():
        print(f"ERROR: ROM file not found: {rom}", file=sys.stderr)
        sys.exit(1)

    game_type = _detect_game_type(str(rom))
    size = rom.stat().st_size

    # Compute SHA-256
    sha = hashlib.sha256()
    with open(rom, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)

    print(f"ROM path:     {rom}")
    print(f"File size:    {size:,} bytes ({size / 1024 / 1024:.2f} MB)")
    print(f"SHA-256:      {sha.hexdigest()}")
    print(f"Extension:    {rom.suffix}")
    print(f"Detected as:  {game_type}")


def cmd_play(args):
    """Run the standalone autopilot loop (LLM plays the game)."""
    from pokemon_agent.autopilot import run_autopilot
    server = f"http://{args.host}:{args.port}"
    run_autopilot(server=server, model=args.model,
                  turn_delay=args.turn_delay, debug=args.debug)


def cmd_play_api(args):
    """Run the autopilot using Hermes AIAgent API instead of subprocess."""
    from pokemon_agent.hermes_api_driver import APIDriver
    server = f"http://{args.host}:{args.port}"
    driver = APIDriver(
        server=server,
        model=args.model,
        provider=args.provider,
        turn_delay=args.turn_delay,
    )
    driver.run()


def main():
    parser = argparse.ArgumentParser(
        prog="pokemon-agent",
        description="Pokemon Agent — AI-powered Pokemon game controller",
    )
    parser.add_argument(
        "--version", action="version", version=f"pokemon-agent {__version__}"
    )

    sub = parser.add_subparsers(dest="command")

    # --- serve ---
    serve_p = sub.add_parser("serve", help="Start the game server")
    serve_p.add_argument("--rom", required=True, help="Path to Pokemon ROM file")
    serve_p.add_argument("--host", default="127.0.0.1",
                         help="Bind address (default: 127.0.0.1; use 0.0.0.0 to expose on LAN)")
    serve_p.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    serve_p.add_argument(
        "--data-dir", default="~/.pokemon-agent",
        help="Data directory for saves, etc. (default: ~/.pokemon-agent)",
    )
    serve_p.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable dashboard mounting",
    )
    serve_p.add_argument(
        "--load-state", default=None,
        help="Name of a saved state to auto-load on startup (e.g. 'intro_complete')",
    )

    # --- info ---
    info_p = sub.add_parser("info", help="Show ROM information")
    info_p.add_argument("--rom", required=True, help="Path to Pokemon ROM file")

    # --- play (autopilot) ---
    play_p = sub.add_parser("play", help="Run the LLM autopilot against a running server (subprocess CLI)")
    play_p.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    play_p.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    play_p.add_argument("--model", default=None,
                        help="LLM model (default: $POKEMON_LLM_MODEL or anthropic/claude-sonnet-4.5)")
    play_p.add_argument("--turn-delay", type=float, default=1.5,
                        help="Seconds between turns (default: 1.5)")
    play_p.add_argument("--debug", action="store_true", help="Enable debug logging")

    # --- play-api (Hermes AIAgent direct) ---
    play_api_p = sub.add_parser("play-api", help="Run the LLM autopilot using Hermes AIAgent API (direct, faster)")
    play_api_p.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    play_api_p.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    play_api_p.add_argument("--model", default=None,
                        help="LLM model (default: nvidia/nemotron-3-ultra-550b-a55b)")
    play_api_p.add_argument("--provider", default=None,
                        help="LLM provider (default: nvidia)")
    play_api_p.add_argument("--turn-delay", type=float, default=1.5,
                        help="Seconds between turns (default: 1.5)")
    play_api_p.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "play":
        cmd_play(args)
    elif args.command == "play-api":
        cmd_play_api(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()