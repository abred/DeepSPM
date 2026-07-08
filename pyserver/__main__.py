"""
Entry point: python -m pyserver

Starts the asyncio TCP server with a simulator backend.
The existing DeepSPM agent (pyutil/envClient.py) connects directly.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .server import InstrumentServer, ServerConfig
from .simulator import SimulatorBackend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepSPM instrument server (Python/asyncio)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=50008,
        help="TCP port (default: 50008)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for reproducible simulations",
    )
    parser.add_argument(
        "--noise", type=float, default=0.05,
        help="scan image noise level (default: 0.05)",
    )
    parser.add_argument(
        "--ini", type=str, default=None,
        help="path to deepSPM_server.ini config file",
    )
    opts = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s|%(levelname)s|%(message)s",
        stream=sys.stdout,
    )

    if opts.ini:
        config = ServerConfig.from_ini(opts.ini)
        # CLI args override INI values
        config.host = opts.host
        config.port = opts.port
    else:
        config = ServerConfig(host=opts.host, port=opts.port)

    backend = SimulatorBackend(
        seed=opts.seed, noise_level=opts.noise,
    )
    server = InstrumentServer(backend, config)
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
