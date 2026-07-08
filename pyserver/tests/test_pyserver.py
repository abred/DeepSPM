"""Tests for the pyserver protocol and server."""
from __future__ import annotations

import asyncio
import struct

import numpy as np
import pytest
import pytest_asyncio

from pyserver.protocol import (
    ApproachCommand,
    GetParamCommand,
    MoveAreaCommand,
    ScanCommand,
    TipCleanCommand,
    TipShapingCommand,
    encode_scan_response,
    encode_text_response,
    parse_command,
)
from pyserver.server import InstrumentServer, ServerConfig
from pyserver.simulator import SimulatorBackend


# ── Protocol tests ────────────────────────────────────────────


class TestParseCommand:
    def test_scan(self):
        cmd = parse_command(b"scan(10n,20n,50n,64)")
        assert isinstance(cmd, ScanCommand)
        assert cmd.x_nm == pytest.approx(10.0)
        assert cmd.y_nm == pytest.approx(20.0)
        assert cmd.size_nm == pytest.approx(50.0)
        assert cmd.pixels == 64

    def test_tipshaping_stall(self):
        cmd = parse_command(b"tipshaping(5n,10n,stall)")
        assert isinstance(cmd, TipShapingCommand)
        assert cmd.action_str == "stall"

    def test_tipshaping_with_params(self):
        cmd = parse_command(b"tipshaping(5n,10n,0,-4,200m)")
        assert isinstance(cmd, TipShapingCommand)
        assert cmd.x_nm == pytest.approx(5.0)
        assert cmd.y_nm == pytest.approx(10.0)
        assert cmd.dip_m == pytest.approx(0.0)
        assert cmd.bias_v == pytest.approx(-4.0)
        assert cmd.timing_s == pytest.approx(0.2)

    def test_tipclean(self):
        cmd = parse_command(b"tipclean(15n,25n)")
        assert isinstance(cmd, TipCleanCommand)
        assert cmd.x_nm == pytest.approx(15.0)
        assert cmd.y_nm == pytest.approx(25.0)

    def test_getparam_range(self):
        cmd = parse_command(b"getparam(Range)")
        assert isinstance(cmd, GetParamCommand)
        assert cmd.param_name == "Range"

    def test_getparam_zrange(self):
        cmd = parse_command(b"getparam(zRange)")
        assert isinstance(cmd, GetParamCommand)
        assert cmd.param_name == "zRange"

    def test_approach(self):
        cmd = parse_command(b"approach(f)")
        assert isinstance(cmd, ApproachCommand)
        assert cmd.mode == "f"

    def test_movearea(self):
        cmd = parse_command(b"movearea(y+)")
        assert isinstance(cmd, MoveAreaCommand)
        assert cmd.direction == "y+"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown command"):
            parse_command(b"invalidcommand()")


class TestEncode:
    def test_scan_response_format(self):
        img = np.ones((4, 4), dtype=np.float32) * 0.5
        resp = encode_scan_response(img)
        h, w = struct.unpack(">ii", resp[:8])
        assert h == 4
        assert w == 4
        assert len(resp) == 8 + 4 * 4 * 4  # header + pixels

    def test_text_response(self):
        resp = encode_text_response("1")
        assert resp == b"1"


# ── Simulator tests ───────────────────────────────────────────


class TestSimulator:
    def test_scan_returns_image(self):
        sim = SimulatorBackend(seed=42)
        result = sim.scan_image(
            size_nm=10.0,
            offset_nm=np.array([0.0, 0.0]),
            pixel=32,
            bias_mv=-1000.0,
        )
        assert result.img_forward.shape == (32, 32)
        assert result.img_backward.shape == (32, 32)

    def test_scan_reproducible(self):
        s1 = SimulatorBackend(seed=1)
        s2 = SimulatorBackend(seed=1)
        r1 = s1.scan_image(10.0, np.array([0, 0]), 16, -1000)
        r2 = s2.scan_image(10.0, np.array([0, 0]), 16, -1000)
        np.testing.assert_array_equal(
            r1.img_forward, r2.img_forward,
        )

    def test_tip_form_no_crash(self):
        sim = SimulatorBackend()
        sim.tip_form(-50.0, 0.0, 0.0)  # should not raise

    def test_ramp_bias(self):
        sim = SimulatorBackend()
        sim.ramp_bias(500.0)  # should not raise

    def test_name(self):
        sim = SimulatorBackend()
        assert sim.name == "Simulator"


# ── Integration: server + client over TCP ─────────────────────


class TestServerIntegration:
    """End-to-end test: start server, connect, send commands."""

    @pytest_asyncio.fixture
    async def server_port(self):
        backend = SimulatorBackend(seed=42)
        config = ServerConfig(host="127.0.0.1", port=0)
        srv = InstrumentServer(backend, config)
        await srv.start()
        port = srv._server.sockets[0].getsockname()[1]
        yield port
        await srv.stop()

    async def test_getparam(self, server_port):
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server_port,
        )
        writer.write(b"getparam(Range)")
        await writer.drain()
        resp = await reader.read(1024)
        assert b"Range:" in resp
        writer.close()
        await writer.wait_closed()

    async def test_scan(self, server_port):
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server_port,
        )
        writer.write(b"scan(0n,0n,10n,16)")
        await writer.drain()

        # Read header (2x int32 big-endian)
        header = await reader.readexactly(8)
        h, w = struct.unpack(">ii", header)
        assert h == 16
        assert w == 16

        # Read image data
        data = await reader.readexactly(h * w * 4)
        img = np.frombuffer(data, dtype=np.float32)
        img = img.byteswap()
        assert img.shape == (256,)  # 16*16 flattened

        writer.close()
        await writer.wait_closed()

    async def test_approach(self, server_port):
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server_port,
        )
        writer.write(b"approach(f)")
        await writer.drain()
        resp = await reader.read(1024)
        assert b"Approached" in resp
        writer.close()
        await writer.wait_closed()

    async def test_movearea(self, server_port):
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server_port,
        )
        writer.write(b"movearea(y+)")
        await writer.drain()
        resp = await reader.read(1024)
        assert b"crashes" in resp
        writer.close()
        await writer.wait_closed()

    async def test_tipshaping_stall(self, server_port):
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server_port,
        )
        writer.write(b"tipshaping(5n,10n,stall)")
        await writer.drain()
        resp = await reader.read(1024)
        assert resp == b"0.5"
        writer.close()
        await writer.wait_closed()

    async def test_tipshaping_dip(self, server_port):
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server_port,
        )
        writer.write(b"tipshaping(5n,10n,-10n,0,100m)")
        await writer.drain()
        resp = await reader.read(1024)
        assert resp == b"1"
        writer.close()
        await writer.wait_closed()

    async def test_tipclean(self, server_port):
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server_port,
        )
        writer.write(b"tipclean(5n,10n)")
        await writer.drain()
        resp = await reader.read(1024)
        assert resp == b"1"
        writer.close()
        await writer.wait_closed()
