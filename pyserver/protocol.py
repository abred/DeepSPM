"""
DeepSPM wire protocol: parser and encoder.

The original DeepSPM uses a simple text-based protocol over TCP.
This module provides clean parsing and encoding for both server and client.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

import numpy as np


class CommandType(Enum):
    SCAN = auto()
    TIPSHAPING = auto()
    TIPCLEAN = auto()
    GETPARAM = auto()
    APPROACH = auto()
    MOVEAREA = auto()


# ── Parsed command dataclasses ────────────────────────────────

@dataclass
class ScanCommand:
    cmd: CommandType = CommandType.SCAN
    x_nm: float = 0.0
    y_nm: float = 0.0
    size_nm: float = 10.0
    pixels: int = 64


@dataclass
class TipShapingCommand:
    cmd: CommandType = CommandType.TIPSHAPING
    x_nm: float = 0.0
    y_nm: float = 0.0
    action_str: str = ""
    dip_m: float = 0.0
    bias_v: float = 0.0
    timing_s: float = 0.0


@dataclass
class TipCleanCommand:
    cmd: CommandType = CommandType.TIPCLEAN
    x_nm: float = 0.0
    y_nm: float = 0.0


@dataclass
class GetParamCommand:
    cmd: CommandType = CommandType.GETPARAM
    param_name: str = "Range"


@dataclass
class ApproachCommand:
    cmd: CommandType = CommandType.APPROACH
    mode: str = "f"


@dataclass
class MoveAreaCommand:
    cmd: CommandType = CommandType.MOVEAREA
    direction: str = "y+"


Command = Union[
    ScanCommand, TipShapingCommand, TipCleanCommand,
    GetParamCommand, ApproachCommand, MoveAreaCommand,
]


# ── SI suffix parser ─────────────────────────────────────────

_SUFFIXES = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6}


def _parse_value(s: str) -> float:
    """Parse a numeric string with optional SI suffix. Returns SI value."""
    s = s.strip()
    if not s:
        return 0.0
    if s[-1].isalpha() and s[-1] in _SUFFIXES:
        return float(s[:-1]) * _SUFFIXES[s[-1]]
    return float(s)


def _extract_args(text: str) -> list[str]:
    """Extract args from 'command(a,b,c)' -> ['a','b','c']."""
    inner = text.split("(", 1)[1].rstrip(")")
    return [a.strip() for a in inner.split(",")]


# ── Command parser ────────────────────────────────────────────

def parse_command(raw: bytes) -> Command:
    """
    Parse a raw TCP command from the DeepSPM client.

    Examples:
        parse_command(b"scan(10n,20n,50n,64)")
        parse_command(b"tipshaping(5n,10n,0,-4,200m)")
        parse_command(b"getparam(Range)")
    """
    text = raw.decode("utf-8", errors="replace").strip().replace(" ", "")

    if text.startswith("scan("):
        args = _extract_args(text)
        return ScanCommand(
            x_nm=_parse_value(args[0]) * 1e9,
            y_nm=_parse_value(args[1]) * 1e9,
            size_nm=_parse_value(args[2]) * 1e9,
            pixels=int(float(args[3])),
        )

    elif text.startswith("tipshaping("):
        args = _extract_args(text)
        x_m = _parse_value(args[0])
        y_m = _parse_value(args[1])
        action_parts = args[2:]
        action_str = ",".join(action_parts)

        dip_m = 0.0
        bias_v = 0.0
        timing_s = 0.0
        if action_str != "stall" and len(action_parts) >= 3:
            dip_m = _parse_value(action_parts[0])
            bias_v = _parse_value(action_parts[1])
            timing_s = _parse_value(action_parts[2])

        return TipShapingCommand(
            x_nm=x_m * 1e9,
            y_nm=y_m * 1e9,
            action_str=action_str,
            dip_m=dip_m,
            bias_v=bias_v,
            timing_s=timing_s,
        )

    elif text.startswith("tipclean("):
        args = _extract_args(text)
        return TipCleanCommand(
            x_nm=_parse_value(args[0]) * 1e9,
            y_nm=_parse_value(args[1]) * 1e9,
        )

    elif text.startswith("getparam("):
        args = _extract_args(text)
        return GetParamCommand(param_name=args[0] if args else "Range")

    elif text.startswith("approach("):
        args = _extract_args(text)
        return ApproachCommand(mode=args[0] if args else "f")

    elif text.startswith("movearea("):
        args = _extract_args(text)
        return MoveAreaCommand(direction=args[0] if args else "y+")

    else:
        raise ValueError(f"Unknown command: {text[:60]}")


# ── Response encoders ─────────────────────────────────────────

def encode_scan_response(image: np.ndarray) -> bytes:
    """
    Encode a scan image into DeepSPM binary format.

    Format: [4B height BE int32][4B width BE int32][H*W float32 BE]
    Client calls byteswap() after receiving.
    """
    img = image.astype(np.float32)
    h, w = img.shape
    header = struct.pack(">ii", h, w)
    img_be = img.byteswap()
    return header + img_be.tobytes()


def encode_text_response(text: str) -> bytes:
    """Encode a UTF-8 text response."""
    return text.encode("utf-8")


def encode_param_response(name: str, value: float) -> bytes:
    """Encode a getparam response as 'Name:value'."""
    return f"{name}:{value}".encode()


def encode_approach_response(z_range: float) -> bytes:
    """Encode an approach response."""
    return f"Approached. Z-range: {z_range}".encode()


def encode_movearea_response(crashes: int = 0) -> bytes:
    """Encode a movearea response."""
    return f"Approach area changed with {crashes} crashes".encode()
