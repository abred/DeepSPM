# pyserver — Python replacement for the LabVIEW instrument server

Pure Python/asyncio implementation of the DeepSPM TCP server.
Drop-in replacement for `labview/Nanonis Server` — the existing agent
code (`pyutil/envClient.py`) connects without modification.

## Why?

- **No LabVIEW dependency** — removes the proprietary, Windows-only,
  ~$3500/yr requirement
- **Cross-platform** — runs on Linux, macOS, Windows
- **Simulator included** — train and test the agent without a real STM
- **Extensible** — add new backends by implementing the `STMBackend` ABC

## Quick start

```bash
pip install numpy
python -m pyserver
```

The server starts on `0.0.0.0:50008` with a simulated STM backend.
Then run the agent as usual — it connects to `localhost:50008`.

## Usage

```
python -m pyserver [OPTIONS]

Options:
  --host HOST     Bind address (default: 0.0.0.0)
  --port PORT     TCP port (default: 50008)
  --seed SEED     Random seed for reproducible simulations
  --noise LEVEL   Scan image noise level (default: 0.05)
  --ini PATH      Path to deepSPM_server.ini config file
```

## Protocol

All 6 commands from the original LabVIEW server are supported:

| Command | Example | Response |
|---------|---------|----------|
| `scan` | `scan(10n,20n,50n,64)` | Binary: `[4B height][4B width][H×W float32 BE]` |
| `tipshaping` | `tipshaping(5n,10n,0,-4,200m)` | Text: `"1"` or `"0.5"` (stall) |
| `tipclean` | `tipclean(15n,25n)` | Text: `"1"` |
| `getparam` | `getparam(Range)` | Text: `"Range:8e-07"` |
| `approach` | `approach(f)` | Text: `"Approached. Z-range: ..."` |
| `movearea` | `movearea(y+)` | Text: `"Approach area changed with 0 crashes"` |

## Custom backends

To use real STM hardware, implement the `STMBackend` interface:

```python
from pyserver.simulator import STMBackend, ScanResult

class MySTMBackend(STMBackend):
    def scan_image(self, size_nm, offset_nm, pixel, bias_mv):
        # Talk to your hardware here
        ...
        return ScanResult(img_forward=img, ...)

    def tip_form(self, z_angstrom, x_nm, y_nm):
        ...

    def ramp_bias(self, target_mv):
        ...

    @property
    def name(self):
        return "MySTM"
```

For ready-made Createc and Nanonis adapters, see
[amrl-transport](https://github.com/formeo/amrl-transport).

## Testing

```bash
pip install pytest pytest-asyncio
pytest pyserver/tests/ -v
```

## Files

```
pyserver/
├── __init__.py
├── __main__.py      # CLI entry point
├── protocol.py      # Wire protocol parser & encoders
├── server.py        # Async TCP server
├── simulator.py     # Simulated STM backend
├── requirements.txt
└── tests/
    └── test_pyserver.py
```

## License

Same as DeepSPM — BSD-3-Clause.
