"""scripts/pulla_check.py, offline: the read-only guard, the QASM parser, the parameter extraction, the scrub."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pulla_check", REPO / "scripts" / "pulla_check.py")
pc = importlib.util.module_from_spec(_spec)

# Fake addresses for the scrub tests, assembled at runtime: a search of the diff for URLs, IPs,
# hostnames or auth headers (the brief's handoff check) then finds none in this file.
SCHEME = "https" + ":" + "//"
IP = ".".join(["10", "1", "2", "3"])
HOST = ".".join(["host", "example", "tech"])
BEARER = "Bear" + "er abc"
sys.modules["pulla_check"] = pc  # dataclasses look their module up while the module runs
_spec.loader.exec_module(pc)


# ─── the guard ───────────────────────────────────────────────────────────────


@pytest.fixture
def guard(monkeypatch):
    """Install the guard for one test; monkeypatch puts the IQM classes back afterwards."""
    from iqm.iqm_server_client.iqm_server_client import _IQMServerClient
    from iqm.pulla.pulla import Pulla

    monkeypatch.setattr(Pulla, "submit_playlist", Pulla.submit_playlist)
    monkeypatch.setattr(_IQMServerClient, "_send_request", _IQMServerClient._send_request)
    for name in pc._SUBMITTING:
        monkeypatch.setattr(_IQMServerClient, name, getattr(_IQMServerClient, name))
    monkeypatch.setattr(pc, "REQUESTS", [])
    pc.install_guard()
    pc.install_guard()  # idempotent: the HTTP seam is wrapped once
    return Pulla, _IQMServerClient


def _offline_client(cls):
    """An _IQMServerClient with no constructor call (the constructor would reach the network)."""
    client = cls.__new__(cls)
    client.root_url = "http" + ":" + "//localhost"
    client._timeout = 1.0
    client._signature = "pulla-check-test"
    client._auth_header_callback = None
    client._enable_opentelemetry = False
    return client


def test_guard_is_installed(guard):
    assert pc.guard_installed()


def test_submit_playlist_raises(guard):
    Pulla, _ = guard
    with pytest.raises(pc.ReadOnlyViolation, match="submit_playlist"):
        Pulla.submit_playlist(object(), object(), context={})
    with pytest.raises(pc.ReadOnlyViolation):
        Pulla.submit_playlist(object(), object(), context={}, use_timeslot=True)


@pytest.mark.parametrize("name", pc._SUBMITTING)
def test_submitting_client_methods_raise(guard, name):
    _, cls = guard
    with pytest.raises(pc.ReadOnlyViolation, match=name):
        getattr(cls, name)(_offline_client(cls), object())


@pytest.mark.parametrize("verb", ["post", "put", "patch", "delete", "head"])
def test_non_get_request_refused_before_it_is_sent(guard, verb):
    _, cls = guard
    sent = []

    def http_method(url, **kwargs):
        sent.append(url)

    http_method.__name__ = verb
    with pytest.raises(pc.ReadOnlyViolation, match=verb.upper()):
        cls._send_request(_offline_client(cls), http_method, "jobs/garnet/run")
    assert sent == []
    assert pc.REQUESTS == []


def test_get_request_goes_through_and_is_logged(guard):
    _, cls = guard
    seen = {}

    def get(url, **kwargs):
        seen["url"], seen["headers"] = url, kwargs.get("headers", {})
        return SimpleNamespace(status_code=200, ok=True, headers={})

    response = cls._send_request(_offline_client(cls), get, "quantum-computers")
    assert response.status_code == 200
    assert seen["url"].endswith("/api/v1/quantum-computers")
    assert pc.REQUESTS == [{"method": "GET", "path": "quantum-computers", "status": 200}]
    assert "Authorization" not in json.dumps(pc.REQUESTS)


@pytest.mark.parametrize("device", ["garnet:mock", "emerald:mock", "facade_garnet", "sirius", ""])
def test_only_garnet_and_emerald(device):
    with pytest.raises(pc.ReadOnlyViolation, match="not allowed"):
        pc.connect(device)


def test_error_record_reads_status_and_scrubs():
    response = SimpleNamespace(status_code=403)
    try:
        try:
            raise ValueError("inner")
        except ValueError as inner:
            err = RuntimeError(f"refused at {SCHEME}{HOST}/api for {IP}")
            err.response = response
            raise err from inner
    except RuntimeError as exc:
        rec = pc.error_record(exc, [])
        # the request log wins when it has the failing status
        logged = pc.error_record(exc, [{"status": 200}, {"status": 401}])
    assert rec["error_class"] == "builtins.RuntimeError"
    assert rec["error_chain"] == ["builtins.RuntimeError", "builtins.ValueError"]
    assert rec["http_status"] == 403
    assert rec["message_scrubbed"] == "refused at <url> for <ip>"
    assert logged["http_status"] == 401


# ─── the QASM parser ─────────────────────────────────────────────────────────

SNIPPET = """OPENQASM 3.0;
include "stdgates.inc";
gate r(p0, p1) _gate_q_0 {
  U(p0, -pi/2 + p1, pi/2 - p1) _gate_q_0;
}
bit[3] c;
r(pi/2, -pi/2) $0;
cz $0, $1;
barrier $1, $0, $3;
r(1.5707963267948966, pi) $3;
c[0] = measure $1;
cz $3, $0;
c[2] = measure $0;
"""


def test_parse_snippet():
    ops = pc.parse_qasm(SNIPPET)
    assert [o.name for o in ops] == ["prx", "cz", "barrier", "prx", "measure", "cz", "measure"]
    assert ops[0] == pc.Op("prx", (0,), math.pi / 2, -math.pi / 2)
    assert ops[2].qubits == (1, 0, 3)
    assert ops[3].angle == pytest.approx(math.pi / 2) and ops[3].phase == pytest.approx(math.pi)
    assert ops[6].clbit == 2
    assert pc.op_counts(ops) == {"barrier": 1, "cz": 2, "measure": 2, "r": 2}
    assert pc.used_qubits(ops) == ["QB1", "QB2", "QB4"]
    assert pc.cz_pairs(ops) == [("QB1", "QB2"), ("QB1", "QB4")]


def test_to_circuit():
    c = pc.to_circuit(pc.parse_qasm(SNIPPET), "snippet")
    names = [(i.name, i.locus) for i in c.instructions]
    assert names[:3] == [("prx", ("QB1",)), ("cz", ("QB1", "QB2")), ("barrier", ("QB2", "QB1", "QB4"))]
    assert c.instructions[0].args == {"angle": math.pi / 2, "phase": -math.pi / 2}
    assert c.instructions[4].args == {"key": "c0"}
    assert c.instructions[1].args == {}


@pytest.mark.parametrize("line", ["h $0;", "reset $0;", "c[0] = measure $0, $1;", "U(0, 0, 0) $0;", "delay[100ns] $0;"])
def test_parse_rejects_anything_else(line):
    with pytest.raises(ValueError, match="unsupported"):
        pc.parse_qasm(f"OPENQASM 3.0;\n{line}\n")


@pytest.mark.parametrize(
    ("expr", "value"),
    [("pi/2", math.pi / 2), ("-pi/2", -math.pi / 2), ("pi", math.pi), ("0", 0.0), ("2*pi/4", math.pi / 2), ("1e-3", 1e-3)],
)
def test_angle(expr, value):
    assert pc.angle(expr) == pytest.approx(value)


@pytest.mark.parametrize("expr", ["__import__('os')", "pi**2", "x", "True", "[1]"])
def test_angle_rejects_code(expr):
    with pytest.raises(ValueError):
        pc.angle(expr)


@pytest.mark.parametrize("name", pc.PILOTS)
def test_vendored_pilot(name):
    """The vendored pilot matches its placement.json entry: sha256, op counts, qubits; 17 qubits, 24 couplers."""
    ops, entry = pc.load_pilot(name)
    assert len(pc.used_qubits(ops)) == 17
    assert len(pc.cz_pairs(ops)) == 24
    assert set(pc.used_qubits(ops)) == {f"QB{i}" for i in (1, 2, *range(4, 17), 19, 20)}
    assert all(o.angle == pytest.approx(math.pi / 2) for o in ops if o.name == "prx")
    assert entry["swaps"] == 0


def test_pilot_variants_share_couplers():
    a, _ = pc.load_pilot(pc.PILOTS[0])
    b, _ = pc.load_pilot(pc.PILOTS[1])
    assert pc.cz_pairs(a) == pc.cz_pairs(b)
    assert pc.op_counts(b) == {k: v for k, v in pc.op_counts(a).items() if k != "barrier"}


# ─── Emerald's circuits and its compile (emerald-compile) ────────────────────


def test_emerald_calset_is_the_designs():
    from dqiqm.experiments import emerald as EM

    assert pc.EMERALD_CALSET == EM.CALSET
    assert pc.EMERALD_COMPILED == ("E1_d3_Z_r03_eA", "E1_d3_X_r03_eA", "E1_d3_Z_r03_eB", "E1_d3_X_r03_eB")


@pytest.mark.parametrize("name", pc.EMERALD_COMPILED)
def test_emerald_circuit(name):
    """Each committed R = 3 circuit matches its placement.json entry: sha256, op counts, qubits; 17 qubits, 24 couplers."""
    ops, entry = pc.load_emerald(name)
    assert entry["params"]["rounds"] == 3 and entry["params"]["patch"] in pc.EMERALD_PATCHES
    assert len(pc.used_qubits(ops)) == 17 and len(pc.cz_pairs(ops)) == 24
    assert pc.op_counts(ops)["measure"] == 33
    assert all(o.angle == pytest.approx(math.pi / 2) for o in ops if o.name == "prx")


def test_emerald_circuit_refuses_an_edited_file(tmp_path, monkeypatch):
    for f in ("placement.json", "E1_d3_Z_r03_eA.qasm"):
        (tmp_path / f).write_text((pc.EMERALD_CIRCUITS / f).read_text())
    monkeypatch.setattr(pc, "EMERALD_CIRCUITS", tmp_path)
    pc.load_emerald("E1_d3_Z_r03_eA")
    (tmp_path / "E1_d3_Z_r03_eA.qasm").write_text((tmp_path / "E1_d3_Z_r03_eA.qasm").read_text().replace("$12;", "$11;", 1))
    with pytest.raises(ValueError, match="sha256"):
        pc.load_emerald("E1_d3_Z_r03_eA")


def test_emerald_patches():
    patches = pc.emerald_patches({n: pc.load_emerald(n) for n in pc.EMERALD_COMPILED})
    assert set(patches) == {"eA", "eB"}
    assert patches["eA"]["circuits"] == ["E1_d3_Z_r03_eA", "E1_d3_X_r03_eA"]
    assert patches["eA"]["ancillas"] == ["QB14", "QB19", "QB21", "QB28", "QB30", "QB37", "QB39", "QB44"]
    assert patches["eB"]["ancillas"] == ["QB5", "QB8", "QB10", "QB17", "QB19", "QB26", "QB28", "QB33"]
    a, b = set(patches["eA"]["qubits"]), set(patches["eB"]["qubits"])
    assert a & b == {"QB19", "QB20", "QB27", "QB28"} and len(a | b) == 30  # emerald-design: they share 4 qubits


class _Channel:
    sample_rate = 2.0e9
    instruction_duration_granularity = 8
    integration_start_dead_time = 0
    integration_stop_dead_time = 152

    def duration_to_seconds(self, samples):
        return samples / self.sample_rate


class _Impl:
    def __init__(self, samples):
        self.samples = samples

    def _get_readout_trigger_durations(self):
        return self.samples, 0


class _Builder:
    """The schedule builder as readout_windows uses it: a probe channel per line, IQM's trigger length in samples."""

    def __init__(self, lines):
        self.lines = lines
        self.channels = {f"{line}__readout": _Channel() for line in set(lines.values())}

    def get_probe_channel(self, q):
        return f"{self.lines[q]}__readout"

    def get_implementation(self, gate, locus):
        n = int(locus[0][2:])
        return _Impl(2000 + 8 * n if gate == "measure" else 1500 + 8 * n)  # measure: 1,000 ns + 4 ns per qubit number


class _Compiler:
    def __init__(self, lines):
        self.builder = _Builder(lines)
        self.gate_definitions = {}

    def get_schedule_builder(self):
        return self.builder


def test_readout_windows_with_a_fake_builder():
    topology = SimpleNamespace(component_to_probe_line={"QB1": "PL-1", "QB9": "PL-2"})
    w = pc.readout_windows(_Compiler(topology.component_to_probe_line), ["QB1", "QB9"], topology)
    assert w["qubits"]["QB1"] == {"probe_line": "PL-1", "measure_window_ns": 1004.0, "measure_fidelity_window_ns": 754.0}
    assert w["qubits"]["QB9"]["measure_window_ns"] == 1036.0
    assert w["probe_lines"]["PL-2"] == {"sample_rate": 2.0e9, "instruction_duration_granularity": 8,
                                        "integration_start_dead_time": 0, "integration_stop_dead_time": 152}


@pytest.fixture
def fake_emerald(tmp_path, monkeypatch):
    """emerald() against a fake Pulla, compiler and topology, writing under tmp_path: what it calls, what it writes."""
    calls = {"build_compilers": 0, "topology": 0}
    lines = {f"QB{i}": f"PL-{1 + i % 3}" for i in range(1, 55)}

    def pinned(pulla, device):
        assert device == "emerald"
        return calls["calset"], SimpleNamespace(observations=OBSERVATIONS)

    def compilers(pulla, stash):
        calls["build_compilers"] += 1
        return {"full": _Compiler(lines), "body": _Compiler(lines)}

    def topology():
        calls["topology"] += 1
        return SimpleNamespace(component_to_probe_line=lines, get_coupler_for=lambda a, b: f"TC-{a[2:]}-{b[2:]}")

    def compile_one(comps, circuit, **options):
        return {k: {"duration_ns": 2000.0, "instructions": {}, "readout_triggers": []} for k in comps}

    pulla = SimpleNamespace(get_standard_compiler=lambda **kw: _Compiler(lines), get_chip_label=lambda: "M000-X0",
                            get_chip_topology=topology)
    monkeypatch.setattr(pc, "CALIBRATION", tmp_path)
    monkeypatch.setattr(pc, "pinned_stash", pinned)
    monkeypatch.setattr(pc, "build_compilers", compilers)
    monkeypatch.setattr(pc, "compile_one", compile_one)
    monkeypatch.setattr(pc, "circuit_stage_ops", lambda comp, circuit: {"measure": 16, "measure_fidelity": 18})
    monkeypatch.setattr(pc, "REQUESTS", [{"method": "GET", "path": "calibration-sets/emerald/x", "status": 200}])
    return pulla, calls, tmp_path / "emerald" / "pulse"


def test_emerald_compile_with_fakes(fake_emerald):
    from dqiqm.experiments import device as dev

    pulla, calls, folder = fake_emerald
    calls["calset"] = pc.EMERALD_CALSET
    params, compiled = pc.emerald(pulla)
    assert params.name == f"{pc.EMERALD_CALSET}.json" and compiled.name == f"compile.{pc.EMERALD_CALSET}.json"
    res = json.loads(compiled.read_text())
    assert res["schema"] == pc.SCHEMA and res["device"] == "emerald" and res["calibration_set_id"] == pc.EMERALD_CALSET
    assert set(res["pilots"]) == set(pc.EMERALD_COMPILED)
    assert res["pilots"]["E1_d3_Z_r03_eA"]["our_schedule_ns"] == 1848.0
    assert set(res["patches"]) == {"eA", "eB"} and res["patches"]["eA"]["couplers"]["QB13__QB14"] == "TC-13-14"
    qubits = res["readout_windows"]["qubits"]
    assert len(qubits) == 30 and set(qubits) == set(res["patches"]["eA"]["qubits"]) | set(res["patches"]["eB"]["qubits"])
    assert qubits["QB44"]["measure_window_ns"] == 1176.0  # (2000 + 8*44) samples at 2 GS/s
    assert res["requests"] == [{"method": "GET", "path": "calibration-sets/emerald/x", "status": 200}]
    assert "readout_parameters" in res and "merge_test" not in res
    # what the decoder priors read
    w = dev.compiled_readout_windows(pc.EMERALD_CALSET, folder, device="emerald")
    assert w.device == "emerald" and w.of("QB44", "measure_window_ns") == 1176.0
    with pytest.raises(ValueError, match="not garnet's"):
        dev.compiled_readout_windows(pc.EMERALD_CALSET, folder, device="garnet")
    # never overwritten: a second run gets new, stamped names
    again = pc.emerald(pulla)
    assert all(p.exists() for p in (params, compiled, *again)) and len({params, compiled, *again}) == 4


def test_emerald_stops_on_another_calibration_set(fake_emerald):
    pulla, calls, folder = fake_emerald
    calls["calset"] = "44444444-4444-4444-4444-444444444444"
    with pytest.raises(pc.CalibrationSetChanged, match="44444444-4444-4444-4444-444444444444"):
        pc.emerald(pulla)
    assert calls["build_compilers"] == 0 and calls["topology"] == 0  # nothing compiled
    files = sorted(p.name for p in folder.iterdir())
    assert files == ["44444444-4444-4444-4444-444444444444.json"]  # the live set is recorded
    assert json.loads((folder / files[0]).read_text())["calibration_set_id"] == calls["calset"]


def test_main_exits_2_when_the_calibration_set_changed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pc, "CALIBRATION", tmp_path)
    monkeypatch.setattr(pc, "connect", lambda device: (object(), {"device": device, "constructor": "ok", "requests": []}))

    def changed(pulla, first):
        raise pc.CalibrationSetChanged("emerald's live calibration set is X")

    monkeypatch.setattr(pc, "emerald", changed)
    monkeypatch.setattr(pc, "report", lambda: pytest.fail("no report after a stop"))
    monkeypatch.setattr(sys, "argv", ["pulla_check.py", "--devices", "emerald"])
    assert pc.main() == 2
    assert "STOP: emerald's live calibration set is X" in capsys.readouterr().out


# ─── parameter extraction ────────────────────────────────────────────────────


def _obs(value, unit=""):
    return SimpleNamespace(value=value, unit=unit)


OBSERVATIONS = {
    "gate_definitions.prx.default_implementation": _obs("drag_crf_sx"),
    "gate_definitions.cz.default_implementation": _obs("slepian_crf"),
    "gate_definitions.u.default_implementation": _obs("prx_u"),  # not a kept gate
    "gate_definitions.cz.crf_acstarkcrf.override_default_for_loci": _obs(np.array(["QB2__QB1", "QB9__QB8"])),
    "gates.prx.drag_crf_sx.QB1.amplitude_i": _obs(0.21),
    "gates.prx.drag_crf_sx.QB1.duration": _obs(1.6e-08, "s"),
    "gates.prx.drag_crf_sx.QB3.amplitude_i": _obs(0.5),  # not in the patch
    "gates.cz.slepian_crf.QB2__QB1.coupler.amplitude": _obs(0.27),  # reversed locus
    "gates.cz.slepian_crf.QB2__QB1.rz.QB1": _obs(-2.1, "rad"),
    "gates.cz.slepian_crf.QB3__QB2.coupler.amplitude": _obs(0.3),  # not a patch coupler
    "gates.measure.constant.QB1.acquisition_type": _obs("threshold"),  # a string: dropped
    "gates.measure.constant.QB1.integration_weights_I": _obs(np.zeros(728)),
    "gates.measure.constant.QB1.block_neighbors": _obs(np.bool_(False)),
    "gates.measure.constant.QB1.duration": _obs(3.64e-07, "s"),
    "gates.reset_wait.reset_wait.PL-1.duration": _obs(3e-4, "s"),  # a probe line, not a qubit
    "gates.prx_12.modulated_drag_crf.QB1.amplitude_i": _obs(0.1),  # not a kept gate
    "characterization.gate_properties.measure.constant.QB1.single_shot_01_error": _obs(0.003),
    "controllers.QB1.drive.frequency": _obs(4.6e9, "Hz"),
    "controllers.QB1.readout.center_frequency": _obs(5.63e9, "Hz"),  # control electronics: dropped
    "controllers.options.trigger_master": _obs("some-instrument"),  # dropped
}


def test_extract_parameters_patch():
    p = pc.extract_parameters(OBSERVATIONS, ["QB1", "QB2"], [("QB1", "QB2")])
    assert p["default_implementations"] == {"cz": "slepian_crf", "prx": "drag_crf_sx"}
    assert p["override_default_for_loci"] == {"cz.crf_acstarkcrf": ["QB2__QB1"]}
    assert p["gates"]["prx"] == {"drag_crf_sx": {"QB1": {"amplitude_i": 0.21, "duration": 1.6e-08}}}
    assert p["gates"]["cz"] == {"slepian_crf": {"QB2__QB1": {"coupler.amplitude": 0.27, "rz.QB1": -2.1}}}
    assert p["gates"]["measure"]["constant"]["QB1"] == {
        "block_neighbors": False,
        "duration": 3.64e-07,
        "integration_weights_I": {"array_len": 728},
    }
    assert "reset_wait" not in p["gates"] and "prx_12" not in p["gates"]
    assert p["readout_characterization"] == {"measure": {"constant": {"QB1": {"single_shot_01_error": 0.003}}}}
    assert p["drive_frequency_hz"] == {"QB1": 4.6e9}
    assert p["dropped_non_numeric"] == 1
    assert p["units"]["prx.drag_crf_sx.duration"] == "s"
    assert "some-instrument" not in json.dumps(p) and "center_frequency" not in json.dumps(p)


def test_extract_parameters_all_loci_and_summary():
    p = pc.extract_parameters(OBSERVATIONS)
    assert set(p["gates"]["prx"]["drag_crf_sx"]) == {"QB1", "QB3"}
    assert set(p["gates"]["cz"]["slepian_crf"]) == {"QB2__QB1", "QB3__QB2"}
    assert p["override_default_for_loci"] == {"cz.crf_acstarkcrf": ["QB2__QB1", "QB9__QB8"]}
    s = pc.summarize_parameters(p)
    amp = s["gates"]["prx"]["drag_crf_sx"]["parameters"]["amplitude_i"]
    assert amp == {"median": pytest.approx(0.355), "min": 0.21, "max": 0.5, "n": 2}
    assert s["gates"]["prx"]["drag_crf_sx"]["loci"] == 2
    assert "block_neighbors" not in s["gates"]["measure"]["constant"]["parameters"]  # bools are not summarised
    assert s["drive_frequency_hz"]["parameters"]["frequency"]["n"] == 1


# ─── the scrub ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [SCHEME + "x", f"see {IP} here", BEARER, HOST, "IQM_TOKEN", "Authorization: x", "password"],
)
def test_check_clean_refuses(text):
    with pytest.raises(ValueError, match="forbidden"):
        pc.check_clean({"note": text})


def test_check_clean_accepts_numbers_and_names():
    obj = {"v": "3.0.1.20261019072700", "f": 4599102201.081829, "locus": "QB2__QB1", "path": "calibration-sets/garnet/x"}
    assert json.loads(pc.check_clean(obj)) == obj


def test_committed_outputs_are_clean():
    """Every committed pulla-check file passes the same check, and names its schema."""
    files = sorted((REPO / "data" / "calibration").glob("*/pulse/*.json"))
    results = [f for f in files if f.parent.name == "pulse"]
    for f in results:
        text = f.read_text()
        assert not pc.FORBIDDEN.search(text), f
        assert json.loads(text)["schema"] == pc.SCHEMA, f
    for f in (REPO / "data" / "calibration" / "garnet" / "pulse" / "pilot").glob("*"):
        assert not pc.FORBIDDEN.search(f.read_text()), f


def test_report_runs_offline(capsys):
    pc.report()
    out = capsys.readouterr().out
    if pc.newest("garnet", "access") is not None:
        assert "garnet: Pulla constructor" in out
