"""Explicit, pulse-audited physical trials; importing this module is offline."""
from dataclasses import replace

from iqm_surface_code.hardware_optimization import PulseExecutor

PROFILES = ("baseline", "center", "adaptive", "fast")


def strategy(profile, targets):
    from iqm.iqm_client import DDStrategy
    choices = {
        "baseline": [(5, "YXYX", "asap")],
        "center": [(5, "YXYX", "center")],
        "adaptive": [(9, "XYXYYXYX", "center"), (5, "YXYX", "center"), (2, "XX", "center")],
        "fast": [(5, "YXYX", "asap")],
    }
    if profile not in choices:
        raise ValueError("Unknown physical trial.")
    return DDStrategy(gate_sequences=choices[profile], target_qubits=frozenset(targets),
                      skip_leading_wait=True, skip_trailing_wait=True, merge_contiguous_waits=True)


class TrialExecutor(PulseExecutor):
    def compile(self, circuit, *, shots, dd=None, targets, ancilla_names=(), profile="baseline"):
        from iqm.pulse import Circuit
        from iqm.cpc.compiler.post_process import _process_readout_metrics
        native = self.backend.serialize_circuit(circuit)
        if profile == "fast":
            native = Circuit(name=native.name, instructions=tuple(
                replace(i, implementation="fast_constant") if i.name == "measure" and i.locus[0] in ancilla_names
                else replace(i, name="measure_fidelity", implementation="constant") if i.name == "measure"
                else i for i in native.instructions))
        settings = self.compiler.get_settings(circuits=[native])
        settings.set_shots(shots)
        settings.stages.dynamical_decoupling.apply_dd_strategy.dd_is_disabled = False
        if profile == "fast":
            settings.stages.circuit_stage.subscribe_and_probe.convert_terminal_measurements = False
            settings.stages.circuit_stage.subscribe_and_probe.probe_all = False
        dd_plan = strategy(profile, targets)
        run, context = self.compiler.compile(circuits=[native], settings=settings, context={"DDStrategy": dd_plan})
        added = {ch: stats["instructions"].get("IQPulse", 0)-self._before.get(ch, {}).get("instructions", {}).get("IQPulse", 0)
                 for ch, stats in self._after.items()}
        mapping = _process_readout_metrics(context["readout_metrics"], context.get("mapped_readout_keys", {}))
        audit = dict(before=self._before, after=self._after, dd_added_iq_pulses=added,
                     total_dd_added_iq_pulses=sum(added.values()), dd_strategy=dd_plan.model_dump(mode="json"),
                     schedule_duration_ns=max(v["duration_ns"] for v in self._after.values()),
                     measurement_operations=self._measurement_operations,
                     measurement_implementation="fast_constant ancillas, fidelity data" if profile == "fast" else "IQM defaults",
                     calibration_set_id=str(self.backend.architecture.calibration_set_id),
                     readout_mapping=mapping, mapped_readout_keys=context.get("mapped_readout_keys", {}),
                     heralding=False, all_requested_shots_retained=True)
        return run, context, audit


def prepare(output, case, backend, executor, cal, quality, mapping):
    # Reuse the validated source/native/frame conversion without altering its
    # physical optimizer. Only the explicitly named readout/DD trial changes.
    from .benchmark_headline import prepare_case
    trial = case["profile"]
    if trial == "fast":
        # The pulse is calibrated, but this snapshot omits its SSRO fidelity.
        # Explicit standard-readout proxies affect only the decoder prior;
        # the original saved calibration remains unchanged.
        from iqm_surface_code.hardware_optimization import quality_values
        values = quality_values(quality)
        proxies, additions = {}, []
        for name in mapping.values():
            target = f"metrics.ssro.measure.fast_constant.{name}.fidelity"
            source = f"metrics.ssro.measure.constant.{name}.fidelity"
            if target not in values:
                additions.append(dict(dut_field=target, value=values[source], invalid=False))
                proxies[target] = source
        quality = {**quality, "observations": quality["observations"]+additions}
        case["readout_prior_proxy"] = proxies

    class SelectedExecutor:
        def compile(self, *args, **kwargs):
            return executor.compile(*args, **kwargs, profile=trial)

    case["profile"] = "ethan"
    try:
        return prepare_case(output, case, backend, SelectedExecutor(), cal, quality, mapping)
    finally:
        case["profile"] = trial
