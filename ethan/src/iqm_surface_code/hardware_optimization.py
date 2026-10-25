"""Native-frame compilation, pulse-audited IQM execution, measured noise priors.

No credential or network access occurs on import. PulLA is initialized only by
an explicit PulseExecutor construction, and submit is a separate explicit call.
"""
from __future__ import annotations

from collections import Counter
import math
import uuid
import numpy as np
import stim

VALIDATED_MEMORY_PROFILES={
    "X":dict(profile="fused_xy4",decoder=dict(family="mwpm",parameters=dict(phase_round=0.,backaction_scale=.5))),
    "Z":dict(profile="fenced_xy4",decoder=dict(family="posterior_10",parameters=dict(phase_round=0.,backaction_scale=0.))),
}


def validated_memory_profile(basis):
    from copy import deepcopy
    if basis.upper() not in VALIDATED_MEMORY_PROFILES:raise ValueError("Memory basis must be X or Z.")
    return deepcopy(VALIDATED_MEMORY_PROFILES[basis.upper()])


def optimize_native(circuit,*,mode="fenced"):
    from iqm.qiskit_iqm.iqm_transpilation import IQMOptimizeSingleQubitGates
    from qiskit.transpiler import PassManager
    working=circuit
    if mode in {"fenced","fused"}:
        working=circuit.copy_empty_like()
        measuring=False
        for inst in circuit.data:
            if inst.operation.name=="barrier":continue
            is_measure=inst.operation.name=="measure"
            if is_measure!=measuring:working.barrier()
            working.append(inst)
            measuring=is_measure
        working._layout=circuit.layout
    else:
        raise ValueError("Final memory optimization mode must be fenced or fused.")
    if mode=="fused":
        # After the last entangler, all remaining basis rotations commute with
        # measurements on other wires. Move them ahead of the final joint
        # readout. A terminal local uncompute after a wire's last measurement
        # cannot change any returned probability, so discard it. Never move
        # such a rotation across measurement of its own wire.
        last=max(i for i,inst in enumerate(working.data) if inst.operation.name=="cz")
        tail=list(working.data[last+1:]);measured=set()
        rotations,measurements=[],[]
        for inst in tail:
            name=inst.operation.name
            if name=="barrier":continue
            if name=="measure":
                if measured.intersection(inst.qubits):raise ValueError("Terminal fusion encountered repeated measurement.")
                measured.update(inst.qubits);measurements.append(inst)
            else:
                if name not in {"r","id"}:
                    raise ValueError("Terminal fusion cannot cross a same-wire measurement.")
                if measured.intersection(inst.qubits):continue
                rotations.append(inst)
        fused=working.copy_empty_like()
        for inst in list(working.data[:last+1])+rotations:fused.append(inst)
        fused.barrier()
        for inst in measurements:fused.append(inst)
        fused._layout=circuit.layout;working=fused
    optimized=PassManager(IQMOptimizeSingleQubitGates(ignore_barriers=False)).run(working)
    optimized._layout=circuit.layout
    return optimized,dict(before=dict(circuit.count_ops()),after=dict(optimized.count_ops()),
                          depth_before=circuit.depth(),depth_after=optimized.depth(),mode=mode)


def physical_indices(circuit):
    initial=circuit.layout.initial_layout if circuit.layout is not None else None
    use_layout=initial is not None and initial.get_registers()==set(circuit.qregs)
    return {q:initial[q] if use_layout else circuit.find_bit(q).index for q in circuit.qubits}


def r_clifford(angle,phase):
    angle_steps=round(float(angle)/(math.pi/2))
    phase_steps=round(float(phase)/(math.pi/2))
    if abs(float(angle)-angle_steps*math.pi/2)<=1e-7 and angle_steps%4==0:
        return "I"
    diagonal=round(float(phase)/(math.pi/4))
    if angle_steps%4==2 and abs(float(angle)-angle_steps*math.pi/2)<=1e-7 and abs(float(phase)-diagonal*math.pi/4)<=1e-7 and diagonal%2:
        return "H_XY" if diagonal%4==1 else "H_NXY"
    if abs(float(angle)-angle_steps*math.pi/2)>1e-7 or abs(float(phase)-phase_steps*math.pi/2)>1e-7:
        raise ValueError("Native decoder currently requires Clifford R angles/axes.")
    axis="X" if phase_steps%2==0 else "Y"
    sign=1 if phase_steps%4 in (0,1) else -1
    step=(angle_steps*sign)%4
    return {0:"I",1:"SQRT_"+axis,2:axis,3:"SQRT_"+axis+"_DAG"}[step]


def native_stim(circuit,reference,mapping,backend):
    """Actual native Clifford instructions, with exact classical-record remap.

    Annotate at the end using absolute source detector coordinates; parity
    references are rewritten from Qiskit clbits to actual native M order.
    """
    physical=physical_indices(circuit)
    out=stim.Circuit()
    for q,coord in reference.get_final_qubit_coordinates().items():
        out.append("QUBIT_COORDS",[backend.qubit_name_to_index(mapping[q])],coord)
    out.append("R",sorted(backend.qubit_name_to_index(p) for p in mapping.values()))
    columns=[]
    for inst in circuit.data:
        name=inst.operation.name
        qs=[physical[q] for q in inst.qubits]
        if name=="r": out.append(r_clifford(*inst.operation.params),qs)
        elif name=="cz": out.append("CZ",qs)
        elif name=="measure":
            out.append("M",qs)
            columns.append(circuit.find_bit(inst.clbits[0]).index)
        elif name=="barrier": out.append("TICK")
        elif name=="id": out.append("I",qs)
        else: raise ValueError(f"Unsupported native instruction: {name}")
    if sorted(columns)!=list(range(reference.num_measurements)):
        raise ValueError("Native measurement columns do not bijectively match source records.")
    inverse={column:index for index,column in enumerate(columns)}
    measured,detector=0,0
    coords=reference.get_detector_coordinates()
    for inst in reference.flattened():
        if inst.name in {"M","MX","MR","MRX"}: measured+=len(inst.targets_copy())
        elif inst.name in {"DETECTOR","OBSERVABLE_INCLUDE"}:
            targets=[stim.target_rec(inverse[measured+t.value]-len(columns)) for t in inst.targets_copy()]
            args=coords[detector] if inst.name=="DETECTOR" else inst.gate_args_copy()
            out.append(inst.name,targets,args)
            detector+=inst.name=="DETECTOR"
    out.detector_error_model()  # Fail on any non-deterministic parity.
    return out,columns


def quality_values(quality):
    return {o["dut_field"]:float(o["value"]) for o in quality["observations"]
            if not o.get("invalid") and isinstance(o["value"],(float,int))}


def native_noise(ideal,backend,calibration,quality,*,phase_round=0.,backaction_scale=0.,
                 measurement_operations=None):
    """Native-gate priors plus explicit residual measurement-noise proxies.

    QND/leakage numbers are repeated-readout metrics, not per-CZ leakage.
    The backaction channel is a phenomenological Pauli proxy; no hidden flags
    or evaluation labels enter this constructor. Residual phase/backaction
    parameters must be frozen using independent pilot data before evaluation.
    """
    if phase_round != 0 or backaction_scale not in (0., .5):
        raise ValueError("Invalid residual noise parameters.")
    from .internal_helpers import get_qubit_lists
    data,ancillas=get_qubit_lists(ideal)
    values=quality_values(quality)
    # Compiler observations override the terminal-readout heuristic. Per-wire
    # queues tolerate joint readout grouping and never include discarded probes.
    from collections import defaultdict,deque
    readouts=defaultdict(deque)
    if measurement_operations is not None:
        for operation in measurement_operations:
            if operation.get("key")=="__ONLY_PROBE":continue
            for name in operation["locus"]:
                readouts[name].append((operation["gate"],operation.get("implementation") or "constant"))
    flat=list(ideal.flattened())
    last_measure={t.value:index for index,i in enumerate(flat) if i.name=="M" for t in i.targets_copy()}
    out=stim.Circuit()
    def bounded(p): return min(.45,max(0.,float(p)))
    for index,inst in enumerate(flat):
        qs=[t.value for t in inst.targets_copy() if t.is_qubit_target]
        if inst.name=="M":
            for q in qs:
                name=backend.index_to_qubit_name(q)
                terminal=index==last_measure[q]
                use_fidelity=terminal
                if measurement_operations is not None:
                    if not readouts[name]:raise ValueError("Compiled readout trace does not cover native measurements.")
                    gate,implementation=readouts[name].popleft()
                    ro=1-values[f"metrics.ssro.{gate}.{implementation}.{name}.fidelity"]
                else:
                    ro=(1-values[f"metrics.ssro.measure_fidelity.constant.{name}.fidelity"]
                        if use_fidelity else calibration.qubits[name].readout_error)
                out.append("M",[q],bounded(ro))
                if q in ancillas:
                    if phase_round:
                        out.append("Z_ERROR",data,bounded(phase_round/len(ancillas)))
                    if not terminal and backaction_scale:
                        loss=1-values[f"metrics.rrc.measure.constant.{name}.qndness"]
                        leak=values[f"metrics.rrc.measure.constant.{name}.leakage"]
                        # Proxy combines state disturbance and leakage-driven
                        # randomization; it is not a calibrated qutrit model.
                        p=backaction_scale*(.5*loss+.25*leak)
                        out.append("DEPOLARIZE1",[q],bounded(p))
        else:
            out.append(inst)
            if inst.name=="R": out.append("X_ERROR",qs,.002)
            elif inst.name=="CZ":
                for a,b in zip(qs[::2],qs[1::2]):
                    out.append("DEPOLARIZE2",[a,b],bounded(1.25*calibration.edge_error(
                        backend.index_to_qubit_name(a),backend.index_to_qubit_name(b))))
            elif inst.name in {"X","Y","H_XY","H_NXY","SQRT_X","SQRT_X_DAG","SQRT_Y","SQRT_Y_DAG"}:
                for q in qs:
                    p=bounded(1.5*calibration.qubits[backend.index_to_qubit_name(q)].single_qubit_infidelity)
                    out.append("DEPOLARIZE1",[q],1-(1-p)**(2 if inst.name in {"X","Y","H_XY","H_NXY"} else 1))
    if out.without_noise()!=ideal:
        raise ValueError("Native noise insertion changed noiseless instructions/parities.")
    if any(readouts.values()):raise ValueError("Compiled readout trace has unmatched returned measurements.")
    return out


def dd_strategy(sequence,targets):
    from iqm.iqm_client import DDStrategy
    choices={"xy4":[(5,"YXYX","asap")]}
    if sequence=="off": return None
    if sequence not in choices: raise ValueError("Unknown pulse DD strategy.")
    return DDStrategy(gate_sequences=choices[sequence],target_qubits=frozenset(targets),
                      skip_leading_wait=True,skip_trailing_wait=True,merge_contiguous_waits=True)


def pulse_snapshot(schedules,builder):
    result=[]
    for schedule in schedules:
        channels={}
        for name,segment in schedule.items():
            elapsed,events=0,[]
            timing=builder.channels[name]
            for instruction in segment:
                kind=type(instruction).__name__
                if kind in {"IQPulse","FluxPulse","ReadoutTrigger"}:
                    events.append(dict(kind=kind,start_ns=1e9*timing.duration_to_seconds(elapsed),
                        duration_ns=1e9*timing.duration_to_seconds(instruction.duration),
                        labels=[getattr(a,"label","") for a in getattr(instruction,"acquisitions",())]))
                elapsed+=instruction.duration
            channels[name]=dict(instructions=dict(Counter(type(i).__name__ for i in segment)),events=events,
                duration_ns=1e9*timing.duration_to_seconds(segment.duration))
        result.append(channels)
    return result


def audit_pulses_before(schedules,builder,context):
    context["audit_pulses_before"]=pulse_snapshot(schedules,builder)
    return schedules


def audit_pulses_after(schedules,builder,context):
    context["audit_pulses_after"]=pulse_snapshot(schedules,builder)
    return schedules


class PulseExecutor:
    """Compile the exact submitted playlist and persist enough data to resume.

    Result retrieval uses saved JSON readout mappings, not pickled credentials
    or compiler objects. Every returned shot is retained (no heralding filter).
    """
    def __init__(self,backend,token):
        if backend.name!="garnet":
            raise ValueError("Audited pulse execution is currently validated only for the Garnet backend.")
        from iqm.pulla.pulla import Pulla
        from iqm.cpc.core.observation.observation_loading_rules import LatestFromStash
        from iqm.cpc.compiler.compiler import CompilationStage
        self.backend=backend
        self.pulla=Pulla("https://resonance.iqm.tech",quantum_computer="garnet",token=token)
        stash=self.pulla.get_calibration_stash(str(backend.architecture.calibration_set_id))
        self.compiler=self.pulla.get_standard_compiler(loading_rules=[LatestFromStash(stash)],exa_style_pp=False)
        def readout_pass(circuits,context):
            self._measurement_operations=[dict(gate=i.name,implementation=i.implementation,
                locus=list(i.locus),key=i.args.get("key")) for c in circuits for i in c.instructions
                if i.name in {"measure","measure_fidelity"}]
            return circuits
        self.compiler.circuit_stages[0].add_passes(readout_pass)
        def before_pass(schedules,builder,context):
            self._before=pulse_snapshot(schedules,builder)[0]
            return schedules
        def after_pass(schedules,builder,context):
            self._after=pulse_snapshot(schedules,builder)[0]
            return schedules
        before=CompilationStage(name="pulse_audit_before");before.add_passes(before_pass)
        after=CompilationStage(name="pulse_audit_after");after.add_passes(after_pass)
        self.compiler.pulse_stages.insert(3,before)
        self.compiler.pulse_stages.insert(5,after)

    def compile(self,circuit,*,shots,dd="xy4",targets=()):
        from iqm.cpc.compiler.post_process import _process_readout_metrics
        native=self.backend.serialize_circuit(circuit)
        circuits=[native]
        settings=self.compiler.get_settings(circuits=circuits)
        settings.set_shots(shots)
        settings.stages.dynamical_decoupling.apply_dd_strategy.dd_is_disabled=dd=="off"
        strategy=dd_strategy(dd,targets)
        context={"DDStrategy":strategy} if strategy else {}
        run,context=self.compiler.compile(circuits=circuits,settings=settings,context=context)
        before,after=self._before,self._after
        added={channel:stats["instructions"].get("IQPulse",0)-before.get(channel,{}).get("instructions",{}).get("IQPulse",0)
               for channel,stats in after.items()}
        mapping=_process_readout_metrics(context["readout_metrics"],context.get("mapped_readout_keys",{}))
        audit=dict(before=before,after=after,dd_added_iq_pulses=added,total_dd_added_iq_pulses=sum(added.values()),
            dd_strategy=strategy.model_dump(mode="json") if strategy else None,
            schedule_duration_ns=max(v["duration_ns"] for v in after.values()),
            measurement_implementation="standard IQM defaults",
            measurement_operations=self._measurement_operations,
            calibration_set_id=str(self.backend.architecture.calibration_set_id),
            readout_mapping=mapping,mapped_readout_keys=context.get("mapped_readout_keys",{}),
            heralding=False,all_requested_shots_retained=True)
        return run,context,audit

    def submit(self,run,context):
        return self.pulla.submit_playlist(run,context=context)

    def status(self,job_id):
        return self.pulla.get_job(uuid.UUID(str(job_id)))

    def memory(self,job_id,audit,shots):
        from iqm.cpc.compiler.post_process import _unmap_label
        from iqm.pulla.utils import convert_sweep_spot_to_arrays
        from iqm.qiskit_iqm.iqm_job import IQMJob
        results=self.pulla._iqm_server_client.get_job_artifact_sweep_results(str(job_id))
        values={_unmap_label(label,audit["mapped_readout_keys"]):spots[0] for label,spots in results.items()}
        mapping=tuple({key:tuple(labels) for key,labels in item.items()} for item in audit["readout_mapping"])
        batch=list(convert_sweep_spot_to_arrays(values,mapping))
        if len(batch)!=1: raise ValueError("Expected one pulse circuit per job.")
        memory=IQMJob._iqm_format_measurement_results(batch[0],shots,True)
        return dict(Counter(memory)),memory


def run_integrated_qiskit(backend, *, token, distance, rounds, shots, basis,
                          calibration=None, qubit_map=None, artifact_dir=None,
                          train_dets=None, train_obs=None):
    """Explicit final-policy execution; selection and research profiles are removed."""
    from datetime import datetime,timezone
    from pathlib import Path
    import os,time
    from .benchmark_garnet import calibration_from_observations,select_native_layout,write_json
    from .run_on_hardware import prepare_qiskit
    from .surface_code import make_stim_circuit
    from .internal_helpers import get_qubit_lists
    from .benchmark_vanilla_lut import memory_syndromes
    from .extract_syndromes import extract_syndromes
    from challenge.decoders.hybrid import build_decoder
    from qiskit import qpy
    from iqm.station_control.client.serializers.run_serializers import serialize_run_definition
    token=token or os.environ.get("IQM_TOKEN")
    if not token:raise ValueError("Pulse profile needs an explicit token or IQM_TOKEN.")
    config = validated_memory_profile(basis)
    name = config["profile"]
    mode = "fused" if basis == "X" else "fenced"
    selection = config["decoder"]
    parameters = dict(selection["parameters"])
    quality=backend.client.get_quality_metric_set(backend.architecture.calibration_set_id).model_dump(mode="json")
    if calibration is None:
        pulse=backend.client.get_calibration_set(backend.architecture.calibration_set_id).model_dump(mode="json")
        calibration=calibration_from_observations(backend.architecture.model_dump(mode="json"),quality,pulse)
    if qubit_map is None and backend.num_qubits<=20:
        ref=make_stim_circuit(distance,rounds,basis,gate_basis="cz")
        edges=[(backend.index_to_qubit_name(a),backend.index_to_qubit_name(b)) for a,b in backend.coupling_map.get_edges()]
        qubit_map,_=select_native_layout(ref,edges,backend.physical_qubits,calibration)
    prepared=prepare_qiskit(backend,distance,rounds,basis,calibration=calibration,qubit_map=qubit_map)
    reference=prepared["stim_circuit"]
    compiled,optimization=optimize_native(prepared["circuit"],mode=mode)
    native,columns=native_stim(compiled,reference,prepared["placement"].stim_to_physical,backend)
    data,anc=get_qubit_lists(reference)
    mapping=prepared["placement"].stim_to_physical
    executor=PulseExecutor(backend,token)
    run,context,audit=executor.compile(compiled,shots=shots,dd="xy4",targets=[mapping[q] for q in data])
    model = native_noise(native, backend, calibration, quality,
                         measurement_operations=audit["measurement_operations"], **parameters)
    family = selection["family"]
    decoder, dem = build_decoder(model, family, train_dets=train_dets, train_obs=train_obs)
    output=Path(artifact_dir) if artifact_dir else Path("results/integrated_api")/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    native.to_file(output/"executed.stim");model.to_file(output/"decoder.stim");reference.to_file(output/"source_reference.stim")
    with (output/"native.qpy").open("wb") as stream:qpy.dump(compiled,stream)
    import hashlib
    write_json(output/"quality_metrics.json",quality);calibration.save(output/"calibration.json")
    from challenge.decoders.hybrid import save_decoder
    save_decoder(decoder, dem, output/"inference")
    report=dict(profile=name,optimization=optimization,pulse_audit=audit,parameters=parameters,
                distance=distance,rounds=rounds,basis=basis,shots=shots,placement=mapping,
                source_sha256=hashlib.sha256(str(reference).encode()).hexdigest(),
                native_sha256=hashlib.sha256(str(native).encode()).hexdigest(),
                decoder_sha256=hashlib.sha256(str(model).encode()).hexdigest(),
                decoder_family=family,status="prepared",measurement_columns=columns)
    write_json(output/"manifest.json",report)
    job=executor.submit(run,context);report.update(job_id=str(job.job_id),status="submitted")
    write_json(output/"manifest.json",report)
    (output/"playlist.pb").write_bytes(serialize_run_definition(run).SerializeToString())
    deadline=time.monotonic()+1800
    while True:
        state=executor.status(job.job_id);report["status"]=str(state.status)
        if report["status"] in {"completed","failed","cancelled"}:break
        if time.monotonic()>deadline:raise TimeoutError(f"Pulse job {job.job_id} pending; retrieval mapping saved in {output}.")
        time.sleep(2)
    write_json(output/"manifest.json",report)
    if report["status"]!="completed":raise RuntimeError(f"Pulse job {job.job_id} failed: {state.errors}")
    counts,memory=executor.memory(job.job_id,audit,shots)
    write_json(output/"counts.json",counts);write_json(output/"memory.json",memory)
    raw,_=memory_syndromes(counts,memory,reference,shots)
    src=extract_syndromes(raw,reference,print_summary=False)
    dst=extract_syndromes(raw[:,columns],native,print_summary=False)
    if not (np.array_equal(src["det_events"],dst["det_events"]) and np.array_equal(src["obs_flips"],dst["obs_flips"])):
        raise ValueError("Native/source detector frames differ on returned records.")
    np.savez_compressed(output/"shots.npz",raw_meas=raw,native_raw_meas=raw[:,columns],
                        det_events=dst["det_events"],obs_flips=dst["obs_flips"])
    report.update(dataset_sha256=hashlib.sha256((output/"shots.npz").read_bytes()).hexdigest(),
                  playlist_sha256=hashlib.sha256((output/"playlist.pb").read_bytes()).hexdigest())
    write_json(output/"manifest.json",report)
    source=native
    syndromes=dst
    syndromes.update(stim_circuit=source,decoder_circuit=model,decoder=decoder,job_id=str(job.job_id),
        hardware_report=report,rounds=rounds,basis=basis,artifacts=str(output))
    return syndromes
