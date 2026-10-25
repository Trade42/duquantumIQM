"""Select and independently confirm Garnet stacks using the challenge LER.

No credential, network, or hardware action occurs on import. Use --submit for
explicit bounded acquisition, --resume to retrieve existing jobs only, and
--analyze for an offline replay. The primary fit always uses R=1,2,3 and A=1.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import stim
from scipy.special import xlogy
from scipy.stats import chi2

from .decoding import fit_ler, wilson_interval
from .decoders.hybrid import build_decoder, load_decoder, save_decoder

ROUNDS = (1, 2, 3)
PHYSICAL = ("ethan", "dara", "iyanu", "combined")
FAMILIES = ("mwpm", "mwpm_fit", "ensemble", "ensemble_fit", "ml", "ml_fit",
            "empirical_1", "empirical_3", "empirical_10", "empirical_30")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str)+"\n", encoding="utf8")
    temporary.replace(path)


def challenge_fit(rounds, errors, shots):
    """Unchanged challenge estimator plus an explicit lack-of-fit diagnostic."""
    result = fit_ler(rounds, errors, shots, fit_spam=False)
    k, n, p = np.asarray(errors), np.asarray(shots), np.array(result["fitted_probabilities"])
    result["deviance"] = max(0., float(2*np.sum(xlogy(k, k/n)+xlogy(n-k, 1-k/n)
                                                -xlogy(k, p)-xlogy(n-k, 1-p))))
    result["nominal_df"] = len(k)-1
    result["goodness_of_fit_p"] = float(chi2.sf(result["deviance"], len(k)-1))
    result["stationary_model_rejected_5pct"] = result["goodness_of_fit_p"] < .05
    result["rounds"] = list(rounds)
    result["errors"], result["shots"] = list(map(int, k)), list(map(int, n))
    result["probabilities"] = (k/n).tolist()
    result["wilson_ci95"] = [wilson_interval(int(a), int(b)) for a, b in zip(k, n)]
    return result


def fit_many(k, n, rounds=ROUNDS):
    """Vectorized binomial MLE for a nonparametric stratified bootstrap."""
    k, n, r = np.asarray(k), np.asarray(n), np.asarray(rounds)
    lo, hi = np.zeros(k.shape[0]), np.full(k.shape[0], .5)
    def objective(e):
        p = np.clip((1-(1-2*e[:, None])**r)/2, 1e-15, 1-1e-15)
        return -np.sum(xlogy(k, p)+xlogy(n-k, 1-p), axis=1)
    for _ in range(64):
        left, right = (2*lo+hi)/3, (lo+2*hi)/3
        choose = objective(left) < objective(right)
        hi, lo = np.where(choose, right, hi), np.where(choose, lo, left)
    result = (lo+hi)/2
    result[np.all(k == 0, axis=1)] = 0
    return result


def ler_difference(first, second, *, paired, draws=10000, alpha=.05, seed=2203):
    """first minus second; resample within round and acquisition block.

    Paired samples use the joint four-cell failure table. Independent physical
    arms use separate empirical binomials. Model fit is recomputed each draw.
    """
    rng = np.random.default_rng(seed)
    a = np.zeros((draws, len(ROUNDS)), int)
    b = np.zeros_like(a)
    na, nb = np.zeros(len(ROUNDS), int), np.zeros(len(ROUNDS), int)
    for j, r in enumerate(ROUNDS):
        for block in sorted(first[r]):
            fa, fb = np.asarray(first[r][block], bool), np.asarray(second[r][block], bool)
            na[j] += len(fa); nb[j] += len(fb)
            if paired:
                if len(fa) != len(fb):
                    raise ValueError("Paired failure vectors must have identical shots.")
                cells = np.bincount(2*fa.astype(int)+fb, minlength=4)
                sample = rng.multinomial(len(fa), cells/len(fa), size=draws)
                a[:, j] += sample[:, 2]+sample[:, 3]
                b[:, j] += sample[:, 1]+sample[:, 3]
            else:
                a[:, j] += rng.binomial(len(fa), fa.mean(), size=draws)
                b[:, j] += rng.binomial(len(fb), fb.mean(), size=draws)
    difference = fit_many(a, na)-fit_many(b, nb)
    observed = challenge_fit(ROUNDS, [sum(v.sum() for v in first[r].values()) for r in ROUNDS], na)["epsilon_per_round"] \
        -challenge_fit(ROUNDS, [sum(v.sum() for v in second[r].values()) for r in ROUNDS], nb)["epsilon_per_round"]
    return dict(absolute_ler_reduction=observed, ci=np.quantile(difference, [alpha/2, 1-alpha/2]).tolist(),
                confidence=1-alpha, paired=paired, draws=draws, seed=seed,
                method="stratified nonparametric bootstrap of fixed-shot failures; refit challenge equation",
                excludes="calibration drift between blocks and model-selection uncertainty")


def choose(pilot, basis, profiles=PHYSICAL):
    """One profile/family across every primary round; tie favors simpler policy."""
    rows = [v for v in pilot if v["basis"] == basis and v["profile"] in profiles]
    if not rows:
        raise ValueError("Missing complete pilot results.")
    return min(rows, key=lambda v: (v["fit"]["epsilon_per_round"], v["family"] != "mwpm", v["family"], v["profile"]))


def prepare_case(output, case, backend, executor, cal, quality, mapping):
    """Reconstruct standalone paths; preserve native and source detector frames."""
    from iqm_surface_code.run_on_hardware import prepare_qiskit
    from iqm_surface_code.hardware_optimization import native_stim, native_noise, optimize_native
    from iqm_surface_code.internal_helpers import get_qubit_lists
    from darawork.garnet_dataset import prepare_garnet, suite_and_layout
    from qiskit import qpy
    basis, rounds, profile = case["basis"], case["rounds"], case["profile"]
    original = prepare_qiskit(backend, 3, rounds, basis, calibration=cal, qubit_map=mapping, dd="off")
    reference, compiled = original["stim_circuit"], original["circuit"]
    if profile in {"dara", "combined"}:
        ref, compiled = prepare_garnet(backend, rounds, basis, mapping, sqrt_y=True, barriers="measure")
        if ref != reference:
            raise ValueError("Standalone source circuits disagree.")
    before = dict(compiled.count_ops())
    if profile in {"ethan", "combined"}:
        compiled, optimization = optimize_native(compiled, mode="fused" if basis == "X" else "fenced")
    else:
        optimization = dict(before=before, after=dict(compiled.count_ops()))
    compiled.name = case["id"]
    native, columns = native_stim(compiled, reference, mapping, backend)
    data, anc = get_qubit_lists(reference)
    case.update(transport="pulse" if profile in {"ethan", "combined"} else "server",
                source_sha256=hashlib.sha256(str(reference).encode()).hexdigest(),
                native_sha256=hashlib.sha256(str(native).encode()).hexdigest(),
                native_measurement_columns=columns, optimization=optimization)
    run = context = None
    if case["transport"] == "pulse":
        run, context, audit = executor.compile(compiled, shots=case["shots"], dd="xy4",
            targets=[mapping[q] for q in data], ancilla_names=[mapping[q] for q in anc])
        case["pulse_audit"] = audit
        model = native_noise(native, backend, cal, quality, phase_round=0.,
            backaction_scale=.5 if basis == "X" else 0., measurement_operations=audit["measurement_operations"])
        case["model_frame"] = "native"
    elif profile == "dara":
        from challenge.noise_models import add_noise
        suite = suite_and_layout(.02, {"1q":20, "2q":80, "measure":700, "reset":1000}, .001)[0]
        model = add_noise(reference, suite["garnet_coherence"], qubit_map=mapping)
        case["model_frame"] = "source; Garnet medians and assumed durations/readout"
    else:
        model = original["decoder_circuit"]
        case["model_frame"] = "source; calibrated Pauli"
    for suffix, circuit in (("source", reference), ("native", native), ("decoder", model)):
        circuit.to_file(output / f"{case['id']}.{suffix}.stim")
    with (output / f"{case['id']}.qpy").open("wb") as stream:
        qpy.dump(compiled, stream)
    case["model_sha256"] = digest(output / f"{case['id']}.decoder.stim")
    return compiled, run, context


def collect(output, manifest, backend, executor, secret):
    """Collect ledgered IDs only. Return after 45 seconds if jobs remain pending."""
    from iqm_surface_code.benchmark_vanilla_lut import memory_syndromes
    from iqm_surface_code.extract_syndromes import extract_syndromes
    end = time.monotonic()+45
    while True:
        pending = [c for c in manifest["cases"] if c.get("job_id") and not c.get("data_file")
                   and c.get("status") not in {"failed", "cancelled"}]
        if not pending:
            return
        for case in pending:
            job_id = case["job_id"]
            if case["transport"] == "pulse":
                job = executor.status(job_id)
                status = str(job.status)
                case["status"] = status
                if status in {"failed", "cancelled"}:
                    case["error"] = str(job.errors).replace(secret, "[REDACTED]")
                    continue
                if status != "completed":
                    continue
                counts, memory = executor.memory(job_id, case["pulse_audit"], case["shots"])
                write_json(output / f"{case['id']}.job.json", job.model_dump(mode="json"))
            else:
                job = backend.retrieve_job(job_id)
                status = str(job.status())
                case["status"] = status
                if status.endswith(("ERROR", "CANCELLED")):
                    case["status"] = "failed" if status.endswith("ERROR") else "cancelled"
                    case["error"] = "Server execution failed; inspect saved job ID."
                    continue
                if not status.endswith("DONE"):
                    continue
                result = job.result()
                counts, memory = result.get_counts(0), result.get_memory(0)
            reference = stim.Circuit.from_file(output / f"{case['id']}.source.stim")
            native = stim.Circuit.from_file(output / f"{case['id']}.native.stim")
            raw, source_syndromes = memory_syndromes(counts, memory, reference, case["shots"])
            native_raw = raw[:, case["native_measurement_columns"]]
            native_syndromes = extract_syndromes(native_raw, native, print_summary=False)
            for key in ("det_events", "obs_flips"):
                if not np.array_equal(source_syndromes[key], native_syndromes[key]):
                    raise ValueError("Native/source detector frames disagree on physical shots.")
            filename = f"{case['id']}.shots.npz"
            np.savez_compressed(output / filename, raw_meas=raw, native_raw_meas=native_raw,
                                det_events=source_syndromes["det_events"], obs_flips=source_syndromes["obs_flips"])
            write_json(output / f"{case['id']}.counts.json", counts)
            case.update(data_file=filename, dataset_sha256=digest(output / filename),
                        status="completed", returned_shots=len(raw), collected_utc=utc())
            print("DONE", case["id"], len(raw), flush=True)
        write_json(output / "manifest.json", manifest)
        if time.monotonic() > end:
            return
        if all(c.get("data_file") or c.get("status") in {"failed", "cancelled"} for c in manifest["cases"]):
            return
        time.sleep(2)


def arrays(output, case):
    path = output / case["data_file"]
    if digest(path) != case["dataset_sha256"]:
        raise ValueError("Immutable hardware dataset hash changed.")
    with np.load(path, allow_pickle=False) as data:
        return data["det_events"].copy(), data["obs_flips"].copy()


def pilot_analysis(output, manifest):
    """All training and physical/decoder selection precede confirmation jobs."""
    if (output / "frozen_selection.json").exists():
        return json.loads((output / "frozen_selection.json").read_text())
    cases = [c for c in manifest["cases"] if c["phase"] == "pilot"]
    if len(cases) != len(PHYSICAL)*2*len(ROUNDS) or any(not c.get("data_file") for c in cases):
        raise ValueError("All fixed pilot cases must complete before selection.")
    scores = []
    for case in cases:
        dets, obs = arrays(output, case)
        train = manifest["train_shots"]
        model = stim.Circuit.from_file(output / f"{case['id']}.decoder.stim")
        case.setdefault("validation", {})
        families=[f for f in manifest["families"] if (f not in {"mlp","conv"} or
                  case["profile"] in manifest.get("neural_profiles",{}).get(case["basis"],[]))
                  and (f!='coherence' or case['profile']=='iyanu')]
        for family in families:
            if family in case["validation"]:
                continue
            path = output / f"{case['id']}.{family}.frozen"
            started = time.perf_counter()
            options={}
            if family=='coherence':
                from dataclasses import replace
                from challenge.calibration import CalibrationSnapshot
                from challenge.noise_models import NoiseModel,add_noise
                cal=CalibrationSnapshot.load(output/'calibration.json')
                mapping={int(q):p for q,p in manifest['placement'].items()}
                reference=stim.Circuit.from_file(output/f"{case['id']}.source.stim")
                options['coherence_models']={ns:add_noise(reference,NoiseModel(name='calibrated_coherence',
                    calibration=replace(cal,durations_ns={**cal.durations_ns,'measure':ns,'reset':ns}),
                    coherence=True,idle=0),qubit_map=mapping) for ns in (300,800,1500,3000)}
            decoder, dem = build_decoder(model, family, train_dets=dets[:train], train_obs=obs[:train],**options)
            setup = time.perf_counter()-started
            save_decoder(decoder, dem, path)
            started = time.perf_counter()
            prediction = decoder.decode_batch(dets[train:])
            seconds = time.perf_counter()-started
            fail = np.any(prediction != obs[train:], axis=1)
            case["validation"][family] = dict(errors=int(fail.sum()), shots=len(fail),
                setup_seconds=setup, decode_seconds=seconds, decoder_dir=path.name,
                metadata=decoder.metadata)
            write_json(output / "manifest.json", manifest)
        print("PILOT", case["id"], {f:v["errors"] for f,v in case["validation"].items()}, flush=True)
    for profile in PHYSICAL:
        for basis in ("X", "Z"):
            points = sorted([c for c in cases if c["profile"] == profile and c["basis"] == basis], key=lambda c:c["rounds"])
            for family in manifest["families"]:
                if any(family not in c['validation'] for c in points):
                    continue
                fit = challenge_fit(ROUNDS, [c["validation"][family]["errors"] for c in points],
                                    [c["validation"][family]["shots"] for c in points])
                scores.append(dict(profile=profile, basis=basis, family=family, fit=fit))
    frozen = dict(created_utc=utc(), primary_equation="P_L(r) = (1 - (1-2*epsilon_L)**r)/2; A=1",
                  primary_rounds=list(ROUNDS), selection="Minimum validation challenge LER, one family across all rounds",
                  scores=scores, arms={},
                  confirmation_protocol=dict(comparisons=6,alpha_family=.05,bootstrap_draws=10000,bootstrap_seed=2203,
                    promotion_rule='Promote per basis only if adjusted LER gain over all three standalone references; otherwise retain pilot-best standalone'))
    frozen['neural_profiles']=manifest.get('neural_profiles',{})
    for basis in ("X", "Z"):
        # Keep the released Ethan policy as the literal standalone reference.
        ethan_family = "mwpm" if basis == "X" else "ml"
        ethan = next(s for s in scores if s["basis"] == basis and s["profile"] == "ethan" and s["family"] == ethan_family)
        dara = choose(scores, basis, ("dara",))
        iyanu_families = {"mwpm", "mwpm_fit", "empirical_1", "mlp", "conv", "coherence"}
        iyanu = choose([s for s in scores if s["family"] in iyanu_families], basis, ("iyanu",))
        candidate = choose(scores, basis)
        frozen["arms"][basis] = {"ethan":ethan, "dara":dara, "iyanu":iyanu, "candidate":candidate}
    write_json(output / "pilot_results.json", scores)
    write_json(output / "frozen_selection.json", frozen)
    manifest["selection_sha256"] = digest(output / "frozen_selection.json")
    manifest["selection_frozen_utc"] = frozen["created_utc"]
    write_json(output / "manifest.json", manifest)
    print("FROZEN", {b:{a:(v["profile"],v["family"]) for a,v in arms.items()} for b,arms in frozen["arms"].items()}, flush=True)
    return frozen


def extend_neural(output,manifest):
    """Bounded second pilot stage; never consumes confirmation observations.

    Reuse Iyanu's MLP and conv training/fine-tuning on the literal Iyanu path
    and on the classical pilot's leading physical path in each memory basis.
    Keep the classical selection unchanged in an archive for transparency.
    """
    if any(c['phase']=='confirmation' for c in manifest['cases']):
        raise ValueError('Neural expansion must precede every confirmation submission.')
    frozen_path=output/'frozen_selection.json'
    archived=output/'classical_selection.json'
    if not archived.exists():
        frozen=json.loads(frozen_path.read_text())
        archived.write_bytes(frozen_path.read_bytes())
        manifest['neural_profiles']={b:sorted({'iyanu',arms['candidate']['profile']})
                                     for b,arms in frozen['arms'].items()}
        manifest['families']+=['mlp','conv']
        manifest['neural_expansion_utc']=utc()
        source_files=[Path(__file__),Path(__file__).parent/'decoders'/'hybrid.py']
        manifest['neural_source_hashes']={str(p.relative_to(Path.cwd())):digest(p) for p in source_files}
        for source in source_files:
            (output/(source.name+'.neural.source.txt')).write_bytes(source.read_bytes())
        frozen_path.unlink()
        write_json(output/'manifest.json',manifest)
    elif frozen_path.exists():
        raise ValueError('Neural pilot selection already frozen.')
    return pilot_analysis(output,manifest)


def extend_posterior(output,manifest):
    """Test the existing posterior-plus-training-count rule before confirmation."""
    if any(c['phase']=='confirmation' for c in manifest['cases']):
        raise ValueError('Posterior expansion must precede every confirmation submission.')
    frozen_path=output/'frozen_selection.json'
    archive=output/'neural_selection.json'
    if not archive.exists():
        archive.write_bytes(frozen_path.read_bytes())
        manifest['families']+=['posterior_1','posterior_10','posterior_100']
        manifest['posterior_expansion_utc']=utc()
        frozen_path.unlink()
        write_json(output/'manifest.json',manifest)
    elif frozen_path.exists():
        raise ValueError('Posterior pilot already frozen.')
    return pilot_analysis(output,manifest)


def extend_coherence(output,manifest):
    if any(c['phase']=='confirmation' for c in manifest['cases']):
        raise ValueError('Coherence expansion must precede confirmation submission.')
    path=output/'frozen_selection.json'
    archive=output/'posterior_selection.json'
    if not archive.exists():
        archive.write_bytes(path.read_bytes())
        manifest['families'].append('coherence')
        manifest['coherence_expansion_utc']=utc()
        path.unlink();write_json(output/'manifest.json',manifest)
    elif path.exists():
        raise ValueError('Coherence selection already frozen.')
    return pilot_analysis(output,manifest)


def confirmation_analysis(output, manifest):
    if digest(output / "frozen_selection.json") != manifest["selection_sha256"]:
        raise ValueError("Frozen selection changed after confirmation started.")
    seal_inference(output,manifest,verify_only=True)
    frozen = json.loads((output / "frozen_selection.json").read_text())
    cases = [c for c in manifest["cases"] if c["phase"] == "confirmation"]
    expected=sum(len({s['profile'] for s in arms.values()}) for arms in frozen['arms'].values())*len(ROUNDS)*2
    if len(cases)!=expected or len({c['id'] for c in cases})!=expected or any(not c.get("data_file") for c in cases):
        raise ValueError("Confirmation jobs incomplete; retrieve existing IDs.")
    points, failures = [], {}
    for basis, arms in frozen["arms"].items():
        for arm, selection in arms.items():
            for rounds in ROUNDS:
                pilot = next(c for c in manifest["cases"] if c["phase"] == "pilot" and
                             (c["basis"],c["rounds"],c["profile"]) == (basis,rounds,selection["profile"]))
                decoder = load_decoder(output / pilot["validation"][selection["family"]]["decoder_dir"])
                for case in cases:
                    if (case["basis"],case["rounds"],case["profile"]) != (basis,rounds,selection["profile"]):
                        continue
                    if case["source_sha256"] != pilot["source_sha256"] or case["native_sha256"] != pilot["native_sha256"]:
                        raise ValueError("Confirmation physical circuit differs from frozen pilot.")
                    dets, obs = arrays(output, case)
                    decoder.decode_batch(dets[:min(32,len(dets))])
                    started = time.perf_counter()
                    prediction = decoder.decode_batch(dets)
                    elapsed = time.perf_counter()-started
                    fail = np.any(prediction != obs, axis=1)
                    failures.setdefault((basis,arm), {}).setdefault(rounds, {})[case["block"]] = fail
                    points.append(dict(basis=basis, arm=arm, profile=selection["profile"], family=selection["family"],
                        rounds=rounds, block=case["block"], errors=int(fail.sum()), shots=len(fail),
                        probability=float(fail.mean()), ci95=wilson_interval(int(fail.sum()), len(fail)),
                        decode_seconds=elapsed, shots_per_second=len(fail)/elapsed, timing_scope='warmed batch; load/build/conversion excluded',
                        case_id=case["id"], dataset_sha256=case["dataset_sha256"]))
                    np.savez_compressed(output / f"{case['id']}.{arm}.predictions.npz", predictions=prediction, failures=fail)
    fits, block_fits, comparisons = [], [], []
    for (basis,arm), by_round in failures.items():
        k = [sum(int(v.sum()) for v in by_round[r].values()) for r in ROUNDS]
        n = [sum(len(v) for v in by_round[r].values()) for r in ROUNDS]
        fits.append(dict(basis=basis, arm=arm, **challenge_fit(ROUNDS, k, n)))
        for block in sorted(by_round[ROUNDS[0]]):
            block_fits.append(dict(basis=basis,arm=arm,block=block,
                **challenge_fit(ROUNDS,[int(by_round[r][block].sum()) for r in ROUNDS],
                                [len(by_round[r][block]) for r in ROUNDS])))
    for basis, arms in frozen["arms"].items():
        for baseline in ("ethan", "dara", "iyanu"):
            paired = arms[baseline]["profile"] == arms["candidate"]["profile"]
            comparison = ler_difference(failures[basis,baseline], failures[basis,"candidate"], paired=paired,
                                        alpha=.05/6, seed=2203)
            comparisons.append(dict(basis=basis, baseline=baseline, **comparison))
    proven = all(v["ci"][0] > 0 for v in comparisons)
    release={}
    for basis,arms in frozen['arms'].items():
        promote=all(v['ci'][0]>0 for v in comparisons if v['basis']==basis)
        reference=min(('ethan','dara','iyanu'),key=lambda name:arms[name]['fit']['epsilon_per_round'])
        adopted='candidate' if promote else reference
        release[basis]=dict(arm=adopted,profile=arms[adopted]['profile'],family=arms[adopted]['family'],
            promoted=promote,rule='Fresh adjusted LER gain over all standalone references; otherwise retain pilot-best standalone',
            calibration_set_id=manifest['calibration_set_id'],rounds=list(ROUNDS),
            fit=next(f for f in fits if (f['basis'],f['arm'])==(basis,adopted)))
    report = dict(written_utc=utc(), calibration_set_id=manifest["calibration_set_id"],
                  primary_rounds=list(ROUNDS), selection_sha256=manifest["selection_sha256"],
                  points=points, fits=fits, block_fits=block_fits,comparisons=comparisons,adopted_policy=release,
                  outperforms_all_standalones_both_bases=proven,
                  uncertainty="Six predeclared comparisons; Bonferroni 95% family confidence; conditional fixed-shot inference",
                  limits=["Distance 3, one Garnet calibration; no distance-scaling or bare-qubit claim",
                          "R1-3 fixed before acquisition because released exact LUT supports <=24 detectors",
                          "Rejected stationary fits remain descriptive challenge-equation estimates",
                          "Dara uses its median/assumed logical prior; native profiles use measured native-gate priors",
                          "No postselection, hidden leakage flags, or confirmation-label training"])
    write_json(output / "results.json", report)
    write_json(output / "adopted_policy.json",release)
    lines = ["# Garnet challenge LER comparison", "", "All shots retained. Frozen pilot choices; independent interleaved confirmation blocks.",
             "Primary equation: `P_L(r) = [1-(1-2 epsilon_L)^r]/2`, rounds 1, 2, 3; amplitude fixed at 1.", "",
             "| Basis | Arm | LER / round (95% profile CI) | Errors at R1/R2/R3 | Fit deviance / df |", "|---|---|---|---|---|"]
    for f in sorted(fits, key=lambda f:(f["basis"],f["arm"])):
        lo, hi = f["ci95_profile"]
        lines.append(f"| {f['basis']} | {f['arm']} | {100*f['epsilon_per_round']:.3f}% ({100*lo:.3f}–{100*hi:.3f}%) | {f['errors']} / {f['shots']} | {f['deviance']:.2f} / {f['nominal_df']} |")
    lines += ["", f"Improvement over every standalone in both bases established: **{proven}**.", "",
              "Positive differences favor candidate; intervals account for the six predeclared comparisons."]
    for v in comparisons:
        lines.append(f"- {v['basis']} vs {v['baseline']}: {100*v['absolute_ler_reduction']:.3f} percentage points; CI {100*v['ci'][0]:.3f}–{100*v['ci'][1]:.3f}; paired={v['paired']}.")
    lines += ['', 'Adopted policy (predeclared promotion/fallback rule):']
    for basis,policy in release.items():
        lines.append(f"- {basis}: {policy['profile']} / {policy['family']}; promoted={policy['promoted']}.")
    lines += ["", *["- "+v for v in report["limits"]]]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    plot(output, report)
    print("RESULT", {f['basis']+':'+f['arm']:round(100*f['epsilon_per_round'],4) for f in fits}, "proven", proven, flush=True)
    return report


def seal_inference(output,manifest,*,verify_only=False):
    """Bind every selected inference byte before new confirmation acquisition."""
    seal_path=output/'frozen_artifacts.json'
    if seal_path.exists():
        if digest(seal_path)!=manifest['inference_artifacts_sha256']:
            raise ValueError('Frozen inference artifact manifest changed.')
        sealed=json.loads(seal_path.read_text())
        for filename,expected in sealed.items():
            if digest(output/filename)!=expected:
                raise ValueError('Frozen decoder state changed: '+filename)
        return
    if verify_only or any(c['phase']=='confirmation' for c in manifest['cases']):
        raise ValueError('Decoder state must be sealed before confirmation.')
    frozen=json.loads((output/'frozen_selection.json').read_text())
    sealed={}
    for basis,arms in frozen['arms'].items():
        for selection in arms.values():
            for r in ROUNDS:
                case=next(c for c in manifest['cases'] if c['phase']=='pilot' and
                          (c['basis'],c['rounds'],c['profile'])==(basis,r,selection['profile']))
                directory=output/case['validation'][selection['family']]['decoder_dir']
                for path in sorted(directory.iterdir()):
                    if path.is_file():
                        sealed[path.relative_to(output).as_posix()]=digest(path)
    write_json(seal_path,sealed)
    manifest['inference_artifacts_sha256']=digest(seal_path)
    manifest['inference_sealed_utc']=utc()
    write_json(output/'manifest.json',manifest)


def plot(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4),sharey=True)
    colors = dict(ethan="tab:blue", dara="tab:orange", iyanu="tab:green", candidate="tab:red")
    for ax, basis in zip(axes, ("X","Z")):
        for fit in [f for f in report["fits"] if f["basis"] == basis]:
            p, intervals = np.array(fit["probabilities"]), np.array(fit["wilson_ci95"])
            ax.errorbar(ROUNDS,100*p,yerr=100*np.array([p-intervals[:,0],intervals[:,1]-p]),fmt="o",color=colors[fit["arm"]])
            x=np.linspace(1,3,100)
            rejected=fit['stationary_model_rejected_5pct']
            ax.plot(x,50*(1-(1-2*fit["epsilon_per_round"])**x),color=colors[fit["arm"]],
                    linestyle='--' if rejected else '-',
                    label=f"{fit['arm']}: {100*fit['epsilon_per_round']:.2f}%/round"+(' *' if rejected else ''))
        ax.set(title=f"Garnet d=3 {basis} memory",xlabel="QEC rounds",xticks=ROUNDS)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel("Logical failure probability (%)")
    fig.suptitle("Fresh confirmation; fixed IQM challenge equation (A=1)")
    fig.text(.5,.01,'* Stationary model rejected (p < .05): descriptive challenge-equation fit.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(output/"ler.png",dpi=180);plt.close(fig)


def execute_policy(args,study,manifest,backend,executor,cal,quality,mapping,secret,*,active_calibration_set_id):
    """Execute the confirmed policy; refuse changed calibration or physical frame."""
    from iqm.station_control.client.serializers.run_serializers import serialize_run_definition
    from iqm.iqm_client import CircuitCompilationOptions,DDMode
    if not args.policy_output:
        raise ValueError('--policy-output is required for a separate immutable execution ledger.')
    if not 1<=args.policy_shots<=20000:
        raise ValueError('Policy shots must be 1..20000.')
    if str(active_calibration_set_id)!=manifest['calibration_set_id']:
        raise ValueError('Device calibration changed; requalify before using frozen hardware-trained decoder states.')
    seal_inference(study,manifest,verify_only=True)
    frozen=json.loads((study/'frozen_selection.json').read_text())
    adopted=json.loads((study/'adopted_policy.json').read_text())[args.policy_basis]
    selection=frozen['arms'][args.policy_basis][adopted['arm']]
    if (selection['profile'],selection['family'])!=(adopted['profile'],adopted['family']):
        raise ValueError('Adopted policy differs from its frozen selection.')
    pilot=next(c for c in manifest['cases'] if c['phase']=='pilot' and
               (c['basis'],c['rounds'],c['profile'])==(args.policy_basis,args.policy_rounds,selection['profile']))
    decoder=load_decoder(study/pilot['validation'][selection['family']]['decoder_dir'])
    output=args.policy_output
    ledger_path=output/'manifest.json'
    if args.submit:
        output.mkdir(parents=True,exist_ok=False)
        case=dict(id=f"selected_{args.policy_basis}_r{args.policy_rounds}",phase='policy',
                  profile=selection['profile'],basis=args.policy_basis,rounds=args.policy_rounds,shots=args.policy_shots)
        compiled,run,context=prepare_case(output,case,backend,executor,cal,quality,mapping)
        if any(case[key]!=pilot[key] for key in ('source_sha256','native_sha256','model_sha256')):
            raise ValueError('Prepared policy differs from the confirmed source/native circuit or prior.')
        ledger=dict(created_utc=utc(),calibration_set_id=manifest['calibration_set_id'],
                    selection_sha256=manifest['selection_sha256'],inference_artifacts_sha256=manifest['inference_artifacts_sha256'],
                    policy=adopted,cases=[case])
        case['status']='submission_started';write_json(ledger_path,ledger)
        if case['transport']=='pulse':
            (output/f"{case['id']}.playlist.pb").write_bytes(serialize_run_definition(run).SerializeToString())
            job=executor.submit(run,context);case['job_id']=str(job.job_id)
        else:
            job=backend.run(compiled,shots=case['shots'],
                            circuit_compilation_options=CircuitCompilationOptions(dd_mode=DDMode.ENABLED))
            case['job_id']=job.job_id()
        case['status']='submitted';write_json(ledger_path,ledger)
        print('POLICY_SUBMITTED',case['job_id'],flush=True)
    else:
        ledger=json.loads(ledger_path.read_text())
        if ledger['selection_sha256']!=manifest['selection_sha256'] or ledger['inference_artifacts_sha256']!=manifest['inference_artifacts_sha256']:
            raise ValueError('Execution ledger belongs to a different frozen study.')
        case=ledger['cases'][0]
        if (case['basis'],case['rounds'])!=(args.policy_basis,args.policy_rounds):
            raise ValueError('Resume arguments differ from ledgered policy execution.')
    collect(output,ledger,backend,executor,secret)
    if not case.get('data_file'):
        print('POLICY_PENDING',case.get('job_id'),'Use --resume with the same policy arguments.',flush=True)
        return
    dets,obs=arrays(output,case)
    predictions=decoder.decode_batch(dets)
    failed=np.any(predictions!=obs,axis=1)
    result=dict(job_id=case['job_id'],basis=case['basis'],rounds=case['rounds'],shots=len(failed),errors=int(failed.sum()),
                failure_probability=float(failed.mean()),ci95=wilson_interval(int(failed.sum()),len(failed)),
                profile=selection['profile'],family=selection['family'],calibration_set_id=manifest['calibration_set_id'],
                metric='Single-count logical failure probability; not a fitted per-round LER',
                source_and_native_frames_verified=True,all_shots_retained=True)
    np.savez_compressed(output/'predictions.npz',predictions=predictions,failures=failed)
    write_json(output/'results.json',result)
    print('POLICY_COMPLETE',result,flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--phase",choices=["pilot","confirmation"],default="pilot")
    parser.add_argument("--token-file",type=Path,default=Path("apikey.jom"))
    parser.add_argument("--submit",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--analyze",action="store_true")
    parser.add_argument("--extend-neural",action="store_true",help="Offline second pilot stage before confirmation")
    parser.add_argument("--extend-posterior",action="store_true",help="Offline posterior-plus-counts pilot stage")
    parser.add_argument("--extend-coherence",action="store_true",help="Offline Iyanu coherence-prior pilot stage")
    parser.add_argument("--train-shots",type=int,default=6000)
    parser.add_argument("--validation-shots",type=int,default=2000)
    parser.add_argument("--confirmation-shots",type=int,default=6000)
    parser.add_argument("--runtime",type=Path,help="Optional isolated directory containing PyTorch")
    parser.add_argument('--policy-basis',choices=['X','Z'],help='Execute adopted policy separately from headline acquisition')
    parser.add_argument('--policy-rounds',type=int,choices=ROUNDS,default=3)
    parser.add_argument('--policy-shots',type=int,default=1000)
    parser.add_argument('--policy-output',type=Path)
    args=parser.parse_args(argv)
    if args.runtime:
        import sys
        sys.path.insert(0,str(args.runtime.resolve()))
        import torch
        torch.set_num_threads(2)
    if sum([args.submit,args.resume,args.analyze,args.extend_neural,args.extend_posterior,args.extend_coherence]) != 1:
        parser.error("Choose exactly one submission, retrieval, analysis or pilot-expansion action.")
    if not 1000<=args.train_shots<=12000 or not 500<=args.validation_shots<=4000 or not 1000<=args.confirmation_shots<=12000 or args.confirmation_shots%2:
        parser.error("Shot counts outside bounded study limits, or confirmation count odd.")
    output=args.run
    path=output/"manifest.json"
    if args.analyze or args.extend_neural or args.extend_posterior or args.extend_coherence:
        manifest=json.loads(path.read_text())
        if args.extend_coherence:
            return extend_coherence(output,manifest)
        if args.extend_posterior:
            return extend_posterior(output,manifest)
        if args.extend_neural:
            return extend_neural(output,manifest)
        return pilot_analysis(output,manifest) if args.phase=="pilot" else confirmation_analysis(output,manifest)
    from iqm_surface_code.benchmark_garnet import read_token,calibration_from_observations,select_native_layout
    from iqm_surface_code.run_on_hardware import connect_backend
    from iqm_surface_code.hardware_optimization import PulseExecutor
    from iqm_surface_code.calibration import CalibrationSnapshot
    from iqm.qiskit_iqm.iqm_provider import IQMBackend
    from iqm.station_control.client.serializers.run_serializers import serialize_run_definition
    from iqm.iqm_client import CircuitCompilationOptions,DDMode
    secret=read_token(args.token_file)
    try:
        backend=connect_backend(secret,"garnet")
        active_calibration_set_id=str(backend.architecture.calibration_set_id)
        if path.exists():
            manifest=json.loads(path.read_text())
            backend=IQMBackend(backend.client,calibration_set_id=manifest["calibration_set_id"])
            cal=CalibrationSnapshot.load(output/"calibration.json")
            quality=json.loads((output/"quality_metrics.json").read_text())
            mapping={int(k):v for k,v in manifest["placement"].items()}
        else:
            if args.resume or args.phase!="pilot":
                parser.error("Existing pilot manifest required.")
            output.mkdir(parents=True,exist_ok=False)
            calid=backend.architecture.calibration_set_id
            backend=IQMBackend(backend.client,calibration_set_id=calid)
            architecture=backend.architecture.model_dump(mode="json")
            quality=backend.client.get_quality_metric_set(calid).model_dump(mode="json")
            pulse=backend.client.get_calibration_set(calid).model_dump(mode="json")
            cal=calibration_from_observations(architecture,quality,pulse);cal.save(output/"calibration.json")
            write_json(output/"architecture.json",architecture);write_json(output/"quality_metrics.json",quality)
            write_json(output/"pulse_calibration.json",pulse)
            from iqm_surface_code.surface_code import make_stim_circuit
            edges=[(backend.index_to_qubit_name(a),backend.index_to_qubit_name(b)) for a,b in backend.coupling_map.get_edges()]
            mapping,_=select_native_layout(make_stim_circuit(3,3,gate_basis="cz"),edges,backend.physical_qubits,cal)
            versions={p:importlib.metadata.version(p) for p in ("stim","numpy","scipy","pymatching","qiskit","iqm-client","iqm-pulla")}
            manifest=dict(created_utc=utc(),calibration_set_id=str(calid),placement=mapping,cases=[],
                rounds=list(ROUNDS),families=list(FAMILIES),train_shots=args.train_shots,
                validation_shots=args.validation_shots,confirmation_shots=args.confirmation_shots,
                fixed_confirmation_blocks=2,versions=versions,
                base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                protocol="Pilot R1-3 minimum challenge LER; freeze one family/profile per basis; independent confirmation; six Bonferroni comparisons",
                physical_definitions=dict(ethan="Released X fused/Z fenced native optimization + data XY4",
                    dara="sqrtY + measurement barriers + server default DD",
                    iyanu="Calibrated literal-H native circuit + server default DD",
                    combined="Dara sqrtY + Ethan X fused/Z fenced native optimization + data XY4"))
            source_files=[Path(__file__),Path(__file__).parent/'decoders'/'hybrid.py']
            manifest['source_hashes']={str(p.relative_to(Path.cwd())):digest(p) for p in source_files}
            for p in source_files:
                (output/(p.name+'.source.txt')).write_bytes(p.read_bytes())
            write_json(path,manifest)
        executor=PulseExecutor(backend,secret)
        if args.policy_basis:
            if not (args.submit or args.resume):
                raise ValueError('Policy execution requires --submit or --resume.')
            return execute_policy(args,output,manifest,backend,executor,cal,quality,mapping,secret,
                                  active_calibration_set_id=active_calibration_set_id)
        if args.submit:
            if any(c["phase"]==args.phase for c in manifest["cases"]):
                raise ValueError("Phase already ledgered. Use --resume; never resubmit an existing phase.")
            if args.phase=="pilot":
                specs=[dict(id=f"pilot_{p}_{b}_r{r}",phase="pilot",profile=p,basis=b,rounds=r,
                            shots=manifest["train_shots"]+manifest["validation_shots"])
                       for p in PHYSICAL for b in ("X","Z") for r in ROUNDS]
            else:
                if digest(output/"frozen_selection.json")!=manifest["selection_sha256"]:
                    raise ValueError("Frozen selection hash mismatch.")
                seal_inference(output,manifest)
                frozen=json.loads((output/"frozen_selection.json").read_text())
                specs=[dict(id=f"confirmation_b{block}_{p}_{b}_r{r}",phase="confirmation",profile=p,basis=b,rounds=r,
                            block=block,shots=manifest["confirmation_shots"]//2)
                       for block in range(2) for b,arms in frozen["arms"].items()
                       for p in sorted({s["profile"] for s in arms.values()}) for r in ROUNDS]
            random.Random(2207 if args.phase=="pilot" else 2213).shuffle(specs)
            # Interleave within each block, while keeping block acquisition windows separate.
            specs.sort(key=lambda c:c.get("block",0))
            for case in specs:
                compiled,run,context=prepare_case(output,case,backend,executor,cal,quality,mapping)
                case.update(status="submission_started",submission_started_utc=utc())
                manifest["cases"].append(case);write_json(path,manifest)
                if case["transport"]=="pulse":
                    (output/f"{case['id']}.playlist.pb").write_bytes(serialize_run_definition(run).SerializeToString())
                    job=executor.submit(run,context)
                    case["job_id"]=str(job.job_id)
                else:
                    job=backend.run(compiled,shots=case["shots"],memory=True,
                                    circuit_compilation_options=CircuitCompilationOptions(dd_mode=DDMode.ENABLED))
                    case["job_id"]=job.job_id()
                case["status"]="submitted";write_json(path,manifest)
                print("SUBMITTED",case["id"],case["job_id"],flush=True)
        collect(output,manifest,backend,executor,secret)
        incomplete=[c["id"] for c in manifest["cases"] if not c.get("data_file")]
        if incomplete:
            print("PENDING_OR_FAILED",incomplete,"Use --resume; no new jobs.",flush=True)
            return
        return pilot_analysis(output,manifest) if args.phase=="pilot" else confirmation_analysis(output,manifest)
    except Exception as exc:
        message=str(exc).replace(secret,"[REDACTED]")
        if output.exists():
            write_json(output/"error.json",dict(type=type(exc).__name__,message=message,written_utc=utc()))
        raise SystemExit(type(exc).__name__+": "+message) from None


if __name__=="__main__":
    main()
