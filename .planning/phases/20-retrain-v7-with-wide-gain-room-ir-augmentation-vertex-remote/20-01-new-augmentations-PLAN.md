---
phase: 20
plan: 01
type: execute
wave: 1
depends_on:
  - "20-00"
files_modified:
  - src/acoustic/training/augmentation.py
  - requirements.txt
  - requirements-vertex.txt
autonomous: true
requirements:
  - D-01
  - D-02
  - D-03
  - D-04
  - D-05
  - D-06
  - D-07
  - D-08
must_haves:
  truths:
    - "WideGainAugmentation applies ±40 dB uniform gain per call and clips to [-1, 1]"
    - "RoomIRAugmentation pre-generates a pool of 500 RIRs at init and convolves with scipy.signal.fftconvolve"
    - "Both classes are pickle-safe (DataLoader num_workers > 0 compatible)"
    - "Output length and dtype are preserved by both augmentations"
    - "All Wave 0 unit tests for these classes turn GREEN"
  artifacts:
    - path: src/acoustic/training/augmentation.py
      provides: "class WideGainAugmentation, class RoomIRAugmentation"
      contains: "class WideGainAugmentation"
    - path: requirements.txt
      provides: "pyroomacoustics>=0.8,<0.11 dependency"
    - path: requirements-vertex.txt
      provides: "pyroomacoustics>=0.8,<0.11 dependency for Vertex image"
  key_links:
    - from: src/acoustic/training/augmentation.py
      to: pyroomacoustics
      via: "pra.ShoeBox(...).compute_rir() in RoomIRAugmentation._generate_one"
      pattern: "pra\\.ShoeBox"
---

<objective>
Implement the two genuinely new augmentation classes mandated by D-01..D-08:
WideGainAugmentation (±40 dB uniform gain with clipping) and RoomIRAugmentation (procedural
pyroomacoustics ShoeBox RIRs precomputed into a pool of 500, convolved per call with
scipy.signal.fftconvolve). Both are pickle-safe so they work with DataLoader num_workers > 0.

Purpose: Closes the deployment-level gap (UMA-16 ~-82 dBFS vs DADS ~-25 dBFS, ~50-60 dB) that
caused v6 to never trigger on real captures, AND adds procedural reverb so v7 generalizes across
indoor/outdoor environments. These are the only two NEW classes in Phase 20 — everything else
is config plumbing.

Output: Two new classes in src/acoustic/training/augmentation.py, pyroomacoustics added to
both requirements files, all Wave 0 unit tests for these classes pass.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
@src/acoustic/training/augmentation.py
@tests/unit/test_wide_gain_augmentation.py
@tests/unit/test_room_ir_augmentation.py

<interfaces>
Existing pickle-safe pattern in src/acoustic/training/augmentation.py:
- ComposedAugmentation(augmentations: list) — chains __call__'ables
- BackgroundNoiseMixer.__call__(audio: np.ndarray) -> np.ndarray
- AudiomentationsAugmentation: class-based, not closures (pickle-safe)

pyroomacoustics ShoeBox API (verified, see 20-RESEARCH.md):
```python
import pyroomacoustics as pra
room = pra.ShoeBox(
    [Lx, Ly, Lz],            # room dimensions in meters
    fs=16000,
    materials=pra.Material(absorption_coefficient),
    max_order=10,
)
room.add_source([sx, sy, sz])
room.add_microphone([mx, my, mz])
room.compute_rir()
rir = room.rir[0][0]  # numpy array, mic 0, source 0
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add pyroomacoustics dependency + implement WideGainAugmentation</name>
  <files>
    requirements.txt,
    requirements-vertex.txt,
    src/acoustic/training/augmentation.py
  </files>
  <read_first>
    requirements.txt,
    requirements-vertex.txt,
    src/acoustic/training/augmentation.py,
    tests/unit/test_wide_gain_augmentation.py,
    .planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md
  </read_first>
  <behavior>
    After this task:
    - `import pyroomacoustics` works in the local env (used by Task 2)
    - `from src.acoustic.training.augmentation import WideGainAugmentation` succeeds
    - All five WideGain unit tests pass (RED → GREEN)
  </behavior>
  <action>
    Step 1 — Add `pyroomacoustics>=0.8,<0.11` to BOTH requirements.txt and requirements-vertex.txt. Add the line in the audio-processing section near scipy/soundfile. Run `pip install pyroomacoustics>=0.8,<0.11` locally to verify install.

    Step 2 — In src/acoustic/training/augmentation.py append the WideGainAugmentation class. Use exactly this signature and behavior (per D-01..D-04):

    ```python
    class WideGainAugmentation:
        """Wide ±wide_gain_db uniform gain (Phase 20 D-01..D-04).

        Replaces the WaveformAugmentation small-gain stage. Runs as a separate
        pre-stage in ComposedAugmentation. Clips to [-1, 1] before returning so
        downstream RIR convolution sees a bounded signal (Pitfall 2 in Phase 20
        research).
        """

        def __init__(self, wide_gain_db: float = 40.0, p: float = 1.0) -> None:
            self._wide_gain_db = float(wide_gain_db)
            self._p = float(p)
            self._rng = np.random.default_rng()

        def __call__(self, audio: np.ndarray) -> np.ndarray:
            if self._rng.random() >= self._p:
                return audio
            gain_db = self._rng.uniform(-self._wide_gain_db, self._wide_gain_db)
            gain_linear = 10.0 ** (gain_db / 20.0)
            out = (audio * gain_linear).astype(np.float32)
            return np.clip(out, -1.0, 1.0)

        def __getstate__(self):
            # Pickle-safe: exclude live RNG (rebuilt on unpickle by worker)
            return {"wide_gain_db": self._wide_gain_db, "p": self._p}

        def __setstate__(self, state):
            self._wide_gain_db = state["wide_gain_db"]
            self._p = state["p"]
            self._rng = np.random.default_rng()
    ```

    Use the existing module-level `import numpy as np`. Do NOT modify the existing
    WaveformAugmentation class — keep for backward compatibility per CONTEXT.md (D-04).
  </action>
  <verify>
    <automated>pytest tests/unit/test_wide_gain_augmentation.py -x -q && python -c "import pyroomacoustics; print(pyroomacoustics.__version__)"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -E "pyroomacoustics" requirements.txt requirements-vertex.txt` returns matches in both files
    - `grep -n "class WideGainAugmentation" src/acoustic/training/augmentation.py` returns one match
    - `grep -n "np.clip(out, -1.0, 1.0)" src/acoustic/training/augmentation.py` returns one match (Pitfall 2 mitigation)
    - `grep -n "__getstate__" src/acoustic/training/augmentation.py` shows pickle-safe hooks
    - `pytest tests/unit/test_wide_gain_augmentation.py -x -q` exits 0 (all five tests GREEN)
    - `python -c "import pyroomacoustics"` works without ImportError
  </acceptance_criteria>
  <done>
    pyroomacoustics installed and listed in both requirements files. WideGainAugmentation class
    exists with the exact signature, clipping, and pickle hooks. All RED tests for it now GREEN.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement RoomIRAugmentation (precomputed pool + scipy.signal.fftconvolve)</name>
  <files>
    src/acoustic/training/augmentation.py
  </files>
  <read_first>
    src/acoustic/training/augmentation.py,
    tests/unit/test_room_ir_augmentation.py,
    .planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
  </read_first>
  <behavior>
    After this task:
    - `from src.acoustic.training.augmentation import RoomIRAugmentation` works
    - All six RED tests for RoomIRAugmentation are GREEN
    - Pool of 500 RIRs builds in <30s; max_order=10; truncated to 1 second
    - Output length preserved; renormalized to preserve perceived peak level
  </behavior>
  <action>
    Append the RoomIRAugmentation class to src/acoustic/training/augmentation.py. Use the pattern
    documented in 20-RESEARCH.md "Procedural RIR Generation Pattern" verbatim with these exact values
    (D-05, D-06, D-07, D-08):

    ```python
    import pyroomacoustics as pra
    from scipy.signal import fftconvolve

    class RoomIRAugmentation:
        """Procedural ShoeBox RIR convolution (Phase 20 D-05..D-08).

        Pre-generates a pool of pool_size RIRs at construction (via
        pyroomacoustics.ShoeBox image source method). Each __call__ samples one
        RIR from the pool and convolves with the input via
        scipy.signal.fftconvolve. Faster than per-call generation (~5-15 ms per
        ShoeBox simulation -- see Pitfall 3 in 20-RESEARCH.md) and removes
        pyroomacoustics from the per-batch hot path.
        """

        def __init__(
            self,
            sample_rate: int = 16000,
            pool_size: int = 500,
            room_dim_min: tuple[float, float, float] = (3.0, 3.0, 2.5),
            room_dim_max: tuple[float, float, float] = (12.0, 12.0, 4.0),
            absorption_range: tuple[float, float] = (0.2, 0.7),
            source_distance_range: tuple[float, float] = (1.0, 8.0),
            max_order: int = 10,
            p: float = 0.7,
            seed: int = 42,
        ) -> None:
            self._sr = int(sample_rate)
            self._pool_size = int(pool_size)
            self._room_dim_min = tuple(room_dim_min)
            self._room_dim_max = tuple(room_dim_max)
            self._absorption_range = tuple(absorption_range)
            self._source_distance_range = tuple(source_distance_range)
            self._max_order = int(max_order)
            self._p = float(p)
            self._seed = int(seed)
            init_rng = np.random.default_rng(self._seed)
            self._pool: list[np.ndarray] = [
                self._generate_one(init_rng) for _ in range(self._pool_size)
            ]
            self._call_rng = np.random.default_rng()

        def _generate_one(self, rng: np.random.Generator) -> np.ndarray:
            room_dim = rng.uniform(self._room_dim_min, self._room_dim_max)  # shape (3,)
            absorption = float(rng.uniform(*self._absorption_range))
            room = pra.ShoeBox(
                room_dim.tolist(),
                fs=self._sr,
                materials=pra.Material(absorption),
                max_order=self._max_order,
            )
            mic_pos = room_dim / 2.0
            src_pos = mic_pos + np.array([1.0, 0.0, 0.0])  # default fallback
            for _ in range(8):
                dist = float(rng.uniform(*self._source_distance_range))
                theta = float(rng.uniform(0, 2 * np.pi))
                phi = float(rng.uniform(np.pi / 4, 3 * np.pi / 4))
                offset = np.array([
                    dist * np.sin(phi) * np.cos(theta),
                    dist * np.sin(phi) * np.sin(theta),
                    dist * np.cos(phi),
                ])
                candidate = mic_pos + offset
                margin = 0.3
                if np.all(candidate > margin) and np.all(candidate < room_dim - margin):
                    src_pos = candidate
                    break
            room.add_source(src_pos.tolist())
            room.add_microphone(mic_pos.tolist())
            room.compute_rir()
            rir = np.asarray(room.rir[0][0], dtype=np.float32)
            max_len = self._sr  # 1 second cap (Pitfall 3)
            if len(rir) > max_len:
                rir = rir[:max_len]
            return rir

        def __call__(self, audio: np.ndarray) -> np.ndarray:
            if not self._pool or self._call_rng.random() >= self._p:
                return audio
            rir = self._pool[self._call_rng.integers(len(self._pool))]
            out = fftconvolve(audio, rir, mode="full")[: len(audio)]
            peak_in = float(np.abs(audio).max())
            peak_out = float(np.abs(out).max())
            if peak_out > 1e-8 and peak_in > 0:
                out = out * (peak_in / peak_out)
            return out.astype(np.float32)

        def __getstate__(self):
            # Pool is reproducible from seed -- exclude RNG and pool, rebuild on unpickle.
            return {
                "sample_rate": self._sr,
                "pool_size": self._pool_size,
                "room_dim_min": self._room_dim_min,
                "room_dim_max": self._room_dim_max,
                "absorption_range": self._absorption_range,
                "source_distance_range": self._source_distance_range,
                "max_order": self._max_order,
                "p": self._p,
                "seed": self._seed,
            }

        def __setstate__(self, state):
            self.__init__(**state)
    ```

    Add `from scipy.signal import fftconvolve` and `import pyroomacoustics as pra` near the top
    of the file. Keep `numpy as np` import as-is.
  </action>
  <verify>
    <automated>pytest tests/unit/test_room_ir_augmentation.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "class RoomIRAugmentation" src/acoustic/training/augmentation.py` returns one match
    - `grep -n "import pyroomacoustics as pra" src/acoustic/training/augmentation.py` returns one match
    - `grep -n "from scipy.signal import fftconvolve" src/acoustic/training/augmentation.py` returns one match
    - `grep -n "max_order=self._max_order" src/acoustic/training/augmentation.py` confirms Pitfall 3 mitigation
    - `grep -n "max_len = self._sr" src/acoustic/training/augmentation.py` confirms RIR truncation
    - `pytest tests/unit/test_room_ir_augmentation.py -x -q` exits 0 (all six tests GREEN)
    - `python -c "from src.acoustic.training.augmentation import RoomIRAugmentation; import pickle; r = RoomIRAugmentation(pool_size=4); pickle.loads(pickle.dumps(r))"` exits 0
  </acceptance_criteria>
  <done>
    RoomIRAugmentation class exists, pool builds with seeded RNG, __call__ uses scipy.signal.fftconvolve,
    pickle round-trip works, all six unit tests GREEN.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| pip install → local + Vertex env | New dependency `pyroomacoustics` introduced from PyPI. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-20-01-01 | Tampering | pyroomacoustics PyPI package | mitigate | Pin version range `>=0.8,<0.11` in both requirements files. Maintained by LCAV (EPFL), widely used in audio research. |
| T-20-01-02 | DoS | RIR pool generation at startup | mitigate | Cap `max_order=10` and pool truncation to 1 s prevents pathological generation times (Pitfall 3 mitigation). Pool builds in <30 s for 500 rooms per research. |
| T-20-01-03 | Tampering | Pickled RIR augmentation in DataLoader | accept | Process-local pickle for DataLoader workers only; not deserialized from untrusted sources. |
</threat_model>

<verification>
- Wave 0 RED tests for WideGainAugmentation and RoomIRAugmentation are now GREEN
- pyroomacoustics importable in local env and listed in requirements-vertex.txt for Vertex builds
- Both classes pickle-safe (round-trip test embedded in unit tests)
</verification>

<success_criteria>
- `pytest tests/unit/test_wide_gain_augmentation.py tests/unit/test_room_ir_augmentation.py -x -q` exits 0
- `grep -c "class WideGainAugmentation\|class RoomIRAugmentation" src/acoustic/training/augmentation.py` returns 2
- `grep -c "pyroomacoustics" requirements.txt requirements-vertex.txt` returns 2 (one per file)
</success_criteria>

<output>
After completion, create `.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-01-SUMMARY.md`
</output>
