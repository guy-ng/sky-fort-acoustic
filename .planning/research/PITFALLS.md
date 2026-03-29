# Domain Pitfalls

**Domain:** Acoustic drone detection and tracking microservice (UMA-16v2, Docker, real-time)
**Researched:** 2026-03-29

---

## Critical Pitfalls

Mistakes that cause rewrites, unreliable detection, or architectural dead ends.

---

### Pitfall 1: USB Audio Device Instability in Docker

**What goes wrong:** The UMA-16v2 is a USB Audio Class 2 device streaming 16 channels at 48kHz. Docker does not natively support USB audio devices. Teams pass `/dev/snd` or use `--device /dev/bus/usb` but hit intermittent failures: the device disappears after a USB reset, the container sees the device but cannot open it, or ALSA/PulseAudio state on the host conflicts with the container. Restarting the container is sometimes the only fix. Using `--privileged` mode is tempting but creates security risks.

**Why it happens:** USB audio devices can re-enumerate (get a new device path) after power events, cable bumps, or host sleep/wake. Docker's `--device` flag maps a static path at container start. If the device re-enumerates to a different path, the container loses access silently. Additionally, the host's audio subsystem (PulseAudio/PipeWire) may grab the device exclusively.

**Consequences:** Complete service outage with no audio input. Silent failures where the service appears healthy but produces no detections. Intermittent issues that are hard to reproduce during development.

**Prevention:**
- Use `--device /dev/snd` combined with `--group-add audio` rather than `--privileged`
- Disable PulseAudio/PipeWire on the host for this device, or use ALSA directly (the POC already uses `hw:3,0` ALSA addressing)
- Implement a health-check loop that periodically verifies `sounddevice.query_devices()` returns the UMA-16v2 and that a short test capture succeeds
- Write a udev rule on the host to give the UMA-16v2 a stable symlink path (e.g., `/dev/uma16`) based on USB vendor/product ID
- Design the audio capture module to handle device-loss gracefully: detect, log, attempt reconnection, and publish a "no-audio" status event over ZeroMQ

**Detection:** Audio capture returns empty buffers or throws `sounddevice.PortAudioError`. No beamforming output despite service appearing healthy. Log messages about device not found.

**Phase relevance:** Phase 1 (audio capture). This must be solved first because every downstream component depends on reliable audio input.

**Confidence:** HIGH -- multiple Docker community reports confirm this pattern. The POC hardcodes `hw:3,0` which is fragile.

---

### Pitfall 2: Buffer Overflows and Xruns in 16-Channel Real-Time Capture

**What goes wrong:** At 48kHz x 16 channels x 32-bit float, the raw data rate is ~3 MB/s. If the processing pipeline (beamforming + CNN inference) cannot keep up, the audio ring buffer overflows. `sounddevice` reports this as input overflow status flags, but many implementations ignore these flags. The POC uses `sd.rec()` with `blocking=True`, which is the simplest but worst approach for real-time -- it blocks the thread and provides no way to handle backpressure.

**Why it happens:** The POC's `capture_chunk()` function calls `sd.rec(blocking=True)` which records a fixed chunk, processes it, then records again. Any processing time exceeding the chunk duration causes gaps between chunks (missed audio) rather than buffer overflow, but the result is the same: lost data. With a 150ms chunk (`CHUNK_SECONDS = 0.15`), you have exactly 150ms to process before the next chunk should start. Beamforming + CNN inference can easily exceed this on modest hardware.

**Consequences:** Dropped audio frames cause missed detections. Inconsistent timing breaks Doppler estimation (which depends on continuous, gapless audio). Under heavy load, the system silently degrades rather than failing obviously.

**Prevention:**
- Use `sounddevice.InputStream` with a callback function instead of `sd.rec()`. The callback runs in a separate PortAudio thread and deposits audio into a ring buffer that the processing thread reads from
- Size the ring buffer to hold 2-5 seconds of audio (a 16-ch float32 ring buffer at 48kHz costs ~6-15 MB, trivial)
- Monitor the `status` parameter in the callback for overflow flags and log/count them
- Decouple capture from processing: capture thread writes to a lock-free ring buffer; processing thread reads at its own pace. If processing falls behind, it skips frames rather than blocking capture
- Set `sounddevice` latency to `'high'` for the input stream to get larger OS-level buffers

**Detection:** `status.input_overflow` flag in the sounddevice callback. Growing ring buffer fill level. Processing time per chunk approaching or exceeding chunk duration.

**Phase relevance:** Phase 1 (audio capture architecture). The capture-to-processing pipeline architecture must be right from the start -- retrofitting a callback-based design onto blocking code is a rewrite.

**Confidence:** HIGH -- the POC code itself demonstrates the anti-pattern (`sd.rec(blocking=True)`).

---

### Pitfall 3: Spatial Aliasing from UMA-16v2's 42mm Grid Spacing

**What goes wrong:** The UMA-16v2 has 42mm microphone spacing. The spatial aliasing frequency is f_alias = c/(2d) = 343/(2 x 0.042) = approximately 4083 Hz. Above this frequency, the beamforming map produces ghost sources (false direction-of-arrival estimates) due to phase wrapping. The POC limits the band to 100-2000 Hz (well below aliasing), but a future developer might widen the band to "improve resolution" and introduce phantom targets.

**Why it happens:** Spatial aliasing is the acoustic equivalent of image aliasing -- when the microphone spacing is wider than half the wavelength, the array cannot distinguish the true direction from aliased directions. Higher frequencies give better angular resolution, so there is a natural temptation to use them. The 4x4 URA with only 16 elements also has inherently poor angular resolution at low frequencies (beamwidth is approximately lambda/D, where D = 3 x 0.042 = 0.126m aperture).

**Consequences:** False targets in the beamforming map. Incorrect bearing estimates sent via ZeroMQ to downstream systems. CNN classifier trained on aliased data learns spurious patterns.

**Prevention:**
- Hard-code the frequency band limits as constants with clear documentation explaining why (the POC's 100-2000 Hz is a good starting point)
- Add a configuration guard that rejects frequency ranges above 4000 Hz
- Document the array's angular resolution at the operating frequency: at 1000 Hz (lambda = 0.343m), beamwidth is roughly 0.343/0.126 = 2.7 radians (~155 degrees) -- this array has very poor low-frequency resolution. At 2000 Hz, beamwidth is ~77 degrees. Expect broad beams, not pinpoint accuracy
- When displaying the beamforming map, do not imply precision the array cannot deliver

**Detection:** Multiple strong peaks in the beamforming map at symmetric angles. Target bearing that jumps erratically between positions.

**Phase relevance:** Phase 2 (beamforming). The frequency band parameters and their rationale must be locked down when implementing the beamforming pipeline.

**Confidence:** HIGH -- physics-based, confirmed by the POC's own band limiting and multiple academic sources on microphone array design.

---

### Pitfall 4: CNN Overfitting to Recording Environment

**What goes wrong:** The CNN is trained on recordings made by the same UMA-16v2 in the same location. The model learns the acoustic signature of the environment (room/site reverberation, background noise profile, microphone frequency response) rather than drone-specific features. It achieves high validation accuracy because validation data comes from the same sessions, then fails in a new deployment location or when background noise changes (wind, traffic, rain).

**Why it happens:** In acoustic drone classification research, a significant limitation is that validation and training datasets derive from the same measurements. Even with train/test splits, the acoustic environment is a confounding variable. Data augmentation (pitch shifting, noise injection) helps but has limits -- excessive augmentation degrades performance as augmented samples introduce artifacts that do not match real-world conditions.

**Consequences:** Model appears to work in testing but produces high false-positive or false-negative rates in production. Retraining is needed at every new deployment site. Confidence scores are miscalibrated.

**Prevention:**
- Record training data across multiple sessions, times of day, and weather conditions
- Split data by recording session, not by random sample -- all samples from one session go to either train or test, never both
- Include deliberate "negative" recordings (no drone, only ambient noise) from many conditions
- Apply moderate augmentation only: background noise mixing at realistic SNR levels, small time shifts. Avoid excessive pitch shifting or harmonic distortion
- Design the training pipeline to report per-session accuracy, not just aggregate accuracy
- Plan for site-specific fine-tuning as a first-class workflow in the web UI

**Detection:** High training/validation accuracy but poor real-world precision/recall. Model confidence is uniformly high (even on noise-only inputs). Accuracy drops after weather changes.

**Phase relevance:** Phase 4 (CNN training pipeline). The recording metadata system (Phase 3) must support session-level labels and conditions to enable proper data splitting later.

**Confidence:** MEDIUM -- well-documented in academic literature on acoustic drone classification. Exact severity depends on deployment diversity.

---

### Pitfall 5: Python GIL Blocking the Processing Pipeline

**What goes wrong:** The service needs to simultaneously: capture audio (callback thread), run beamforming (CPU-intensive NumPy), run CNN inference (CPU-intensive), compute Doppler (CPU-intensive), serve the web UI (asyncio/HTTP), and publish ZeroMQ events. Python's GIL means only one thread executes Python bytecode at a time. If beamforming or CNN inference involves any pure-Python loops (not in NumPy/C extensions), it blocks everything else.

**Why it happens:** NumPy operations release the GIL, so pure-NumPy beamforming runs concurrently with other threads. But any Python-level loop around NumPy calls (e.g., iterating over microphone pairs in pure Python, as the POC does with `for (m, n) in pairs:`) holds the GIL between each NumPy call. The POC's SRP-PHAT iterates over C(16,2) = 120 microphone pairs in a Python loop -- each iteration does a small NumPy operation, but the loop overhead and GIL acquisition/release 120 times per chunk adds up.

**Consequences:** Audio capture callback is delayed, causing xruns. Web UI becomes unresponsive during heavy processing. ZeroMQ event publishing is delayed, making downstream systems see stale data.

**Prevention:**
- Vectorize the microphone pair loop: compute all 120 GCC-PHAT cross-correlations in a single batched NumPy operation rather than a Python for-loop
- Use `multiprocessing` for CPU-bound work: audio capture in the main process, beamforming + CNN in a worker process communicating via shared memory or `multiprocessing.Queue`
- Keep the web server (FastAPI/uvicorn) in the main process where it can respond quickly
- Profile early: measure per-chunk processing time versus chunk duration (150ms target) on the target hardware
- Consider Numba JIT compilation for any remaining Python-level DSP loops (acoular already uses Numba internally)
- Set `OPENBLAS_NUM_THREADS` and `NUMBA_NUM_THREADS` explicitly to avoid thread oversubscription (documented acoular issue #187)

**Detection:** Processing time per chunk exceeds chunk duration. Web UI latency spikes correlate with beamforming cycles. `htop` shows one core at 100% while others are idle.

**Phase relevance:** Phase 1-2 (architecture decision). The process/thread architecture must be designed upfront. Retrofitting multiprocessing onto a single-threaded design is painful.

**Confidence:** HIGH -- the POC code demonstrates the Python-loop-over-pairs pattern. GIL behavior with NumPy is well-documented.

---

## Moderate Pitfalls

---

### Pitfall 6: ZeroMQ PUB/SUB Slow Subscriber Causing Memory Explosion

**What goes wrong:** ZeroMQ PUB sockets queue messages for each connected subscriber. If a subscriber is slow or disconnected but the TCP connection persists, the publisher's send queue grows unboundedly until the process runs out of memory and crashes.

**Prevention:**
- Set `ZMQ_SNDHWM` (send high water mark) on the PUB socket to a reasonable limit (e.g., 100 messages). When the queue is full, ZeroMQ silently drops messages for slow subscribers rather than blocking or growing unboundedly
- For the periodic update stream (speed/bearing), use `ZMQ_CONFLATE` on subscribers so they only keep the latest message, discarding stale updates
- Monitor queue depth if possible, or at minimum set a reasonable HWM from day one
- Document the expected message rate so downstream teams can size their subscribers appropriately

**Phase relevance:** Phase 3 (ZeroMQ event publishing). Set HWM at socket creation time.

**Confidence:** HIGH -- well-documented ZeroMQ pattern (the "slow subscriber" problem).

---

### Pitfall 7: Doppler Estimation Confusion from Multi-Rotor Harmonics

**What goes wrong:** Consumer drones have 4+ rotors each producing fundamental frequency plus harmonics. The Doppler shift from the drone's movement is small relative to the rotor frequencies. Teams attempt to estimate drone speed from frequency shifts but instead measure rotor RPM changes (throttle adjustments) or confuse harmonics of different rotors.

**Prevention:**
- Clearly separate two measurements: rotor RPM estimation (from harmonic spacing in the spectrum) versus translational Doppler shift (from bulk frequency shift of the entire harmonic comb)
- Use the entire harmonic comb for Doppler estimation rather than a single frequency peak -- the translational Doppler shifts all harmonics by the same ratio
- At typical drone speeds (5-20 m/s) and acoustic frequencies (200-2000 Hz), the Doppler shift is only 0.3-12 Hz -- this requires high frequency resolution (long FFT windows), which conflicts with real-time latency requirements
- Accept that acoustic Doppler speed estimation has limited accuracy. Publish confidence bounds, not just point estimates
- Consider using frame-to-frame bearing change rate as an alternative speed proxy (does not require Doppler at all)

**Phase relevance:** Phase 3 (Doppler estimation). This is inherently the hardest part of the pipeline and should be treated as "best effort" rather than a hard requirement.

**Confidence:** MEDIUM -- based on signal processing fundamentals and radar micro-Doppler literature adapted to acoustics. Acoustic Doppler for drones is less studied than radar Doppler.

---

### Pitfall 8: Beamforming Map Serialization Bottleneck for Web UI

**What goes wrong:** The beamforming map is a 2D array (e.g., 61x61 grid of SRP-PHAT values) that needs to be sent to the React web UI multiple times per second. Teams serialize it as JSON (array of arrays of floats), which is both slow to encode and large on the wire. At 10 FPS with a 61x61 grid, that is ~150KB/s of JSON.

**Prevention:**
- Send the beamforming map as a binary blob (Float32Array) over WebSocket, not as JSON
- Use a fixed grid size and document the layout so the frontend can decode without metadata per frame
- Downsample the grid for display (the angular resolution of the array does not justify fine grids anyway -- see Pitfall 3)
- Implement server-side throttling: only send a new frame when the previous one has been acknowledged or after a minimum interval
- Consider sending only the peak location and power level for a lightweight "radar" display, with the full heatmap as an optional high-bandwidth mode

**Phase relevance:** Phase 5 (web UI with live beamforming map). Design the WebSocket protocol early.

**Confidence:** MEDIUM -- standard web real-time data streaming concern, not specific to published issues.

---

### Pitfall 9: Recording File Format and Metadata Loss

**What goes wrong:** Teams store multi-channel recordings as raw WAV files and keep metadata (labels, conditions, drone type) in filenames or separate text files. The metadata gets separated from the recordings, filenames become unmanageable, and the training pipeline cannot reliably associate recordings with their labels.

**Prevention:**
- Use a structured recording format: HDF5 (with `pytables`, already an acoular dependency) or a dedicated SQLite database linking recording files to metadata
- Store metadata atomically with the recording: session ID, timestamp, channel count, sample rate, operator-provided labels, environmental conditions
- Design the metadata schema before implementing recording -- the training pipeline consumes this schema
- Include recording of the "pipeline state" (was the drone detected? what was the bearing?) as ground truth for training

**Phase relevance:** Phase 3 (recording/playback). Metadata schema decisions affect the training pipeline in Phase 4.

**Confidence:** HIGH -- standard data management problem. HDF5/pytables is a natural fit given acoular already depends on it.

---

### Pitfall 10: Single Docker Container Becoming Unmanageable

**What goes wrong:** Putting the Python backend and React frontend in one container simplifies deployment initially but creates problems: the build takes forever (Node.js + Python dependencies), the image is huge, the frontend dev workflow requires rebuilding the entire container, and there is no way to restart the frontend independently of the backend.

**Prevention:**
- Use a multi-stage Dockerfile: build the React app in a Node.js stage, copy the static build artifacts into the Python stage. The final image has no Node.js runtime
- Serve the React static files from the Python web server (FastAPI with `StaticFiles` mount) rather than running a separate dev server
- During development, run the React dev server outside Docker (connecting to the Python backend via CORS) for fast iteration. Only build the combined image for deployment
- Keep the Dockerfile layers ordered by change frequency: OS deps, Python deps, Python code, React build. This maximizes layer caching

**Phase relevance:** Phase 1 (Docker setup) and Phase 5 (web UI integration). Get the multi-stage Dockerfile right in Phase 1, even before the React app exists, so that adding the frontend later is a small change.

**Confidence:** HIGH -- standard Docker best practice.

---

## Minor Pitfalls

---

### Pitfall 11: Numba/OpenBLAS Thread Oversubscription

**What goes wrong:** Acoular uses Numba internally. Numba and OpenBLAS (used by NumPy) each spawn their own thread pools, defaulting to the number of CPU cores. In a Docker container with CPU limits, they detect the host's core count, not the container's, and spawn too many threads. This causes excessive context switching and slower performance than single-threaded execution.

**Prevention:**
- Set environment variables in the Dockerfile: `OPENBLAS_NUM_THREADS=2`, `NUMBA_NUM_THREADS=2`, `MKL_NUM_THREADS=2`
- Tune based on the container's CPU allocation, not the host
- This is documented as acoular GitHub issue #187

**Phase relevance:** Phase 1 (Docker setup). Set these in the Dockerfile from the start.

**Confidence:** HIGH -- documented acoular issue.

---

### Pitfall 12: Microphone Channel Mapping Errors

**What goes wrong:** The UMA-16v2's USB channel order does not match the physical microphone layout. The POC has a hardcoded mapping (`mic_rc` dictionary in `radar_gui_all_mics_fast_drone.py`) that maps MIC numbers to row/column positions. If this mapping is wrong, beamforming points in the wrong direction. The POC comment says "we assume" the mapping -- it may not have been rigorously verified.

**Prevention:**
- Verify the channel-to-position mapping empirically: tap each microphone with a pencil while monitoring individual channel levels. Document which USB channel corresponds to which physical position
- Write a calibration test that can be run on service startup or on demand
- Store the mic geometry as a configuration file, not hardcoded, so it can be corrected without code changes

**Phase relevance:** Phase 1 (audio capture) or Phase 2 (beamforming setup). Verify before any beamforming work begins.

**Confidence:** MEDIUM -- the POC mapping exists and likely works, but the "we assume" comment flags uncertainty.

---

### Pitfall 13: WebSocket Connection Leaks in React UI

**What goes wrong:** React components that open WebSocket connections without proper cleanup in `useEffect` return functions leak connections on re-render, navigation, or hot-module replacement during development. Each leaked connection consumes server resources and memory.

**Prevention:**
- Always close WebSocket connections in the `useEffect` cleanup function
- Use a single WebSocket connection managed by a context provider or state manager, not per-component connections
- Implement reconnection logic with exponential backoff
- During development, verify no connection leaks by monitoring WebSocket count in browser DevTools

**Phase relevance:** Phase 5 (web UI). Standard React practice but easy to miss.

**Confidence:** HIGH -- well-documented React pattern.

---

### Pitfall 14: Wind Noise Saturation in Outdoor Deployment

**What goes wrong:** Wind generates broadband low-frequency noise that can be 5-7 dB above the drone signature. The MEMS microphones on the UMA-16v2 can saturate in strong wind gusts (acoustic overload at 120 dB SPL). When saturated, the signal clips and beamforming produces garbage.

**Prevention:**
- Use a windscreen/foam cover over the array for outdoor deployment
- Implement a wind-noise detector: monitor low-frequency energy (below 100 Hz) and flag frames where it exceeds a threshold
- When wind noise is detected, either discard the frame or apply more aggressive high-pass filtering
- Publish wind-noise status in ZeroMQ events so downstream systems know detection reliability is degraded

**Phase relevance:** Phase 2-3 (beamforming + detection pipeline). Not critical for indoor development but essential for production outdoor use.

**Confidence:** MEDIUM -- based on outdoor microphone array research. The exact impact depends on deployment conditions.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|---|---|---|
| Phase 1: Audio capture + Docker | USB device instability (P1), buffer overflow (P2), thread oversubscription (P11) | udev rules, callback-based capture, env vars in Dockerfile |
| Phase 2: Beamforming pipeline | Spatial aliasing (P3), GIL blocking (P5), channel mapping errors (P12) | Hard frequency limits, vectorized operations, empirical calibration |
| Phase 3: Detection + ZeroMQ + recording | Slow subscriber (P6), Doppler confusion (P7), metadata loss (P9) | HWM settings, harmonic comb approach, HDF5 with embedded metadata |
| Phase 4: CNN training | Environment overfitting (P4) | Session-level splits, diverse recording conditions, moderate augmentation |
| Phase 5: Web UI | Serialization bottleneck (P8), WebSocket leaks (P13) | Binary WebSocket protocol, proper useEffect cleanup |
| Phase 6: Docker packaging | Unmanageable container (P10) | Multi-stage Dockerfile from Phase 1, static file serving |
| Deployment | Wind noise (P14), USB re-enumeration (P1) | Windscreen, health checks, stable device paths |

---

## Sources

- [miniDSP UMA-16 v2 Product Page](https://www.minidsp.com/products/usb-audio-interface/uma-16-microphone-array) -- array specifications
- [Acoular Thread Overloading Issue #187](https://github.com/acoular/acoular/issues/187) -- Numba/OpenBLAS thread contention
- [python-sounddevice Buffer Overflow Issue #155](https://github.com/spatialaudio/python-sounddevice/issues/155) -- xrun behavior documentation
- [ZeroMQ Guide Chapter 5: Advanced Pub-Sub](https://zguide.zeromq.org/docs/chapter5/) -- slow subscriber patterns
- [Docker Microphone Passthrough Notes](https://github.com/jwansek/DockerMicrophonePassthroughNotes) -- USB audio in Docker
- [The Sound of Surveillance: Enhancing ML Drone Detection with Acoustic Augmentation](https://www.mdpi.com/2504-446X/8/3/105) -- augmentation pitfalls
- [Audio-Based Drone Detection Using Deep Learning with GAN Enhancement](https://pmc.ncbi.nlm.nih.gov/articles/PMC8348319/) -- overfitting in drone classification
- [Robin Radar: Pros and Cons of Acoustic Drone Detection](https://www.robinradar.com/blog/acoustic-sensors-drone-detection) -- range and wind limitations
- [GFAITech: Frequency Range of Microphone Arrays](https://www.gfaitech.com/knowledge/faq/frequency-range-microphone-array) -- spatial aliasing theory
- [Outdoor Microphone Range Tests and Spectral Analysis of UAV Acoustic Signatures](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656299/) -- wind noise impact on outdoor arrays
- POC source: `POC-code/PT520/PTZ/radar_gui_all_mics_fast_drone.py` -- existing beamforming implementation patterns
