# Feature Landscape

**Domain:** Acoustic drone detection and tracking microservice
**Researched:** 2026-03-29

## Table Stakes

Features users expect from an acoustic drone detection system. Missing any of these and the product fails its core mission.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Real-time 16-ch audio capture | Without continuous capture, nothing works. Foundation of entire pipeline. | Low | POC already uses `sounddevice` at 48kHz. Well-understood. |
| Beamforming spatial map | Core detection mechanism. Every commercial system (Discovair G2+, ALARM) produces a spatial sound map to localize sources. Industry standard. | Med | POC uses `acoular` Delay-and-Sum. 4x4 URA, 42mm spacing, ~105deg FoV. |
| Peak detection with azimuth/elevation | Operators need bearing to target. All commercial systems output DOA (Direction of Arrival). Without this, beamforming map is just a pretty picture. | Med | Convert beamforming peak (x,y on grid) to pan/tilt degrees. POC already does this. |
| Noise gate / adaptive threshold | Urban and field environments are noisy. Without a noise floor calibration, system triggers on wind, vehicles, birds constantly. Every production system filters background. | Med | POC has noise calibration (percentile-based, configurable margin). Must carry forward. |
| CNN drone/not-drone binary classification | Beamforming alone cannot distinguish drone from other sound sources. ML classification is table stakes per Fraunhofer IDMT, Squarehead, DSIAC reviews. Mel-spectrogram + CNN is the proven approach. | High | POC uses TensorFlow/Keras with mel-spectrogram. 2s segments, enter/exit thresholds with hysteresis. |
| ZeroMQ event publishing (detection + updates) | This service exists to feed downstream consumers. Without pub/sub events, it is isolated and useless to the sky-fort system. | Med | PUB/SUB pattern. Initial detection event (target ID, class) + periodic updates (speed, bearing). |
| Target ID assignment and lifecycle | Operators need to track "which drone" across time. Without persistent IDs, every frame is a disconnected detection. | Med | Assign UUID on first detection, maintain until target lost (timeout). POC has TARGET_LOSS_TIMEOUT_S. |
| REST API for beamforming map | External consumers (dashboard, other services) need to query current state. Standard integration pattern. | Low | Serve current beamforming frame as image or JSON grid. |
| Web UI: live beamforming heatmap | Operators need visual confirmation of what the system "hears." Every commercial system (AirGuard, AARTOS, Discovair) has a spatial visualization. | Med | WebSocket stream of beamforming data rendered as heatmap in React. |
| Web UI: active target overlay | Displaying detected targets with class, speed, bearing on the heatmap. Without this, operator cannot assess the situation. | Med | Overlay markers on heatmap with tooltip/sidebar showing target details. |
| Docker deployment | Project requirement. Service must be self-contained and deployable. | Low | Single container with USB passthrough for UMA-16v2. |

## Differentiators

Features that set this system apart. Not expected in every acoustic detector, but add significant value.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Multi-class drone type identification | Most systems do binary drone/not-drone. Classifying drone type (5-inch racer vs DJI Mavic vs heavy-lift FlyCart) is cutting-edge research. Your dataset already has 7+ drone types plus helicopter/airplane/vehicle. | High | Existing data categories: 5inch, 10inch, 13inch, EvoMax, FlyCart, Matrice, Mavic. Multi-class CNN instead of binary. Major competitive advantage. |
| Doppler-based speed estimation | Few acoustic-only systems estimate target speed. Adds tactical value -- approaching vs hovering vs departing changes threat assessment. | High | Requires tracking frequency shifts of BPF harmonics over time. Novel feature for single-array system. |
| Recording with full pipeline replay | Record raw 16-ch audio, then replay through the exact same processing pipeline. This turns field recordings into reproducible test cases and training data simultaneously. | Med | Record to multi-channel WAV. Playback injects audio into pipeline as if live. Existing data uses JSONL index with metadata. |
| Metadata tagging on recordings | Attach labels (drone type, conditions, notes) to recordings. Creates structured training datasets from field operations. | Low | Build on existing JSONL schema: label, drone_type, notes, session info. |
| In-service CNN training pipeline | Most detection systems ship a frozen model. Allowing operators to train/retrain from labeled recordings within the service itself means the model improves with deployment-specific data. | High | Requires training UI, dataset management, model versioning, validation split. |
| Hysteresis-based detection state machine | Enter threshold (0.80) vs exit threshold (0.40) with confirmation hits prevents flickering detections. More robust than simple threshold. | Low | POC already implements this (SOUND_CNN_ENTER/EXIT, CONFIRM_HITS/CLEAR_HITS). Carry forward. |
| Configurable frequency band focus | Different drones have different BPF ranges. Allowing operators to tune beamforming frequency center and bandwidth improves detection for specific threat profiles. | Low | POC exposes FREQ_CENTER (default 4000Hz), FREQ_BANDS. Make this runtime-adjustable via UI. |
| Background noise class in training data | Having an explicit "background" class (not just "not drone") makes the classifier more robust. Your data already has this category. | Low | Three-class minimum: drone, background, other. Already structured in data/. |

## Anti-Features

Features to deliberately NOT build. These are tempting but wrong for this service.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| PTZ camera control | Out of scope per PROJECT.md. Separate service responsibility. Mixing acoustic processing with camera control created the monolithic mess in the POC. | Publish target bearing over ZeroMQ. Let the PTZ service subscribe and act. |
| Visual/YOLO drone detection | This is an acoustic-only service. Fusing visual detection adds complexity, different hardware dependencies, and blurs the service boundary. | Keep YOLO in its own service. Downstream fusion service can combine acoustic + visual tracks. |
| Multi-array support | UMA-16v2 only for v1. Supporting multiple arrays means distributed beamforming, clock sync, network audio -- massive complexity for unclear v1 value. | Design clean interfaces so multi-array could be added later, but do not build it now. |
| Range estimation from single array | A single planar array cannot reliably estimate distance (only direction). Attempting range estimation produces unreliable results that erode operator trust. | Report bearing only. Range requires triangulation from multiple sensors (different service). |
| RF-based drone detection fusion | Some systems fuse RF scanning with acoustic. This adds hardware, protocol complexity, and is a different detection modality entirely. | Keep service acoustic-pure. Fusion happens at the C2 layer, not inside the sensor service. |
| Real-time model hot-swap | Swapping CNN models while the system is live processing audio is risky (memory spikes, brief detection gaps). | Train offline. Restart service with new model. Or at minimum, pause detection briefly during swap. |
| Automatic model retraining | Auto-retraining on new data without human review risks model degradation from mislabeled data. | Require explicit operator action to trigger training. Human reviews labels before training. |
| Audio playback through speakers | Playing back drone sounds through speakers is not useful for operators and creates feedback loops with the mic array. | Playback means re-processing through the pipeline, not audio output. Visual-only replay. |

## Feature Dependencies

```
Audio Capture ──> Beamforming ──> Peak Detection ──> Target ID Assignment
                      |                |                     |
                      v                v                     v
               Beamforming Map    Azimuth/Elevation    ZeroMQ Events
               (REST API)             |                     ^
                      |                v                     |
                      v          CNN Classification ────────┘
               Web UI Heatmap         |
                      ^                v
                      |          Doppler Speed ────────> ZeroMQ Updates
                      |
               Target Overlay

Audio Capture ──> Recording ──> Metadata Tagging ──> Training Pipeline
                      |
                      v
                  Playback ──> Pipeline Replay ──> (feeds back to Beamforming)
```

**Critical path:** Audio Capture > Beamforming > Peak Detection > CNN Classification > ZeroMQ Events.
Everything else hangs off this spine.

**Recording branch** is independent of live detection but shares the audio capture component.

**Training pipeline** depends on labeled recordings existing first.

## MVP Recommendation

### Phase 1: Detection Core (must ship first)
1. **Real-time 16-ch audio capture** -- foundation, everything depends on this
2. **Beamforming spatial map** -- core detection mechanism
3. **Peak detection with azimuth/elevation** -- actionable output
4. **Noise gate / adaptive threshold** -- without this, false alarms make system unusable
5. **CNN drone/not-drone classification** -- binary first, multi-class later
6. **Target ID assignment** -- persistent tracking across frames
7. **ZeroMQ event publishing** -- downstream integration, the whole point of the service
8. **REST API for beamforming map** -- simple endpoint, low effort
9. **Docker deployment** -- ship it containerized from day one

### Phase 2: Operator Interface
1. **Web UI: live beamforming heatmap** -- visual monitoring
2. **Web UI: active target overlay** -- operational awareness
3. **Hysteresis detection state machine** -- polish detection quality

### Phase 3: Data Collection
1. **Recording raw 16-ch audio** -- start collecting field data
2. **Metadata tagging** -- label recordings for training
3. **Playback with pipeline replay** -- validate recordings work through pipeline

### Phase 4: Model Evolution
1. **Multi-class drone type identification** -- upgrade CNN from binary to multi-class
2. **CNN training pipeline in UI** -- operators can retrain from their own data
3. **Configurable frequency band** -- tune for deployment-specific threats

### Defer Indefinitely
- **Doppler speed estimation**: High complexity, uncertain accuracy with single planar array at 42mm spacing. Research this during Phase 1 to validate feasibility before committing. The UMA-16v2's small aperture may not provide sufficient frequency resolution for reliable Doppler extraction. Flag for feasibility research.

## Sources

- [Squarehead Discovair G2+ drone detection](https://www.sqhead.com/drone-detection) -- commercial system features and specs (128-mic array, 105deg FoV, ML classification)
- [Fraunhofer IDMT acoustic drone detection](https://www.idmt.fraunhofer.de/en/Press_and_Media/press_releases/2025/acoustic-drone-detection.html) -- research institute validation of acoustic approach
- [DSIAC primer on acoustic drone detection](https://dsiac.dtic.mil/primers/what-is-an-acoustic-drone-detection-system/) -- US DoD overview of capabilities and limitations
- [Classification, positioning, and tracking via HMM + circular mic array beamforming](https://link.springer.com/article/10.1186/s13638-019-1632-9) -- academic reference for beamforming + classification pipeline
- [Deep Learning-based drone acoustic event detection for mic arrays](https://link.springer.com/article/10.1007/s11042-023-17477-1) -- CNN + mel-spectrogram approach validation
- [Multiclass acoustic dataset for drone signatures](https://arxiv.org/html/2509.04715v1) -- 32 drone types, MFCC/mel-spectrogram features
- [AUDRON framework for drone type recognition](https://arxiv.org/html/2512.20407) -- multi-feature fusion for classification
- [Army seeks acoustic detection for counter-UAS](https://defensescoop.com/2026/01/21/army-counter-drone-small-uas-acoustic-detection-systems/) -- 2026 operational requirements
- [Robin Radar: pros and cons of acoustic detection](https://www.robinradar.com/blog/acoustic-sensors-drone-detection) -- honest assessment of limitations
- [AirGuard drone detection software](https://www.airsight.com/airguard-drone-detection-software) -- commercial dashboard/UI patterns
- [AARTOS drone detection system](https://drone-detection-system.com/aartos-dds/software-integration/) -- commercial system with heatmap visualization
- [MicrodB ALARM acoustic detection](https://microdb.fr/en/product/environmental-monitoring/) -- false alarm reduction in urban environments
- Existing POC code analysis: `POC_Recorder.py`, `unified_drone_collection_web_gui.py`, `poc_config.py`, `data/index.jsonl`
