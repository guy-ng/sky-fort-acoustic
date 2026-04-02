# Phase 10: Field Data Collection - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 10-field-data-collection
**Areas discussed:** Recording session flow, Label taxonomy & directory layout, Metadata schema, Audio capture format

---

## Recording Session Flow

### Q1: How should users initiate a recording session?

| Option | Description | Selected |
|--------|-------------|----------|
| Label-first | User selects label and conditions BEFORE hitting record. Ensures every recording is labeled from the start. | |
| Record-first, label later | User hits record immediately, then labels after stopping. Faster to capture unexpected events. | ✓ |
| Quick-record with defaults | One-click record with default label, edit metadata afterward. | |

**User's choice:** Record-first, label later
**Notes:** Prioritizes capture speed in field conditions.

### Q2: Max recording duration?

| Option | Description | Selected |
|--------|-------------|----------|
| Unlimited | Record until user clicks stop. Simple, flexible. | |
| Configurable max | Set a max duration with auto-stop. Prevents huge files. | ✓ |
| Fixed short clips | Fixed-length clips (e.g., 10s or 30s) with auto-stop. | |

**User's choice:** Configurable max
**Notes:** None

### Q3: Live recording indicator?

| Option | Description | Selected |
|--------|-------------|----------|
| Timer + levels | Elapsed time, remaining time, audio level meter. | ✓ |
| Timer only | Elapsed time and pulsing record indicator. | |
| You decide | Claude chooses based on existing patterns. | |

**User's choice:** Timer + levels (Recommended)
**Notes:** None

### Q4: What happens after stopping?

| Option | Description | Selected |
|--------|-------------|----------|
| Inline label form | Form appears in UI to label and tag. Recording saves to unlabeled until labeled. | ✓ |
| Modal dialog | Modal pops up for label and metadata before finalization. | |
| Auto-save, edit from list | Saves immediately as unlabeled, label later from list. | |

**User's choice:** Inline label form (Recommended)
**Notes:** None

---

## Label Taxonomy & Directory Layout

### Q1: What label categories?

| Option | Description | Selected |
|--------|-------------|----------|
| Binary: drone/background | Matches current binary CNN. Simple. | |
| Multi-class by drone type | Separate labels per drone type plus background. | |
| Hierarchical: label + sub-label | Top-level (drone/background/other) for directory, sub-label in metadata. | ✓ |

**User's choice:** Hierarchical: label + sub-label
**Notes:** Best of both worlds -- structure for training, detail for analysis.

### Q2: Fixed or user-definable top-level labels?

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed preset | Hardcoded: drone, background, other. Prevents fragmentation. | ✓ |
| User-definable | Users can create new top-level labels from UI. | |
| You decide | Claude picks based on training pipeline compatibility. | |

**User's choice:** Fixed preset (Recommended)
**Notes:** None

### Q3: Directory location?

| Option | Description | Selected |
|--------|-------------|----------|
| Separate: data/recordings/ | New directory tree separate from research data. | |
| Same tree: audio-data/data/ | Add recordings into existing tree alongside research data. | |
| You decide | Claude picks based on collect_wav_files() and TrainingConfig. | ✓ |

**User's choice:** You decide
**Notes:** Claude's discretion -- key requirement is training pipeline compatibility.

---

## Metadata Schema

### Q1: What metadata fields?

| Option | Description | Selected |
|--------|-------------|----------|
| Distance estimate | Approximate distance of drone from array. | ✓ |
| Altitude estimate | Approximate altitude (low/medium/high or meters). | ✓ |
| Weather/conditions | Wind, rain, ambient noise level. | ✓ |
| Free-text notes | Open notes field for anything else. | ✓ |

**User's choice:** All four fields selected
**Notes:** None

### Q2: Metadata storage format?

| Option | Description | Selected |
|--------|-------------|----------|
| Sidecar JSON | Matching .json file per WAV. Simple, portable. | ✓ |
| Central JSON index | Single index.jsonl file listing all recordings. | |
| SQLite database | Local SQLite DB for metadata. Enables queries. | |

**User's choice:** Sidecar JSON (Recommended)
**Notes:** None

### Q3: Required fields?

| Option | Description | Selected |
|--------|-------------|----------|
| Label required only | Top-level label required, all else optional. | ✓ |
| Label + sub-label required | Both required for complete dataset. | |
| All optional | Everything optional, saves to unlabeled if no label. | |

**User's choice:** Label required only (Recommended)
**Notes:** None

---

## Audio Capture Format

### Q1: Channel count?

| Option | Description | Selected |
|--------|-------------|----------|
| Mono downmix | Average 16 channels to mono. Matches training format. ~16x less storage. | ✓ |
| All 16 channels | Full 16-channel WAV. Preserves spatial info. | |
| Both: 16-ch + mono | Full archive plus mono downmix. | |

**User's choice:** Mono downmix (Recommended)
**Notes:** None

### Q2: Sample rate?

| Option | Description | Selected |
|--------|-------------|----------|
| 16kHz direct | Downsample to match MelConfig SR=16000. Training-ready. | |
| 48kHz native | Keep native rate. ResearchPreprocessor resamples on-the-fly. | |
| You decide | Claude picks based on preprocessing pipeline behavior. | ✓ |

**User's choice:** You decide
**Notes:** Claude's discretion based on ResearchPreprocessor capabilities.

---

## Claude's Discretion

- Directory location for field recordings (separate tree vs existing audio-data/data/)
- Sample rate for saved recordings (16kHz vs 48kHz)
- Recording file naming convention
- Recording module code location
- Audio level meter implementation
- Temporary/unlabeled recording management

## Deferred Ideas

None -- discussion stayed within phase scope
