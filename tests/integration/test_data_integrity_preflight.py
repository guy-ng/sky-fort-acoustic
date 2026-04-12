"""Phase 22 Wave 0: data integrity preflight. Green after Plan 04."""
from pathlib import Path

DRONE_DIR = Path("data/field/drone")
BG_DIR = Path("data/field/background")


def test_all_field_files_exist_and_decode():
    import soundfile as sf
    drones = sorted(DRONE_DIR.glob("20260408_*.wav"))
    bgs = sorted(BG_DIR.glob("20260408_*.wav"))
    assert len(drones) == 13, f"expected 13 drone files, got {len(drones)}"
    assert len(bgs) == 4, f"expected 4 background files, got {len(bgs)}"
    for wav in drones + bgs:
        info = sf.info(str(wav))
        assert info.samplerate == 16000, f"{wav.name}: sr={info.samplerate}"
        assert info.channels == 1, f"{wav.name}: channels={info.channels}"


def test_preflight_script_imports_and_returns_manifest():
    from scripts.preflight_v8_data import preflight_field_recordings
    manifest = preflight_field_recordings(
        drone_dir=DRONE_DIR, bg_dir=BG_DIR,
        holdout_files={
            "20260408_091054_136dc5.wav",
            "20260408_092615_1a055f.wav",
            "20260408_091724_bb0ed8.wav",
            "20260408_084222_44dc5c.wav",
            "20260408_090757_1c50e9.wav",
        },
    )
    assert len(manifest["drone"]) == 9
    assert len(manifest["background"]) == 3


def test_trimmed_file_is_61_4s():
    import soundfile as sf
    info = sf.info(str(DRONE_DIR / "20260408_091054_136dc5.wav"))
    dur = info.frames / info.samplerate
    assert 60.0 < dur < 62.0, f"expected ~61.4s, got {dur:.2f}s"
