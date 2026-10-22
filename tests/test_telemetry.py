"""Check durable, isolated match logs without touching the real log folder."""

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from adaptive_boss import telemetry
from adaptive_boss.telemetry import SessionRecorder


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_metadata_is_flushed_immediately_and_uses_utc(tmp_path):
    recorder = SessionRecorder(tmp_path / "nested" / "logs", "adaptive", 7)
    try:
        records = read_records(recorder.path)
        assert len(records) == 1
        start = records[0]
        assert start["type"] == "session_start"
        assert start["schema_version"] == 3
        assert start["mode"] == "adaptive"
        assert start["seed"] == 7
        assert start["input_source"] == "keyboard"
        assert start["boss_attack"] is None
        assert datetime.fromisoformat(start["started_at"]).utcoffset() == timedelta(0)
        assert set(start) == {"type", "schema_version", "mode", "seed", "started_at",
                              "input_source", "boss_attack"}
        assert recorder.path.suffix == ".jsonl"
    finally:
        recorder.close()


def test_automated_runs_are_identifiable_for_experiment_filtering(tmp_path):
    recorder = SessionRecorder(tmp_path, "adaptive", 7, input_source="smoke_test")
    recorder.close()
    assert read_records(recorder.path)[0]["input_source"] == "smoke_test"


def test_filters_events_and_preserves_original_fields_and_chinese(tmp_path, capsys):
    recorder = SessionRecorder(str(tmp_path), "fixed", 17)
    decision = {
        "type": "decision", "time": 1.0, "attack": "锁定轰击",
        "prediction": "cw", "context": [0.2, 0.4], "using_prediction": False,
        "probabilities": {"cw": 0.4, "stay": 0.6},
    }
    observation = {"type": "observation", "actual": "ccw", "correct": False}
    ended = {"type": "end", "result": "win", "samples": 25}
    attack = {"type": "attack", "charge": 0.8, "damage": 20}
    dash = {"type": "dash", "move_x": 0.7, "move_y": 0.7}
    player_hit = {"type": "player_hit", "damage": 16}
    boss_hit = {"type": "boss_hit", "damage": 20}
    boss_block = {"type": "boss_block", "x": 430.0, "y": 260.0, "charge": 0.2}
    phase = {"type": "phase", "phase": "lock", "attack_id": 2}
    accepted = [attack, decision, dash, player_hit, observation, boss_hit, boss_block, phase, ended]
    try:
        for event in [*accepted, {"unknown": 1}, {"type": "pointer_motion"}]:
            recorder.record(event)
        records = read_records(recorder.path)
        assert records[1:] == accepted
        assert "锁定轰击" in recorder.path.read_text(encoding="utf-8")
        assert records[2] is not decision
        assert decision["time"] == 1.0
    finally:
        recorder.close()
    assert capsys.readouterr().out == ""


def test_prediction_correctness_and_real_counter_hits_remain_distinct(tmp_path):
    recorder = SessionRecorder(tmp_path, "adaptive", 7)
    decision = {
        "type": "decision", "attack_id": 4, "prediction": "cw",
        "using_prediction": True, "confidence": 0.7,
    }
    observation = {
        "type": "observation", "attack_id": 4, "actual": "cw", "correct": True,
    }
    player_hit = {
        "type": "player_hit", "attack_id": 4, "hazard_id": "main_blast",
        "is_counter": False, "using_prediction": True, "damage": 16,
    }
    ended = {"type": "end", "counter_attempts": 1, "counter_hits": 0}
    try:
        for event in (decision, observation, player_hit, ended):
            recorder.record(event)
        records = read_records(recorder.path)
        assert records[1:] == [decision, observation, player_hit, ended]
        assert records[2]["correct"] is True
        assert records[3]["is_counter"] is False
        assert records[4]["counter_hits"] == 0
    finally:
        recorder.close()


def test_independent_sessions_never_overwrite_even_with_same_start_time(tmp_path, monkeypatch):
    same_time = datetime(2026, 10, 19, 12, 34, 56, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(telemetry, "_utc_now", lambda: same_time)
    first = SessionRecorder(tmp_path, "fixed", 7)
    second = SessionRecorder(tmp_path, "adaptive", 7)
    try:
        assert first.path != second.path
        assert "20261019T123456.123456Z" in first.path.name
        first.record({"type": "end", "result": "win"})
        second.record({"type": "end", "result": "lose"})
        assert read_records(first.path)[1]["result"] == "win"
        assert read_records(second.path)[1]["result"] == "lose"
        assert len(list(tmp_path.glob("*.jsonl"))) == 2
    finally:
        first.close()
        second.close()


def test_exclusive_open_retries_a_filename_collision(tmp_path, monkeypatch):
    same_time = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(telemetry, "_utc_now", lambda: same_time)
    suffixes = iter(("aaaaaaaa", "aaaaaaaa", "bbbbbbbb"))
    monkeypatch.setattr(telemetry, "uuid4", lambda: SimpleNamespace(hex=next(suffixes)))
    first = SessionRecorder(tmp_path, "fixed", 7)
    original = first.path.read_bytes()
    second = SessionRecorder(tmp_path, "fixed", 7)
    try:
        assert first.path != second.path
        assert first.path.read_bytes() == original
        assert second.path.name.endswith("bbbbbbbb.jsonl")
    finally:
        first.close()
        second.close()


def test_close_is_idempotent_and_summary_is_kept_separately(tmp_path):
    recorder = SessionRecorder(tmp_path, "adaptive", 7)
    summary = {"type": "custom", "result": "win", "nested": {"命中": 12}}
    recorder.close(summary)
    first_bytes = recorder.path.read_bytes()
    recorder.close({"result": "lose"})
    assert recorder.path.read_bytes() == first_bytes
    records = read_records(recorder.path)
    assert [event["type"] for event in records] == ["session_start", "session_end"]
    assert records[-1]["summary"] == summary
    assert datetime.fromisoformat(records[-1]["ended_at"]).utcoffset() == timedelta(0)
    assert recorder._stream.closed
    with pytest.raises(ValueError, match="closed"):
        recorder.record({"type": "end"})


def test_omitted_summary_still_writes_session_end(tmp_path):
    recorder = SessionRecorder(tmp_path, "fixed", 3)
    recorder.close()
    assert read_records(recorder.path)[-1]["summary"] is None


def test_bad_json_value_cannot_leave_a_partial_line(tmp_path):
    recorder = SessionRecorder(tmp_path, "fixed", 3)
    original = recorder.path.read_bytes()
    try:
        with pytest.raises(ValueError):
            recorder.record({"type": "observation", "confidence": float("nan")})
        assert recorder.path.read_bytes() == original
    finally:
        recorder.close()


def test_filesystem_failure_reaches_caller(tmp_path):
    blocker = tmp_path / "file_instead_of_folder"
    blocker.write_text("existing content", encoding="utf-8")
    with pytest.raises(OSError):
        SessionRecorder(blocker, "adaptive", 7)
    assert blocker.read_text(encoding="utf-8") == "existing content"


def test_close_releases_handle_and_stays_idempotent_after_write_failure(tmp_path):
    recorder = SessionRecorder(tmp_path, "adaptive", 7)
    recorder._stream.close()

    class BrokenStream:
        closed = False

        def write(self, line):
            raise OSError("simulated disk failure")

        def close(self):
            self.closed = True

    broken = BrokenStream()
    recorder._stream = broken
    with pytest.raises(OSError, match="disk failure"):
        recorder.close()
    assert broken.closed
    recorder.close()
