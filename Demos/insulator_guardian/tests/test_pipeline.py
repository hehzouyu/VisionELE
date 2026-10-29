"""集成管线单测。

口径：用 scene 的 12×8 m 巡检场地 + SimulatedDetector 跑 120 拍，验证
「感知 → 调度 → 决策 → 运动」整条链路真的在动：有记录、有缺陷事件、
每一步机器人都落在原始地图的可通行格里、延迟有采样、summary 字段齐全；
再验 detector.mode=sim 的配置分支，以及 ledger_path=None 时不写台账。
"""
from __future__ import annotations

import copy
import json

import pytest

from src.integration.coords import CameraModel
from src.integration.pipeline import InspectionPipeline, SimRobot
from src.integration.scene import build_inspection_field
from src.integration.sim_detector import SimulatedDetector

SUMMARY_KEYS = {
    "steps", "events", "defect_events", "tasks_completed", "tasks_pending",
    "replans", "scheduler_stats", "latency", "duration_s", "final_pose", "ledger",
}


@pytest.fixture()
def detector(scene, pipeline_cfg) -> SimulatedDetector:
    cfg = pipeline_cfg["detector"]
    return SimulatedDetector(
        marks=scene.marks,
        camera=CameraModel.from_dict(pipeline_cfg["camera"]),
        max_range_m=cfg["max_range_m"],
        fov_deg=cfg["fov_deg"],
        conf_jitter=cfg["conf_jitter"],
        seed=cfg["seed"],
    )


def test_pipeline_runs_without_collision(scene, decision_cfg, pipeline_cfg, detector, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    pipeline = InspectionPipeline.build(
        decision_cfg, pipeline_cfg, scene.gridmap,
        detector=detector,
        start_pose=scene.start_pose,
        inspect_points=scene.inspect_points,
        ledger_path=ledger,
    )
    for _ in range(120):
        pipeline.tick()

    assert len(pipeline.records) == 120
    assert any(record.task is not None for record in pipeline.records)
    for record in pipeline.records:
        x, y = record.pose[:2]
        assert scene.gridmap.is_free_world(x, y), f"第 {record.step} 拍撞墙: {record.pose}"

    latency = pipeline.latency()
    assert set(latency) >= {"detect_ms", "decision_ms", "cycle_ms"}
    for keys in latency.values():
        assert keys["n"] == 120
        assert keys["mean"] >= 0.0

    summary = pipeline.summary()
    assert SUMMARY_KEYS <= set(summary)
    assert summary["steps"] == 120
    assert summary["events"] > 0
    assert summary["defect_events"] > 0
    assert summary["scheduler_stats"]["events"] == 120
    assert scene.gridmap.is_free_world(*summary["final_pose"][:2])

    assert ledger.exists()
    lines = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    assert lines and all("type" in row for row in lines)


def test_build_accepts_sim_detector_mode(scene, decision_cfg, pipeline_cfg, detector, tmp_path):
    cfg = copy.deepcopy(pipeline_cfg)
    cfg["detector"]["mode"] = "sim"
    pipeline = InspectionPipeline.build(
        decision_cfg, cfg, scene.gridmap,
        detector=detector,
        start_pose=scene.start_pose,
        inspect_points=scene.inspect_points,
        ledger_path=tmp_path / "sim.jsonl",
    )
    assert isinstance(pipeline, InspectionPipeline)
    assert isinstance(pipeline.detector, SimulatedDetector)
    for _ in range(5):
        pipeline.tick()
    assert len(pipeline.records) == 5


def test_ledger_none_writes_nothing(scene, decision_cfg, pipeline_cfg, detector):
    pipeline = InspectionPipeline(
        scene.gridmap,
        detector=detector,
        robot=SimRobot(*scene.start_pose),
        inspect_points=scene.inspect_points,
        dt=pipeline_cfg["pipeline"]["dt"],
        ledger_path=None,
    )
    for _ in range(30):
        pipeline.tick()
    assert pipeline.ledger_path is None
    assert pipeline.summary()["ledger"] is None
