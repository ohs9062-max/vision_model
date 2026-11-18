# 버전 경로와 모델·후처리 설정의 구조를 검증하고 불러온다.
"""Resolve immutable version bundles without importing torch."""

from pathlib import Path
import math
import re
import yaml

POST_KEYS = {
    "DET_THRESH",
    "POST_THRESH",
    "MERGE_CLASSES",
    "MERGE_GAP",
    "FAR_MARGIN",
    "FAR_DROP_CLASSES",
    "OBS_DUP_IOU",
    "OBS_DUP_CONTAIN",
    "OBS_DUP_AREA_RATIO",
    "BOX_BIAS",
}


# 버전 이름과 실제 폴더 경로를 검사한다.
def version_dir(package, version):
    if not isinstance(version, str) or not re.fullmatch(
        r"[0-9]{8}_[A-Za-z0-9][A-Za-z0-9_-]*", version
    ):
        raise ValueError(
            "Version must be YYYYMMDD_name (letters, digits, _ and - only)"
        )
    folder = Path(package).resolve() / "weights" / version
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(f"Missing version directory: {folder}")
    return folder


# 후처리 설정의 키·타입·수치 범위를 검사한다.
def validate_postprocess(config):
    optional = {"OBS_KEEP_RING", "OBS_KEEP_MIN_FRAC", "OBS_KEEP_CLASSES"}
    group = {"OBS_GROUP_CONTAIN", "OBS_GROUP_MIN_COUNT", "OBS_GROUP_AREA_RATIO"}
    box_drop = {"DRIVABLE_BOX_DROP", "MERGE_RULE", "SEG_ENCLOSED_TO_NON", "BOX_RESCUE", "POST_SCALE", "ADVICE"}  # 11-13 POST_SCALE, 11-18 ADVICE(ViT-S 조언) 추가  # 선택 키(없으면 기존 동작)
    if not isinstance(config, dict) or set(config) - box_drop not in (POST_KEYS, POST_KEYS | optional, POST_KEYS | group, POST_KEYS | optional | group):
        raise ValueError("Unsupported postprocess constant set")
    if not isinstance(config["POST_THRESH"], dict) or set(config["POST_THRESH"]) != {
        0,
        1,
        2,
        3,
    }:
        raise ValueError("Unsupported detector classes")
    if not isinstance(config["BOX_BIAS"], dict) or set(config["BOX_BIAS"]) != {
        "obstacle_non",
        "hazard_caution",
        "min_score",
    }:
        raise ValueError("Unsupported BOX_BIAS structure")
    for key in ("MERGE_CLASSES", "FAR_DROP_CLASSES"):
        value = config[key]
        if (
            not isinstance(value, list)
            or any(type(x) is not int or x not in range(4) for x in value)
            or len(value) != len(set(value))
        ):
            raise ValueError(f"Unsupported {key} structure")
    if "OBS_KEEP_RING" in config:
        for k in ("OBS_KEEP_RING", "OBS_KEEP_MIN_FRAC"):
            if type(config[k]) not in (int, float) or not math.isfinite(config[k]) or not 0 < config[k] <= 1:
                raise ValueError(f"Invalid {k}")
        if not isinstance(config["OBS_KEEP_CLASSES"], list) or not config["OBS_KEEP_CLASSES"] or any(type(x) is not int or x not in range(3) for x in config["OBS_KEEP_CLASSES"]) or len(set(config["OBS_KEEP_CLASSES"])) != len(config["OBS_KEEP_CLASSES"]):
            raise ValueError("Invalid OBS_KEEP_CLASSES")
    if "OBS_GROUP_CONTAIN" in config:
        for k in ("OBS_GROUP_CONTAIN", "OBS_GROUP_AREA_RATIO"):
            if type(config[k]) not in (int, float) or not math.isfinite(config[k]) or not 0 < config[k] <= 1:
                raise ValueError(f"Invalid {k}")
        if type(config["OBS_GROUP_MIN_COUNT"]) is not int or config["OBS_GROUP_MIN_COUNT"] < 1:
            raise ValueError("Invalid OBS_GROUP_MIN_COUNT")
    if "DRIVABLE_BOX_DROP" in config:
        drop = config["DRIVABLE_BOX_DROP"]
        if not isinstance(drop, dict) or set(drop) - {"class_rules"} != {"enabled", "classes", "max_caution", "min_drivable"} or type(drop["enabled"]) is not bool:
            raise ValueError("Invalid DRIVABLE_BOX_DROP structure")
        for c, r in drop.get("class_rules", {}).items():  # 11-09 선택: {클래스: {max_caution, min_drivable}}
            if type(c) is not int or c not in drop["classes"] or not isinstance(r, dict) or set(r) != {"max_caution", "min_drivable"} or any(type(r[k]) not in (int, float) or not math.isfinite(r[k]) or not 0 <= r[k] <= 1 for k in r):
                raise ValueError("Invalid DRIVABLE_BOX_DROP class_rules")
        classes = drop["classes"]
        if not isinstance(classes, list) or any(type(x) is not int or x not in (0, 1, 2) for x in classes) or len(set(classes)) != len(classes):
            raise ValueError("Invalid DRIVABLE_BOX_DROP classes")
        for key in ("max_caution", "min_drivable"):
            if type(drop[key]) not in (int, float) or not math.isfinite(drop[key]) or not 0 <= drop[key] <= 1:
                raise ValueError("Invalid DRIVABLE_BOX_DROP threshold")
    if "SEG_ENCLOSED_TO_NON" in config:
        rule = config["SEG_ENCLOSED_TO_NON"]
        if not isinstance(rule, dict) or set(rule) != {"ring", "max_drivable"} or any(type(rule[k]) not in (int, float) or not math.isfinite(rule[k]) or not 0 <= rule[k] <= 1 for k in rule):
            raise ValueError("Invalid SEG_ENCLOSED_TO_NON")
    if "BOX_RESCUE" in config:
        rule = config["BOX_RESCUE"]
        if not isinstance(rule, dict) or set(rule) != {"classes", "low", "min_caution", "gap"} or not isinstance(rule["classes"], list) or not rule["classes"] or any(type(x) is not int or x not in (0, 1, 2) for x in rule["classes"]) or len(set(rule["classes"])) != len(rule["classes"]) or any(type(rule[k]) not in (int, float) or not math.isfinite(rule[k]) or not 0 <= rule[k] <= 1 for k in ("low", "min_caution", "gap")):
            raise ValueError("Invalid BOX_RESCUE")
    if "MERGE_RULE" in config:
        rule = config["MERGE_RULE"]
        if not isinstance(rule, dict) or set(rule) != {"mode", "gap", "fill", "cap"} or rule["mode"] != "greedy" or any(type(rule[k]) not in (int, float) or not math.isfinite(rule[k]) or not 0 <= rule[k] <= 1 for k in ("gap", "fill", "cap")):
            raise ValueError("Invalid MERGE_RULE")
    numbers = (
        [
            config[k]
            for k in (
                "DET_THRESH",
                "MERGE_GAP",
                "FAR_MARGIN",
                "OBS_DUP_IOU",
                "OBS_DUP_CONTAIN",
                "OBS_DUP_AREA_RATIO",
            )
        ]
        + list(config["POST_THRESH"].values())
        + list(config["BOX_BIAS"].values())
    )
    if any(
        type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in numbers
    ):
        raise ValueError("Invalid postprocess numeric value")
    if "ADVICE" in config:
        from .advice import validate_advice

        validate_advice(config["ADVICE"])
    bounded = numbers[:10] + [config["BOX_BIAS"]["min_score"]]
    if any(x > 1 for x in bounded):
        raise ValueError("Thresholds and normalized margins must be <= 1")
    return config


# 활성 또는 지정 버전의 모델·후처리 설정을 불러온다.
def load_version(package=None, version=None):
    package = Path(package or Path(__file__).resolve().parents[1])
    if version is None:
        version = yaml.safe_load((package / "config.yaml").read_text())["active_model"]
    folder = version_dir(package, version)
    model = yaml.safe_load((folder / "model_config.yaml").read_text())
    post = validate_postprocess(
        yaml.safe_load((folder / "postprocess.yaml").read_text())
    )
    return folder, model, post
