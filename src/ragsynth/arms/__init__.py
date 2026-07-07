"""Experiment arms A0/A1/A2/ORACLE, self-registered on import (SPEC §10)."""

from ragsynth.arms.a0_naive import A0Naive
from ragsynth.arms.a1_quota import A1Quota
from ragsynth.arms.a2_spec import A2Spec
from ragsynth.arms.base import ARMS, ArmPreset, GenerativeArmPreset, run_arm
from ragsynth.arms.oracle import OraclePreset

__all__ = [
    "ARMS",
    "A0Naive",
    "A1Quota",
    "A2Spec",
    "ArmPreset",
    "GenerativeArmPreset",
    "OraclePreset",
    "run_arm",
]
