"""Sampling and statistics: vMF/movMF planning density, demand map, spec sampler.

Ports the statistical core of the vendored prototype
(``reference/synth_query_eval.py`` L62-L321, SPEC §7): sphere utilities and
the Wood (1994) vMF rejection sampler, the Banerjee et al. (2005) movMF
mixture with EM, demand estimation with exponential time decay and
coverage-guaranteeing tilting, the frozen KMeans reference partition, and
the guarded spec sampler behind arm A2.
"""

from ragsynth.sampling.demand import demand_from_responsibilities, nn_cos_threshold, tilt_weights
from ragsynth.sampling.movmf import MovMF
from ragsynth.sampling.partition import ReferencePartition
from ragsynth.sampling.spec_sampler import SpecSampler
from ragsynth.sampling.vmf import (
    l2_normalize,
    log_sphere_area,
    sample_vmf,
    sphere_uniform,
    vmf_log_norm_const,
)

__all__ = [
    "MovMF",
    "ReferencePartition",
    "SpecSampler",
    "demand_from_responsibilities",
    "l2_normalize",
    "log_sphere_area",
    "nn_cos_threshold",
    "sample_vmf",
    "sphere_uniform",
    "tilt_weights",
    "vmf_log_norm_const",
]
