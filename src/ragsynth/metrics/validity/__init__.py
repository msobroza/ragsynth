"""Validity metric suite: ranking agreement, positive controls, system zoo.

Answers SPEC §8-9's criterion-validity question: does a synthetic query
set rank retrieval systems the way real production queries do, and does
it detect injected regressions? Three modules:

- :mod:`~ragsynth.metrics.validity.agreement` -- Kendall tau, tau_AP
  (Yilmaz/Aslam/Robertson, SIGIR 2008), RBO (Webber/Moffat/Zobel, TOIS
  2010), and the bootstrap-CI wrapper :func:`ranking_agreement`.
- :mod:`~ragsynth.metrics.validity.controls` -- degradation factory
  (index deletion, embedding noise) and the one-sided paired bootstrap
  (Sakai, SIGIR 2006).
- :mod:`~ragsynth.metrics.validity.systems` -- the deterministic
  retrieval-system zoo and binary-qrel nDCG@k scoring (PLAN D16).
"""

from ragsynth.metrics.validity.agreement import (
    RankingAgreement,
    ranking_agreement,
    rbo_ext,
    system_ranking,
    tau_ap,
)
from ragsynth.metrics.validity.controls import (
    drop_index_mask,
    noise_transform,
    paired_bootstrap_pvalue,
)
from ragsynth.metrics.validity.systems import (
    MatrixSystem,
    RetrievalSystem,
    evaluate_zoo,
    make_system_zoo,
)

__all__ = [
    "MatrixSystem",
    "RankingAgreement",
    "RetrievalSystem",
    "drop_index_mask",
    "evaluate_zoo",
    "make_system_zoo",
    "noise_transform",
    "paired_bootstrap_pvalue",
    "ranking_agreement",
    "rbo_ext",
    "system_ranking",
    "tau_ap",
]
