"""
AGentic_C — Reward Engine (Enhanced)
======================================
Five-component composite reward replacing the simplistic pipeline reward.

Formula:
    R = 0.50 * latency_improvement
      + 0.20 * instruction_reduction
      + 0.15 * antipattern_fix_ratio
      - 0.10 * retry_penalty
      + 0.05 * stability_bonus

All sub-scores are normalised to [0, 1] before weighting.
Final reward is clamped to [0, 1].

Compared to the old reward:
  OLD: R = lat_improvement + 0.2*budget_hit - 0.05*retries (uncapped)
  NEW: Full 5-component with explanations, per-unit breakdown, and
       stability bonus that rewards first-attempt budget hits.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Reward breakdown — one per unit, one for the pipeline total
# ---------------------------------------------------------------------------

@dataclass
class UnitRewardBreakdown:
    """Per-unit reward components — for logging and explainability."""
    unit_name:              str     = ""
    path_label:             str     = "hot"     # 'hot' | 'cold'

    # Sub-scores (0–1 each)
    latency_score:          float   = 0.0       # 0.50 weight
    instruction_score:      float   = 0.0       # 0.20 weight
    antipattern_score:      float   = 0.0       # 0.15 weight
    retry_penalty:          float   = 0.0       # 0.10 weight (subtracted)
    stability_bonus:        float   = 0.0       # 0.05 weight

    total:                  float   = 0.0       # final weighted reward
    explanation:            str     = ""        # human-readable breakdown


@dataclass
class PipelineRewardBreakdown:
    """Aggregate reward across all units."""
    total:                  float   = 0.0
    unit_rewards:           list    = field(default_factory=list)

    # Aggregate sub-scores (weighted averages)
    avg_latency_score:      float   = 0.0
    avg_instruction_score:  float   = 0.0
    avg_antipattern_score:  float   = 0.0
    avg_retry_penalty:      float   = 0.0
    avg_stability_bonus:    float   = 0.0

    # Meta
    hot_units_passed:       int     = 0
    cold_units:             int     = 0
    total_retries:          int     = 0
    explanation:            str     = ""

    def to_dict(self) -> dict:
        """Serialise for JSON output to Web UI."""
        return {
            "total":                 round(self.total, 4),
            "avg_latency_score":     round(self.avg_latency_score, 4),
            "avg_instruction_score": round(self.avg_instruction_score, 4),
            "avg_antipattern_score": round(self.avg_antipattern_score, 4),
            "avg_retry_penalty":     round(self.avg_retry_penalty, 4),
            "avg_stability_bonus":   round(self.avg_stability_bonus, 4),
            "hot_units_passed":      self.hot_units_passed,
            "cold_units":            self.cold_units,
            "total_retries":         self.total_retries,
            "unit_rewards": [
                {
                    "unit_name":         u.unit_name,
                    "path_label":        u.path_label,
                    "latency_score":     round(u.latency_score, 4),
                    "instruction_score": round(u.instruction_score, 4),
                    "antipattern_score": round(u.antipattern_score, 4),
                    "retry_penalty":     round(u.retry_penalty, 4),
                    "stability_bonus":   round(u.stability_bonus, 4),
                    "total":             round(u.total, 4),
                    "explanation":       u.explanation,
                }
                for u in self.unit_rewards
            ],
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Reward Engine
# ---------------------------------------------------------------------------

class RewardEngine:
    """
    Computes the five-component composite reward for the entire pipeline run.

    Weights (must sum to 1.0):
      latency     = 0.50  — biggest driver: did we cut the hot path?
      instruction = 0.20  — code size / complexity reduction
      antipattern = 0.15  — fraction of detected anti-patterns resolved
      retry       = 0.10  — penalty: each retry beyond the first costs points
      stability   = 0.05  — bonus: passing budget on the first attempt

    Design notes:
      - The old formula was: lat_improvement + 0.2*budget_bonus - 0.05*retries
        This has no upper bound and doesn't explain the score components.
      - The new formula is fully decomposed, clamped, and explainable.
      - Cold units contribute only via latency and instruction scores.
    """

    WEIGHTS = {
        "latency":     0.50,
        "instruction": 0.20,
        "antipattern": 0.15,
        "retry":       0.10,   # subtracted
        "stability":   0.05,   # bonus
    }

    MAX_RETRIES_FOR_ZERO = 3   # 3+ retries → maximum penalty

    def compute(self,
                hot_results:  list,   # list of UnitResult from pipeline
                cold_results: list    # list of UnitResult from pipeline
                ) -> PipelineRewardBreakdown:
        """
        Main entry point.
        Takes hot_results and cold_results from the pipeline and returns
        a full PipelineRewardBreakdown with per-unit details.

        Args:
            hot_results:  list of UnitResult objects (path_label='hot')
            cold_results: list of UnitResult objects (path_label='cold')

        Returns:
            PipelineRewardBreakdown with all metrics populated
        """
        if not hot_results and not cold_results:
            return PipelineRewardBreakdown(
                total       = 0.5,
                explanation = "No units processed — default reward 0.5."
            )

        unit_rewards = []

        for r in hot_results:
            ur = self._compute_hot_unit(r)
            unit_rewards.append(ur)

        for r in cold_results:
            ur = self._compute_cold_unit(r)
            unit_rewards.append(ur)

        # Aggregate
        totals     = [u.total             for u in unit_rewards]
        lat_scores = [u.latency_score     for u in unit_rewards]
        ins_scores = [u.instruction_score for u in unit_rewards]
        ap_scores  = [u.antipattern_score for u in unit_rewards]
        ret_pens   = [u.retry_penalty     for u in unit_rewards]
        stab_bons  = [u.stability_bonus   for u in unit_rewards]

        pipeline_total = float(np.mean(totals)) if totals else 0.5

        hot_passed = sum(
            1 for r in hot_results
            if getattr(r, "verdict", "") == "PASS"
        )
        total_retries = sum(
            getattr(r, "retries", 0) for r in hot_results
        )

        explanation = self._pipeline_explanation(
            total        = pipeline_total,
            n_hot        = len(hot_results),
            n_cold       = len(cold_results),
            hot_passed   = hot_passed,
            total_retries= total_retries,
            avg_lat      = float(np.mean(lat_scores)) if lat_scores else 0.0,
            avg_ap       = float(np.mean(ap_scores))  if ap_scores  else 0.0,
        )

        return PipelineRewardBreakdown(
            total                 = round(pipeline_total, 4),
            unit_rewards          = unit_rewards,
            avg_latency_score     = round(float(np.mean(lat_scores)), 4),
            avg_instruction_score = round(float(np.mean(ins_scores)), 4),
            avg_antipattern_score = round(float(np.mean(ap_scores)),  4),
            avg_retry_penalty     = round(float(np.mean(ret_pens)),   4),
            avg_stability_bonus   = round(float(np.mean(stab_bons)),  4),
            hot_units_passed      = hot_passed,
            cold_units            = len(cold_results),
            total_retries         = total_retries,
            explanation           = explanation,
        )

    # ------------------------------------------------------------------
    # Per-unit computation
    # ------------------------------------------------------------------

    def _compute_hot_unit(self, r) -> UnitRewardBreakdown:
        """Compute full 5-component reward for one HOT unit."""
        lat_before = getattr(r, "latency_before_ns", 0.0)
        lat_after  = getattr(r, "latency_after_ns",  0.0)
        retries    = getattr(r, "retries", 0)
        verdict    = getattr(r, "verdict", "FAIL")
        ap_list    = getattr(r, "anti_patterns", [])
        passes     = getattr(r, "passes_applied", [])
        unit_name  = getattr(r, "unit_name", "unknown")

        # --- Latency improvement (0–1) ---
        lat_score = 0.0
        if lat_before > 0:
            raw = (lat_before - lat_after) / lat_before
            lat_score = float(np.clip(raw, 0.0, 1.0))

        # --- Instruction reduction (0–1) estimated from passes applied ---
        # We don't always store instr before/after in UnitResult, so
        # we approximate from latency delta scaled by a constant factor.
        # (IR-level instruction reduction ≈ 0.8× latency improvement heuristic)
        ins_score = min(1.0, lat_score * 0.8)

        # --- Anti-pattern fix ratio (0–1) ---
        ap_score = 0.0
        if ap_list:
            total_aps  = len(ap_list)
            # Count how many AP codes have a corresponding pass that was applied
            resolved = 0
            for ap in ap_list:
                ap_code = ap.split(":")[0]   # e.g. "LAP-001"
                # Check if any pass in passes would resolve this AP
                resolving = _PASS_RESOLVES_AP.get(ap_code, set())
                if any(p in resolving for p in passes):
                    resolved += 1
            ap_score = resolved / total_aps if total_aps > 0 else 0.0
        else:
            # No anti-patterns detected → clean code → full score
            ap_score = 1.0

        # --- Retry penalty (0–1) ---
        # 0 or 1 retry → no penalty
        # 2 retries → 0.5 penalty
        # 3+ retries → full penalty (1.0)
        if retries <= 1:
            ret_penalty = 0.0
        else:
            ret_penalty = min(1.0, (retries - 1) / (self.MAX_RETRIES_FOR_ZERO - 1))

        # --- Stability bonus (0 or 1) ---
        # Budget hit on first attempt or no retries needed
        stab_bonus = 1.0 if (verdict == "PASS" and retries == 0) else 0.0

        total = (
             self.WEIGHTS["latency"]     * lat_score
           + self.WEIGHTS["instruction"] * ins_score
           + self.WEIGHTS["antipattern"] * ap_score
           - self.WEIGHTS["retry"]       * ret_penalty
           + self.WEIGHTS["stability"]   * stab_bonus
        )
        total = float(np.clip(total, 0.0, 1.0))

        explanation = self._unit_explanation(
            unit_name   = unit_name,
            path_label  = "hot",
            lat_before  = lat_before,
            lat_after   = lat_after,
            lat_score   = lat_score,
            ins_score   = ins_score,
            ap_score    = ap_score,
            ret_penalty = ret_penalty,
            stab_bonus  = stab_bonus,
            total       = total,
            retries     = retries,
            verdict     = verdict,
            ap_count    = len(ap_list),
        )

        return UnitRewardBreakdown(
            unit_name         = unit_name,
            path_label        = "hot",
            latency_score     = round(lat_score,   4),
            instruction_score = round(ins_score,   4),
            antipattern_score = round(ap_score,    4),
            retry_penalty     = round(ret_penalty, 4),
            stability_bonus   = round(stab_bonus,  4),
            total             = round(total,       4),
            explanation       = explanation,
        )

    def _compute_cold_unit(self, r) -> UnitRewardBreakdown:
        """Compute simplified reward for COLD units (no budget enforcement)."""
        lat_before = getattr(r, "latency_before_ns", 0.0)
        lat_after  = getattr(r, "latency_after_ns",  0.0)
        unit_name  = getattr(r, "unit_name", "unknown")

        lat_score  = 0.0
        if lat_before > 0:
            lat_score = float(np.clip(
                (lat_before - lat_after) / lat_before, 0.0, 1.0
            ))
        ins_score = min(1.0, lat_score * 0.8)

        # Cold units: no retry, no budget, no AP scan — give simple reward
        total = (
             self.WEIGHTS["latency"]     * lat_score
           + self.WEIGHTS["instruction"] * ins_score
           + self.WEIGHTS["antipattern"] * 0.5     # advisory, not penalised
           + self.WEIGHTS["stability"]   * 0.5     # cold path always stable
        )
        total = float(np.clip(total, 0.0, 1.0))

        explanation = (
            f"COLD unit '{unit_name}': general optimisation applied. "
            f"Latency {lat_before:.0f}ns → {lat_after:.0f}ns "
            f"(improvement {lat_score*100:.1f}%). No budget enforcement."
        )

        return UnitRewardBreakdown(
            unit_name         = unit_name,
            path_label        = "cold",
            latency_score     = round(lat_score, 4),
            instruction_score = round(ins_score, 4),
            antipattern_score = 0.5,
            retry_penalty     = 0.0,
            stability_bonus   = 0.5,
            total             = round(total, 4),
            explanation       = explanation,
        )

    # ------------------------------------------------------------------
    # Explanation builders
    # ------------------------------------------------------------------

    def _unit_explanation(self, unit_name, path_label, lat_before, lat_after,
                          lat_score, ins_score, ap_score, ret_penalty,
                          stab_bonus, total, retries, verdict, ap_count) -> str:
        W = self.WEIGHTS
        parts = [
            f"Unit '{unit_name}' [HOT] — Reward = {total:.3f}",
            f"  • Latency:     {lat_before:.0f}ns → {lat_after:.0f}ns "
            f"({lat_score*100:.1f}% reduction) "
            f"→ score {lat_score:.3f} × {W['latency']} = {lat_score*W['latency']:.3f}",
            f"  • Instruction: reduction ≈ {ins_score*100:.1f}% "
            f"→ score {ins_score:.3f} × {W['instruction']} = {ins_score*W['instruction']:.3f}",
            f"  • Anti-pattern: {ap_count} detected, "
            f"resolution ratio {ap_score*100:.0f}% "
            f"→ score {ap_score:.3f} × {W['antipattern']} = {ap_score*W['antipattern']:.3f}",
        ]
        if ret_penalty > 0:
            parts.append(
                f"  • Retry penalty: {retries} retries "
                f"→ -{ret_penalty:.3f} × {W['retry']} = -{ret_penalty*W['retry']:.3f}"
            )
        if stab_bonus > 0:
            parts.append(
                f"  • Stability bonus: passed on first attempt "
                f"→ +{stab_bonus:.3f} × {W['stability']} = +{stab_bonus*W['stability']:.3f}"
            )
        parts.append(f"  • Verdict: {verdict}")
        return "\n".join(parts)

    def _pipeline_explanation(self, total, n_hot, n_cold, hot_passed,
                              total_retries, avg_lat, avg_ap) -> str:
        lines = [
            f"Pipeline Reward: {total:.4f}",
            f"  HOT units: {hot_passed}/{n_hot} passed budget",
            f"  COLD units: {n_cold} (advisory only)",
            f"  Total retries: {total_retries}",
            f"  Avg latency improvement: {avg_lat*100:.1f}%",
            f"  Avg anti-pattern resolution: {avg_ap*100:.0f}%",
        ]
        if total >= 0.85:
            lines.append("  ★ Excellent — all units optimised, minimal retries.")
        elif total >= 0.70:
            lines.append("  ✓ Good — most units within budget.")
        elif total >= 0.50:
            lines.append("  ~ Acceptable — some units over budget or required retries.")
        else:
            lines.append("  ⚠ Low — significant optimisation headroom remains.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pass → Anti-pattern resolution mapping
# (mirrors ANTIPATTERN_PASS_MAP from ir_tuner_agent.py, reversed)
# ---------------------------------------------------------------------------

_PASS_RESOLVES_AP = {
    "LAP-001": {"mem2reg", "sroa"},
    "LAP-002": {"-inline", "always-inline"},
    "LAP-003": {"simplifycfg"},
    "LAP-004": {"barrier-noop", "licm"},
    "LAP-005": {"simplifycfg", "dce"},
    "LAP-006": {"-inline", "always-inline"},
    "LAP-007": {"licm", "instcombine"},
    "LAP-008": {"simplifycfg", "-inline"},
    "LAP-009": {"sroa", "instcombine"},
    "LAP-010": {"jump-threading", "simplifycfg"},
}
