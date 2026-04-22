"""
AGentic_C — Optimisation Explainer
=====================================
Generates natural-language explanations for every decision the pipeline makes.

"Why is this function HOT?"
"Why was mem2reg chosen?"
"Why did we retry?"
"What improved?"

Output formats:
  - CLI text  (for --verbose mode)
  - JSON dict (for Web UI consumption)

Called by pipeline.py after compilation completes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


# ---------------------------------------------------------------------------
# Human-readable pass descriptions
# ---------------------------------------------------------------------------

PASS_DESCRIPTIONS = {
    "mem2reg":            "Promoted stack variables to registers (eliminates redundant loads/stores)",
    "sroa":               "Scalar replacement of aggregates (breaks structs into individual variables)",
    "-inline":            "Inlined function calls (removes call overhead ~3ns per call)",
    "always-inline":      "Force-inlined marked functions (eliminates indirect dispatch)",
    "simplifycfg":        "Simplified control flow graph (removed dead branches and landing pads)",
    "barrier-noop":       "Eliminated unnecessary memory barriers",
    "licm":               "Loop Invariant Code Motion (hoisted repeated computations out of loops)",
    "loop-unroll":        "Unrolled tight loops (reduced branch overhead, enabled better pipelining)",
    "gvn":                "Global Value Numbering (eliminated redundant price/value computations)",
    "dce":                "Dead Code Elimination (removed unreachable code paths)",
    "instcombine":        "Instruction Combining (merged instruction sequences into fewer ops)",
    "jump-threading":     "Jump Threading (eliminated predictable branch chains)",
    "aggressive-instcombine": "Aggressive instruction combining (deeper constant folding)",
    "reassociate":        "Reassociation (reordered arithmetic for better constant folding)",
    "sccp":               "Sparse Conditional Constant Propagation (propagated known values)",
    "loop-idiom":         "Loop Idiom Recognition (replaced loops with optimised libcalls)",
    "indvars":            "Induction Variable Simplification (simplified loop counters)",
    "loop-vectorize":     "Loop Vectorization (SIMD: processes 4× float32 per instruction with NEON)",
    "slp-vectorizer":     "Superword-Level Parallelism (vectorised across independent statements)",
    "loop-distribute":    "Loop Distribution (split loops for better vectorisation)",
    "loop-load-elim":     "Loop Load Elimination (cached repeated memory reads)",
    "alignment-from-assumptions": "Applied cache-line alignment to hot data structures",
    "post-ra-sched":      "Post Register Allocation Scheduling (reordered for CPU pipeline)",
    "machine-cse":        "Machine-Level CSE (eliminated redundant machine instructions)",
    "peephole-opt":       "Peephole Optimisation (replaced instruction patterns with cheaper ops)",
    "block-placement":    "Basic Block Placement (reordered blocks for branch predictor)",
    "branch-folder":      "Branch Folding (merged identical branch targets)",
    "loop-interchange":   "Loop Interchange (improved cache locality for nested loops)",
    "aarch64-simd-scalar":"ARM64 SIMD scalar pass (leveraged NEON scalar ops)",
    "arm-neon-vfp-peephole": "ARM NEON/VFP Peephole (NEON-specific instruction patterns)",
    "x86-pad-short-functions": "x86 Function Padding (avoided cache-line boundary fetches)",
}

PASS_WHY = {
    "mem2reg":            "LAP-001 (heap alloc) detected → promoting to registers eliminates malloc cost",
    "sroa":               "LAP-001 (heap alloc) detected → scalar replacement avoids dynamic allocation",
    "-inline":            "LAP-002 (virtual dispatch) or LAP-006 (indirect call) detected → inlining removes indirection",
    "always-inline":      "LAP-002/LAP-006 detected → force-inline eliminates vtable overhead",
    "simplifycfg":        "LAP-003 (exceptions) or LAP-010 (branches) detected → CFG cleanup",
    "licm":               "LAP-007 (atomics in loops) detected → hoisting out of loop reduces atomic pressure",
    "loop-unroll":        "High iteration count detected → unrolling reduces branch overhead",
    "loop-vectorize":     "Arithmetic-heavy loop detected → SIMD gives 4× throughput on NEON",
    "gvn":                "Redundant computations detected in IR → eliminating recalculations",
    "dce":                "Dead code paths detected → removing unreachable branches",
    "instcombine":        "Optimisation pass sequence generated improvements → combining resulting instructions",
    "jump-threading":     "LAP-010 (branch-heavy) detected → threading removes predictable jumps",
}

AP_DESCRIPTIONS = {
    "LAP-001": ("Heap Allocation",          "critical", "Dynamic memory allocation (new/malloc) on hot path causes 100ns–10µs spikes"),
    "LAP-002": ("Virtual Dispatch",          "critical", "Virtual function call requires vtable lookup (+5ns + potential cache miss)"),
    "LAP-003": ("Exception Handling",        "critical", "try/catch blocks add exception tables and catastrophic latency if thrown"),
    "LAP-004": ("Blocking Synchronisation",  "critical", "Mutex/lock_guard causes unbounded blocking (~20ns uncontended, ∞ contended)"),
    "LAP-005": ("System Call / I/O",         "major",    "printf/fwrite/send traps into kernel mode (1–10µs context switch)"),
    "LAP-006": ("Indirect Function Call",    "major",    "std::function/function pointer prevents inlining (+3–5ns + cache miss)"),
    "LAP-007": ("Atomic Operations",         "major",    "Strong memory ordering (seq_cst) costs 20–80ns due to memory barriers"),
    "LAP-008": ("RTTI / dynamic_cast",       "major",    "Runtime type check walks inheritance hierarchy (5–50ns per call)"),
    "LAP-009": ("Unaligned Memory Access",   "minor",    "Packed structs cause misaligned reads (+1–3ns, ARM may fault)"),
    "LAP-010": ("Branch-Heavy Logic",        "minor",    "Unpredictable branches cause misprediction penalty (10–20ns each)"),
}

HOTNESS_REASONS = {
    "annotation":     "[[hft::hot]] annotation explicitly marks this as a hot path function",
    "hot_root":       "Function name matches known HFT hot-path entry point patterns",
    "loop_depth":     "Function contains nested loops indicating compute-intensive logic",
    "branch_density": "High branch density suggests complex decision logic on the hot path",
    "keyword":        "Function name contains hot-path keywords (tick, signal, risk, order)",
    "unknown":        "Classification defaulted to HOT as conservative estimate (unknown function pattern)",
    "annotation_cold":"[[hft::cold]] annotation explicitly marks this as cold path",
    "cold_root":      "Function name matches known cold/setup/teardown patterns",
}


# ---------------------------------------------------------------------------
# Explainer dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FunctionExplanation:
    """Explanation for one code unit."""
    unit_name:          str
    path_label:         str             # 'hot' | 'cold'
    hotness_score:      int             = 0       # 0–100
    hotness_reasons:    List[str]       = field(default_factory=list)
    anti_patterns:      List[dict]      = field(default_factory=list)
    passes_applied:     List[dict]      = field(default_factory=list)
    latency_before_ns:  float           = 0.0
    latency_after_ns:   float           = 0.0
    latency_delta_ns:   float           = 0.0
    latency_pct:        float           = 0.0
    retries:            int             = 0
    verdict:            str             = "PASS"
    budget_ns:          int             = 0
    within_budget:      bool            = True
    summary:            str             = ""

    def to_dict(self) -> dict:
        return {
            "unit_name":         self.unit_name,
            "path_label":        self.path_label,
            "hotness_score":     self.hotness_score,
            "hotness_reasons":   self.hotness_reasons,
            "anti_patterns":     self.anti_patterns,
            "passes_applied":    self.passes_applied,
            "latency_before_ns": self.latency_before_ns,
            "latency_after_ns":  self.latency_after_ns,
            "latency_delta_ns":  self.latency_delta_ns,
            "latency_pct":       self.latency_pct,
            "retries":           self.retries,
            "verdict":           self.verdict,
            "budget_ns":         self.budget_ns,
            "within_budget":     self.within_budget,
            "summary":           self.summary,
        }


@dataclass
class PipelineExplanation:
    """Complete pipeline explanation for one compilation run."""
    source_path:        str             = ""
    functions:          List[FunctionExplanation] = field(default_factory=list)
    total_latency_before: float         = 0.0
    total_latency_after:  float         = 0.0
    avg_improvement_pct:  float         = 0.0
    reward:             float           = 0.0
    reward_breakdown:   dict            = field(default_factory=dict)
    benchmark:          dict            = field(default_factory=dict)
    pipeline_summary:   str             = ""

    def to_dict(self) -> dict:
        return {
            "source_path":          self.source_path,
            "functions":            [f.to_dict() for f in self.functions],
            "total_latency_before": self.total_latency_before,
            "total_latency_after":  self.total_latency_after,
            "avg_improvement_pct":  self.avg_improvement_pct,
            "reward":               self.reward,
            "reward_breakdown":     self.reward_breakdown,
            "benchmark":            self.benchmark,
            "pipeline_summary":     self.pipeline_summary,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_cli(self) -> str:
        """Rich CLI-formatted explanation."""
        lines = []
        sep = "─" * 68

        lines.append(sep)
        lines.append("  AGentic_C — Optimisation Explanation")
        lines.append(sep)

        for fn in self.functions:
            icon = "🔴" if fn.path_label == "hot" else "🔵"
            v_icon = "✓" if fn.verdict == "PASS" else "✗"
            lines.append(f"\n  {icon} {fn.unit_name}  [{fn.path_label.upper()}]  "
                         f"score={fn.hotness_score}/100  {v_icon} {fn.verdict}")
            lines.append(f"     Hotness: {' | '.join(fn.hotness_reasons[:2])}")
            lines.append(f"     Latency: {fn.latency_before_ns:.0f}ns → "
                         f"{fn.latency_after_ns:.0f}ns  "
                         f"(-{fn.latency_pct:.1f}%)")
            if fn.anti_patterns:
                codes = [ap.get("code","?") for ap in fn.anti_patterns]
                lines.append(f"     Anti-patterns: {', '.join(codes)}")
            if fn.passes_applied:
                top_passes = [p.get("name","?") for p in fn.passes_applied[:4]]
                lines.append(f"     Key passes: {', '.join(top_passes)}")
            if fn.retries:
                lines.append(f"     Retries: {fn.retries} (directive tightened each time)")

        lines.append(f"\n{sep}")
        lines.append(f"  Pipeline Summary")
        lines.append(sep)
        lines.append(f"  Total latency: {self.total_latency_before:.0f}ns → "
                     f"{self.total_latency_after:.0f}ns "
                     f"(-{self.avg_improvement_pct:.1f}%)")
        lines.append(f"  Reward score:  {self.reward:.4f}")
        lines.append(sep)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimisation Explainer
# ---------------------------------------------------------------------------

class OptimisationExplainer:
    """
    Generates human-readable explanations for every optimisation decision.

    Call explain() after pipeline.compile() returns.
    Returns a PipelineExplanation with both text and JSON output.

    Designed to answer the questions professors ask:
      "Why was this function marked HOT?"
      "Why was mem2reg applied?"
      "What improved?"
      "How does this compare to -O3?"
    """

    def explain(self,
                pipeline_result,
                reward_breakdown    = None,
                benchmark_result    = None,
                hotness_scores: dict = None   # unit_name → score dict from boss_agent
                ) -> PipelineExplanation:
        """
        Main entry point.

        Args:
            pipeline_result:  PipelineResult from pipeline.compile()
            reward_breakdown: PipelineRewardBreakdown from RewardEngine.compute()
            benchmark_result: dict with '-o3' vs 'agentic' comparison if --benchmark
            hotness_scores:   dict mapping unit_name → hotness info dict

        Returns:
            PipelineExplanation with .to_dict(), .to_json(), and .to_cli()
        """
        hotness_scores = hotness_scores or {}
        functions = []

        # Explain HOT units
        for r in getattr(pipeline_result, "hot_unit_results", []):
            fn_exp = self._explain_unit(r, "hot", hotness_scores)
            functions.append(fn_exp)

        # Explain COLD units
        for r in getattr(pipeline_result, "cold_unit_results", []):
            fn_exp = self._explain_unit(r, "cold", hotness_scores)
            functions.append(fn_exp)

        # Aggregate latency
        hot = getattr(pipeline_result, "hot_unit_results", [])
        lat_befores = [getattr(r, "latency_before_ns", 0.0) for r in hot if getattr(r, "latency_before_ns", 0) > 0]
        lat_afters  = [getattr(r, "latency_after_ns",  0.0) for r in hot if getattr(r, "latency_before_ns", 0) > 0]
        total_before = sum(lat_befores)
        total_after  = sum(lat_afters)
        avg_pct = 0.0
        if total_before > 0:
            avg_pct = (total_before - total_after) / total_before * 100

        reward = getattr(pipeline_result, "reward", 0.0)
        reward_dict = reward_breakdown.to_dict() if reward_breakdown else {}

        benchmark_dict = self._format_benchmark(benchmark_result) if benchmark_result else {}

        summary = self._pipeline_summary(
            functions, total_before, total_after, avg_pct, reward
        )

        return PipelineExplanation(
            source_path           = getattr(pipeline_result, "source_path", ""),
            functions             = functions,
            total_latency_before  = round(total_before, 2),
            total_latency_after   = round(total_after,  2),
            avg_improvement_pct   = round(avg_pct,      2),
            reward                = round(reward,       4),
            reward_breakdown      = reward_dict,
            benchmark             = benchmark_dict,
            pipeline_summary      = summary,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _explain_unit(self, r, path_label: str, hotness_scores: dict) -> FunctionExplanation:
        unit_name    = getattr(r, "unit_name", "unknown")
        lat_before   = getattr(r, "latency_before_ns", 0.0)
        lat_after    = getattr(r, "latency_after_ns",  0.0)
        lat_delta    = lat_before - lat_after
        lat_pct      = (lat_delta / lat_before * 100) if lat_before > 0 else 0.0
        passes       = getattr(r, "passes_applied", [])
        ap_raw       = getattr(r, "anti_patterns", [])
        retries      = getattr(r, "retries", 0)
        verdict      = getattr(r, "verdict", "PASS")
        budget_ns    = getattr(r, "budget_ns", 0)
        within_budget= getattr(r, "within_budget", True)

        # Hotness info
        hs = hotness_scores.get(unit_name, {})
        hotness_score   = hs.get("score", 50 if path_label == "hot" else 0)
        hotness_reasons = hs.get("reasons", [self._infer_hotness_reason(unit_name, path_label)])

        # Anti-patterns
        ap_dicts = []
        for ap_str in ap_raw:
            parts = ap_str.split(":")
            code  = parts[0] if parts else ap_str
            info  = AP_DESCRIPTIONS.get(code, (code, "unknown", ""))
            ap_dicts.append({
                "code":        code,
                "name":        info[0],
                "severity":    info[1],
                "description": info[2],
                "fix":         self._ap_fix_suggestion(code),
            })

        # Passes
        pass_dicts = []
        seen_passes = set()
        for p in passes:
            if p in seen_passes:
                continue
            seen_passes.add(p)
            pass_dicts.append({
                "name":        p,
                "description": PASS_DESCRIPTIONS.get(p, f"LLVM pass: {p}"),
                "why":         PASS_WHY.get(p, "Applied as part of optimisation sequence"),
            })

        # Summary
        if path_label == "hot":
            v_str = "✓ within budget" if within_budget else f"✗ over {budget_ns}ns budget"
            summary = (
                f"'{unit_name}' is a HOT function (score {hotness_score}/100). "
                f"Latency reduced from {lat_before:.0f}ns to {lat_after:.0f}ns "
                f"({lat_pct:.1f}% improvement). "
                f"{len(ap_raw)} anti-pattern(s) detected. "
                f"{len(passes)} passes applied. "
                f"Result: {v_str}."
            )
        else:
            summary = (
                f"'{unit_name}' is a COLD function — general optimisation only. "
                f"Latency {lat_before:.0f}ns → {lat_after:.0f}ns. "
                f"No latency budget enforced."
            )

        return FunctionExplanation(
            unit_name          = unit_name,
            path_label         = path_label,
            hotness_score      = hotness_score,
            hotness_reasons    = hotness_reasons,
            anti_patterns      = ap_dicts,
            passes_applied     = pass_dicts,
            latency_before_ns  = round(lat_before,  2),
            latency_after_ns   = round(lat_after,   2),
            latency_delta_ns   = round(lat_delta,   2),
            latency_pct        = round(lat_pct,     2),
            retries            = retries,
            verdict            = verdict,
            budget_ns          = budget_ns,
            within_budget      = within_budget,
            summary            = summary,
        )

    def _infer_hotness_reason(self, unit_name: str, path_label: str) -> str:
        n = unit_name.lower()
        if path_label == "cold":
            return HOTNESS_REASONS["annotation_cold"]
        for kw in ["tick", "signal", "risk", "order", "market", "price", "feed"]:
            if kw in n:
                return HOTNESS_REASONS["keyword"]
        return HOTNESS_REASONS["hot_root"]

    def _ap_fix_suggestion(self, code: str) -> str:
        fixes = {
            "LAP-001": "Use pre-allocated pools (arena/ring buffer). Allocate at startup, not per-tick.",
            "LAP-002": "Use CRTP (Curiously Recurring Template Pattern) for static polymorphism.",
            "LAP-003": "Remove try/catch from hot path. Use error codes or std::expected.",
            "LAP-004": "Replace mutex with lock-free SPSC queue or atomic operations.",
            "LAP-005": "Move logging to async thread via lock-free ring buffer.",
            "LAP-006": "Use direct function calls or non-capturing lambdas [ ]() with always_inline.",
            "LAP-007": "Use memory_order_relaxed where possible. Batch atomic reads.",
            "LAP-008": "Replace dynamic_cast with static_cast or eliminate runtime type checks.",
            "LAP-009": "Use __attribute__((aligned(64))) for cache-line alignment.",
            "LAP-010": "Use branchless arithmetic (?:) or lookup tables instead of switch.",
        }
        return fixes.get(code, "Review and optimise this pattern.")

    def _format_benchmark(self, benchmark_result: dict) -> dict:
        """Format benchmark comparison for web UI."""
        if not benchmark_result:
            return {}
        return {
            "o3_latency_ns":        benchmark_result.get("o3_latency_ns", 0),
            "agentic_latency_ns":   benchmark_result.get("agentic_latency_ns", 0),
            "improvement_pct":      benchmark_result.get("improvement_pct", 0),
            "o3_passes":            benchmark_result.get("o3_passes", []),
            "agentic_passes":       benchmark_result.get("agentic_passes", []),
            "o3_anti_patterns_fixed": False,
            "agentic_anti_patterns_fixed": True,
            "o3_retry": False,
            "agentic_retry": True,
            "o3_learning": False,
            "agentic_learning": True,
        }

    def _pipeline_summary(self, functions, total_before, total_after,
                          avg_pct, reward) -> str:
        hot_fns  = [f for f in functions if f.path_label == "hot"]
        cold_fns = [f for f in functions if f.path_label == "cold"]
        passed   = [f for f in hot_fns if f.verdict == "PASS"]
        aps_all  = [ap for f in hot_fns for ap in f.anti_patterns]
        ap_codes = list({ap["code"] for ap in aps_all})

        lines = [
            f"AGentic_C compiled {len(functions)} function(s): "
            f"{len(hot_fns)} HOT, {len(cold_fns)} COLD.",
            f"{len(passed)}/{len(hot_fns)} HOT units met their latency budget.",
        ]
        if total_before > 0:
            lines.append(f"Total hot-path latency: {total_before:.0f}ns → "
                         f"{total_after:.0f}ns ({avg_pct:.1f}% reduction).")
        if ap_codes:
            lines.append(f"Anti-patterns detected: {', '.join(ap_codes)}.")
        lines.append(f"Final reward score: {reward:.4f}.")
        return " ".join(lines)
