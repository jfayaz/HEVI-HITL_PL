"""Run the tests and every experiment in order.

python run_all.py           everything from scratch (about a day on a 20-core desktop)
python run_all.py --quick   tests, the fast experiments, and the long ones rebuilt from results/ (minutes)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAVED = ["--analysis-only"]

# script, arguments of the full run, arguments of the quick run (None: skipped)
STEPS = [
    ("01_decision_value", [], []),
    ("02_random_instances", [], SAVED),
    ("03_robustness_stress", [], ["--figure-only"]),
    ("04_certificates", [], ["--figure-only"]),
    ("05_scaling", [], None),
    ("06_diagrams", [], []),
    ("07_realtime_display", [], []),
    ("08_practice_checks", [], []),
    ("09_random_misspecification", [], SAVED),
    ("10_random_relevance", [], SAVED),
    ("11_reliability_grid", [], SAVED),
    ("12_authority_checks", [], None),
    ("13_benchmark_sensitivity", [], SAVED),
    ("14_authorities", [], SAVED),
    ("15_scaled_relevance", ["--workers", "6"], SAVED),
    ("16_joint_sensitivity", [], SAVED),
    ("17_reduction_ablation", [], SAVED),
    ("18_baselines", ["--workers", "6"], SAVED),
    ("19_response_models", [], SAVED),
    ("20_conditional_regret", ["--tolerance", "0.005"], None),
    ("20_conditional_regret", ["--tolerance", "0.025"], None),
    ("21_query_scaling", [], SAVED),
    ("22_social_access", [], []),
    ("23_forecast_skill", [], None),
    ("24_aggregation_baselines", [], SAVED),
    ("25_proof_audit", [], None),
    ("26_access_thresholds", [], None),
    ("27_value_directed_compression", [], None),
    ("28_catalogue_discretisation", [], None),
    ("29_moment_scaling", [], SAVED),
    ("29_moment_scaling", ["--figure"], ["--figure"]),
    ("30_comparison_sensitivity", [], []),
    ("31_common_component", [], []),
    ("32_forecast_free", [], SAVED),
    ("33_question_design", [], []),
    ("34_settled_trade", [], []),
    ("35_exposure_cap", [], []),
]


def run(name, argv):
    print(f"\n=== {name} ===", flush=True)
    start = time.perf_counter()
    if subprocess.run(argv, cwd=ROOT).returncode != 0:
        sys.exit(f"{name} failed")
    print(f"=== {name}: {time.perf_counter() - start:.0f} s ===", flush=True)


def main():
    quick = "--quick" in sys.argv
    run("tests", [sys.executable, "-m", "pytest", "-q", "tests"])
    for script, full, fast in STEPS:
        args = fast if quick else full
        if args is not None:
            run(" ".join([script] + args), [sys.executable, str(ROOT / "experiments" / f"{script}.py")] + args)


if __name__ == "__main__":
    main()
