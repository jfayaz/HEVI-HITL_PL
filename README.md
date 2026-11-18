# Code and results for the paper

> J. Fayaz, S. Ramsdale and A. Howes. *Timed risk and preference queries in catastrophe warnings: value of
> information and decision-preserving reduction.*

During a warning, an authority can request risk information or answer preference comparisons before it commits
protective measures, and every request uses warning time. The code solves this ask-or-commit problem exactly,
splits the value of information into risk, preference and interaction parts, and certifies reduced models.

## Setup

Python 3.13.

```
pip install -r requirements.txt
```

## Run

```
python run_all.py --quick                  # tests, fast experiments, the rest rebuilt from results/ (minutes)
python run_all.py                          # everything from scratch (about a day on a 20-core desktop)
python experiments/01_decision_value.py    # one experiment
python -m pytest -q tests
```

Most long experiments accept `--analysis-only` or `--figure-only` to rebuild their summary or figure from the rows
saved in `results/`. Seeds are fixed in the scripts, and the number of processes does not change any result. Only
the timings of experiments 05, 21 and 29 depend on the machine.

## Layout

| Folder | Content |
|---|---|
| `hevi_hitl/` | model, solvers and certificates |
| `experiments/` | one script per experiment, numbered as in the supplement |
| `results/` | rows (`.csv`) and summary numbers (`.json`) of each experiment |
| `figures/` | figures of the paper and the supplement |
| `tests/` | reference values, invariants and checks of the certificates |

| Module | Content | Paper |
|---|---|---|
| `model.py` | ask-or-commit model, exact history recursion, bounds of Proposition 1 | Sections 2.4 and 3.2 |
| `flood.py` | three-district flood benchmark, 192 configurations | Section 5.1, Appendix B |
| `social.py` | social-access extension, 768 configurations | Appendix C |
| `scaled.py` | scaled instances up to 518,112 configurations, fast history recursion | Section 5.8 |
| `moments.py` | exact solution by answer-subset moments | Section 4 |
| `reduction.py` | partitions and aggregation of configurations | Section 3.1 |
| `certificates.py` | covering endpoints, class-level certificate, query relaxation, guided refinement | Sections 3.3, 3.4 and 4 |
| `insight.py` | planners, value-of-information decomposition, misspecification, stress weighting | Sections 2.5 and 5 |
| `operational.py` | frozen policies and their evaluation | Section 5 |
| `forecast.py`, `access.py` | timed forecast updates and access-check variants | Section 5.7 |
| `plotting.py` | figure style | |

## Experiments

Times are for a 20-core desktop.

| Script | Reproduces | Time |
|---|---|---|
| `01_decision_value` | Fig. 2, Section 5.2 | 30 s |
| `02_random_instances` | Fig. 3, Section 5.3, S1 | 30 min |
| `03_robustness_stress` | Sections 5.4 and 5.5, Fig. S3 | 4 min |
| `04_certificates` | Fig. 5, Section 5.6, S6 | 10 min |
| `05_scaling` | Section 5.8, S2 and S8 (history recursion) | hours |
| `06_diagrams` | Figs. 1 and 4 | 15 s |
| `07_realtime_display` | Fig. S4 | 15 s |
| `08_practice_checks` | further numbers in Sections 5.2-5.8 and 6 | 15 s |
| `09_random_misspecification` | Section 5.4 | 20 min |
| `10_random_relevance` | Section 5.6 | 30 min |
| `11_reliability_grid` | Section 5.4 | 10 min |
| `12_authority_checks` | Sections 5.2 and 5.6 | 4 min |
| `13_benchmark_sensitivity` | Section 5.7, S3 | 20 min |
| `14_authorities` | Section 5.7, S3 | 15 min |
| `15_scaled_relevance` | Section 5.8, S2 | 45 min |
| `16_joint_sensitivity` | Section 5.7, S3 | 90 min |
| `17_reduction_ablation` | Section 5.6, S6 | 1 min |
| `18_baselines` | Section 5.3, S7 | 25 min |
| `19_response_models` | Section 5.4, S7 | 30 min |
| `20_conditional_regret` | Section 5.6, S6 | 12 min per tolerance |
| `21_query_scaling` | Section 5.8, S8 | hours |
| `22_social_access` | Section 5.7, S3 | 2 min |
| `23_forecast_skill` | Section 5.7, S10 | 70 min |
| `24_aggregation_baselines` | Section 5.6, S9 | 15 min |
| `25_proof_audit` | Appendix A, S13 | 2 min |
| `26_access_thresholds` | Section 5.7, S11 | 25 min |
| `27_value_directed_compression` | Sections 3.4 and 5.6, S9 | 15 min |
| `28_catalogue_discretisation` | Section 5.7, S12 | 20 min |
| `29_moment_scaling` | Section 5.8, S8, Fig. S2 | 1 h |
| `30_comparison_sensitivity` | Section 5.3, Fig. S1 | seconds |
| `31_common_component` | Section 5.6, S13 | seconds |
| `32_forecast_free` | Table 5, Sections 3.4 and 5.6, S15 | 10 min |
| `33_question_design` | Sections 5.2 and 5.7, S16 | 15 s |
| `34_settled_trade` | Sections 5.2 and 5.7, S16 | 1 min |
| `35_exposure_cap` | Sections 5.2 and 5.7, S16 | 20 s |

S1 to S16 are sections of the supplement. Table 4 lists the constants of `hevi_hitl/flood.py`.
