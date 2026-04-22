# BMT paper reproduction in Python (GitHub-ready, Debian-friendly)

This repository contains a practical Python reproduction of the **Bipolar Mating Tendency (BMT)** paper, focused on:

- the **21 classical benchmark functions**
- the **CEC 2017 bound-constrained** suite (the benchmark block that is commonly used as the bound-constrained reference for the 2018 competition)
- the **3 engineering problems** from the paper
- the **bipolarity sweep**
- the **diversity and footprint figures**

The project is structured so you can upload it directly to GitHub and run it on a Debian/Ubuntu-style Linux machine with minimal manual setup.

## What is included

Main code:
- `src/bmt_repro/`: GA implementation, selection operators, benchmark definitions, engineering problems, statistics, and runtime helpers.
- `run_basic_benchmarks.py`: 21 classical benchmarks.
- `run_cec2017_bound_benchmarks.py`: CEC benchmark block using `cec2017-py`.
- `run_engineering_problems.py`: engineering problems from the paper.
- `run_bipolarity_sweep.py`: sweep of the BMT `q` parameter.
- `run_diversity_figures.py`: paper-like diversity and footprint figures.
- `scripts/smoke_test.py`: quick validation run for the environment.

This package intentionally excludes legacy and superfluous files from earlier iterations of the project.

## Selection methods included

The six methods compared in the paper:
- `ST`
- `RTS`
- `UTS`
- `FGTS`
- `CS`
- `BMT`

## System requirements (Debian / Ubuntu)

Install Python and the required system packages first:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git build-essential
```

You can also use the included helper script:

```bash
bash scripts/install_debian.sh
```

## Create a virtual environment and install dependencies

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Or use the helper script:

```bash
bash scripts/setup_venv.sh
```

The `requirements.txt` file already includes `cec2017-py` from GitHub.

If you want to install it manually, the command is:

```bash
pip install git+https://github.com/tilleyd/cec2017-py.git
```

## Optional editable install

This is not required, but you may install the project in editable mode:

```bash
pip install -e .
```

## Parallel execution model

The long-running experiment runners support **process-based parallelism**.

### Why process-based parallelism?
The runs are independent, so the cleanest and most robust speed-up is to distribute `(problem, population_size, run)` blocks across worker processes.

### Default and recommended settings
- Default worker count: **8**
- Recommended upper limit on your laptop: **12**
- You can still force a smaller or larger value with `--jobs`, but the command-line interface clamps the default recommended maximum to 12.

### Why this should not hurt RAM too much
Each worker only keeps the state of the current GA run in memory. The project does **not** keep large history arrays for normal benchmark runs, and compact CSV rows are returned from each worker. In practice this is much more CPU-bound than RAM-bound.

### Important implementation detail
The runners automatically set the following environment variables **before importing NumPy/SciPy-heavy modules**:

- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `VECLIB_MAXIMUM_THREADS=1`
- `BLIS_NUM_THREADS=1`

This avoids BLAS/OpenMP oversubscription when several worker processes run at the same time.

## Quick smoke test

To verify that the environment works:

```bash
python scripts/smoke_test.py
```

That smoke test runs:
- `dimension = 10`
- `runs = 1`
- `population_size = 100`
- `evals_per_dim = 100`
- algorithms `ST BMT CS`
- `jobs = 1`

## How to run experiments

### 1) 21 classical benchmark functions

```bash
python run_basic_benchmarks.py
```

Example with only `ST`, `BMT`, and `CS`:

```bash
python run_basic_benchmarks.py --algorithms ST BMT CS
```

Shorter example with parallelism:

```bash
python run_basic_benchmarks.py --runs 3 --population-sizes 100 --algorithms ST BMT CS --jobs 8
```

### 2) CEC 2017 bound-constrained suite

```bash
python run_cec2017_bound_benchmarks.py
```

Short example in 10 dimensions, one population size, one run:

```bash
python run_cec2017_bound_benchmarks.py \
  --runs 1 \
  --population-sizes 100 \
  --dimension 10 \
  --evals-per-dim 100 \
  --algorithms ST BMT CS \
  --jobs 8
```

Notes:
- the script defaults to `dimension = 10`
- it excludes `F2` by default
- if you do not set `--generations`, the script derives the number of generations from `evals_per_dim * dimension`

### 3) Engineering problems

```bash
python run_engineering_problems.py --jobs 8
```

### 4) Bipolarity sweep

```bash
python run_bipolarity_sweep.py --population-size 100 --jobs 8
```

### 5) Diversity figures

```bash
python run_diversity_figures.py
```

The diversity figure generator stays serial because it needs diagnostic traces and footprint collections.

## Outputs

All scripts write results under `results/...`.

The main CSV outputs are usually:
- `raw_results.csv`
- `summary_median_std.csv`
- `friedman_by_population.csv`
- `wilcoxon_by_population.csv`

The CEC runner also writes:
- `run_metadata.csv`

## Paper-style defaults

Default parameters follow the reconstructed paper setup:
- `pc = 0.7`
- `pm = 0.05`
- `lambda = 0.6`
- `tournament_size = 4`
- `population_sizes = 50, 100, 200`
- `runs = 25`
- `bipolarity = 0.25`

Reconstructed and editable method-specific defaults:
- `fgts_ftour = 4.5`
- `rts_window = 4`
- `association_size = 4`

## Reproducibility

Seed generation is stable across machines and executions. The code does **not** rely on Python's randomized `hash()` behavior.

## Repository layout

```text
.
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
├── run_basic_benchmarks.py
├── run_bipolarity_sweep.py
├── run_cec2017_bound_benchmarks.py
├── run_diversity_figures.py
├── run_engineering_problems.py
├── scripts
│   ├── install_debian.sh
│   ├── setup_venv.sh
│   └── smoke_test.py
└── src
    └── bmt_repro
        ├── __init__.py
        ├── benchmarks.py
        ├── cec2017_suite.py
        ├── diversity.py
        ├── engineering.py
        ├── experiment_workers.py
        ├── ga.py
        ├── parallel.py
        ├── runtime.py
        ├── selection.py
        ├── stats.py
        └── utils.py
```

## Uploading to GitHub

Minimal example:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin YOUR_GITHUB_URL
git push -u origin main
```

## Disclaimer

This repository is a **practical Python reconstruction** of the experimental setup from the BMT paper. It does not claim to be a byte-for-byte reproduction of the original MATLAB implementation.
