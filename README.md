# HRSNN

A controlled benchmark of spiking neural networks (SNNs) under hardware faults.
It trains the same network with different **spike coding schemes** and
**learning rules**, then evaluates every trained model under transient and
permanent fault models for four hardware **substrates** (digital / analogue ×
in-memory computing / neuromorphic).

Built on PyTorch and snnTorch.

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate    Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

For GPU training install the CUDA build of PyTorch that matches your driver
(see https://pytorch.org/get-started/locally/), for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Datasets (MNIST, CIFAR-10, N-MNIST) are downloaded into `data/` on first use.
All commands run from the repository root.

## Repository layout

```
config/           One YAML per experiment axis
  datasets/         mnist, cifar10, nmnist
  architectures/    mlp_5layer
  coding_schemes/   rate, ttfs, phase, burst, population
  learning_rules/   surrogate_gradient, bptt, stdp, r_stdp, eprop
src/              Library
  coding/           Spike encoders + ENCODER_REGISTRY
  data/             Datasets + DATASET_REGISTRY, GPU-side spike encoding
  models/           SNN MLP, blocks, neuron factories + MODEL_REGISTRY
  learning/         Learning rules + LEARNER_REGISTRY
  training/         Training engine
  tracking/         Metrics, CSV log, checkpoints, spike / SynOps activity
  utils/            Config, device, seed, logging, I/O, environment record
hardware/         Substrate fault models + HARDWARE_REGISTRY
fault_injection/  Transient and permanent fault injectors
experiments/      Entry points that train or evaluate
analysis/         Statistics and figures
tests/            Unit tests (unittest)
```

Runs write to `results/` (`raw/<run>/`, `aggregated/`, `figures/`), which is
not tracked by git.

## Train

```bash
python experiments/run_single.py --synthetic      # smoke test, no download
python experiments/run_single.py                  # MNIST, rate coding, surrogate gradient

python experiments/run_single.py \
    --dataset  config/datasets/cifar10.yaml \
    --arch     config/architectures/mlp_5layer.yaml \
    --coding   config/coding_schemes/ttfs.yaml \
    --learning config/learning_rules/bptt.yaml \
    --seed 42
```

| Option | Meaning | Default |
|---|---|---|
| `--dataset PATH` | Dataset YAML | `config/datasets/mnist.yaml` |
| `--arch PATH` | Architecture YAML | `config/architectures/mlp_5layer.yaml` |
| `--coding PATH` | Coding scheme YAML | `config/coding_schemes/rate.yaml` |
| `--learning PATH` | Learning rule YAML | `config/learning_rules/surrogate_gradient.yaml` |
| `--run-name STR` | Run folder name | `<dataset>_<coding>_<learning>_seed<N>` |
| `--device STR` | `auto`, `cpu`, `cuda`, `cuda:0` | `auto` |
| `--synthetic` | Random spike data, no download | off |
| `--seed INT` | Random seed | 42 |

Several models at once:

| Script | Varies | Key options |
|---|---|---|
| `experiments/run_sweep.py` | Full grid: datasets × architectures × codings × learners × seeds | `--datasets --archs --codings --learners --seeds --skip-existing --dry-run` |
| `experiments/run_ablation.py` | One axis, all others fixed | `--axis --base-dataset --base-arch --base-coding --base-learning --variants` |
| `experiments/run_learning_rules.py` | Learning rule only | `--dataset --arch --coding --rules --seeds` |

Each run writes `results/raw/<run_name>/`: `config.json` (configs, seed,
device, environment record), `metrics.csv` (one row per epoch),
`best_model.pth` (best-validation checkpoint) and `test_metrics.json` (test
accuracy and loss, spikes and SynOps per sample, accuracy per time step).

## Evaluate under hardware faults

```bash
python experiments/run_fault_evaluation.py --dataset mnist
python experiments/run_fault_evaluation.py --dataset mnist --category permanent
```

Loads every finished run and sweeps each fault model over its fault rates
(10 trials per point by default), without retraining. Results go to
`fault_results.json` in the run folder; an existing file is extended.

- **transient**: weight bit flips and membrane-register upsets on the digital substrates;
- **permanent**: stuck memory rows / columns on the digital substrates,
  stuck-ON / stuck-OFF memristor cells on the analogue substrates.

Options: `--results-dir --dataset --codings --category --faults --n-trials --device --max-batches --cpu-encoding --overwrite --dry-run`.

Further tools:

| Script | Purpose |
|---|---|
| `experiments/membrane_range.py` | Sizes the membrane register from the largest fault-free membrane value of every trained model |
| `experiments/membrane_sensitivity.py` | Repeats one run's membrane-upset sweep at several register ranges (`--run`) |
| `experiments/record_activity.py` | Adds spike / SynOps activity to runs that lack it |
| `experiments/verify_reproduction.py` | Compares runs against an archived copy of `results/raw` (`--archive`) |

## Analyse

| Script | Produces |
|---|---|
| `analysis/build_results_table.py` | `results/aggregated/results_table.{csv,tex}` |
| `analysis/statistical_tests.py` | Friedman / Wilcoxon tests by coding scheme and learning rule |
| `analysis/fault_statistics.py` | Fault-damage statistics and heatmaps (run after `statistical_tests.py`) |
| `analysis/plot_paper_figures.py` | Summary figures: fault curves, latency, accuracy–robustness trade-off, fault scale, sparsity (run after `fault_statistics.py`) |
| `analysis/plot_fault_reliability.py` | Fault retention curves, degradation bars, heatmaps (`--category transient\|permanent`) |
| `analysis/plot_fault_retention_stacked.py` | Stacked retention layout |
| `analysis/plot_learning_curves.py` | Validation accuracy per epoch |
| `analysis/plot_comparison_bars.py` | Accuracy per coding scheme / learning rule |
| `analysis/plot_confusion_matrices.py` | Per-class recall for one run (`--run-name`) |
| `analysis/plot_encoding_comparison.py` | One image under all five coding schemes |
| `analysis/plot_spike_rasters.py` | Input spike rasters for one run (`--run-name`) |
| `analysis/plot_energy_efficiency.py` | SynOps and spike sparsity |
| `analysis/plot_latency_analysis.py` | Accuracy vs time step, time to decision |
| `analysis/plot_power_estimation.py` | Energy per inference per substrate (currently fails; known issue) |

## Components

**Coding schemes** (`src/coding/`), all with `encode(x: [B, N], time_steps) -> [T, B, N]`:

| Scheme | Idea | Stochastic |
|---|---|---|
| Rate | Each input fires Bernoulli(p) at every step | Yes |
| TTFS | Higher input fires earlier, exactly once | No |
| Phase | Fires once per oscillation cycle; phase encodes intensity | No |
| Burst | Higher input gives more consecutive spikes from t = 0 | No |
| Population | K Gaussian receptive fields per input, rate-encoded | Yes |

**Learning rules** (`src/learning/`), all implementing `BaseLearner`
(`setup`, `step`, `on_epoch_end`):

| Rule | Method |
|---|---|
| `surrogate_gradient` | Backprop through time with surrogate spike gradients; AdamW |
| `bptt` | Backprop through time; Adam |
| `stdp` | Epoch 1: unsupervised pair-based STDP on the first hidden layer (k-winners-take-all, synaptic scaling); then that layer is frozen and the rest is trained by backprop |
| `r_stdp` | Reward-modulated STDP |
| `eprop` | Simplified e-prop on a feedforward network |

**Substrate fault models** (`hardware/`):

| Substrate | Weight format | Transient | Permanent |
|---|---|---|---|
| Digital IMC | 4-bit | `digital_imc`, `digital_imc_membrane` | `digital_imc_block` |
| Digital neuromorphic | INT8 | `digital_neuro`, `digital_neuro_membrane` | `digital_neuro_block` |
| Analog IMC | 6-bit effective, σ ≈ 2 % read variation | — | `analog_imc_stuck_on`, `analog_imc_stuck_off` |
| Analog neuromorphic | 4-bit effective, σ ≈ 5 % variation | — | `analog_neuro_stuck_on`, `analog_neuro_stuck_off` |

**Fault injectors** (`fault_injection/`) subclass `FaultInjector`
(`inject`, `restore`, context manager) and plug into `evaluate_under_fault()`:
`int_bit_flip`, `membrane_seu`, `bit_flip`, `probabilistic` (transient);
`memory_block`, `memristor_stuck`, `int_stuck_at_weights`, `stuck_at_weights`,
`stuck_at_logic` (permanent).

## Tests

```bash
python -m unittest discover -s tests -t .
```
