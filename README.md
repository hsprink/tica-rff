# tica-rff

A small pipeline for analyzing protein MD trajectories: fits TICA on featurized
trajectories, then a random-Fourier-feature EDMD (Extended Dynamic Mode
Decomposition) on the TICA-projected data, and finally coarse-grains the
result into metastable states with PCCA+ (including transition-network plots
between states).

## Usage

```bash
python run_tica_rff.py <config_name>
```

where `<config_name>` is the name of a `.json` config file in this directory
(without the extension), e.g. `python run_tica_rff.py BBA`. See `BBA.json` /
`NTL9.json` for example configs — set `traj_paths` to your own featurized
trajectory files before running.

## Layout

- `run_tica_rff.py` — entry point; loads a config and runs the pipeline.
- `tica_rff.py` — `TicaRffModel`: loads data, fits TICA, projects, fits the RFF/EDMD model.
- `coarse_grained_model.py` — `CoarseGrainedModel`: PCCA+ coarse-graining and plotting.
- `koopman.py` — `TransferOperator`/`TICA`/`EDMD` classes (the AMUSE/whitening/eigendecomposition machinery).
- `basis.py` — random-Fourier-feature basis functions used to lift TICA-projected data for EDMD.
- `pcca.py` — `PCCAoperator`: PCCA+ clustering, state statistics, transition matrices/networks.
- `util.py` — shared I/O and plotting helpers.

Results (fitted models, intermediate matrices, plots) are written under
`Results/<protein_name>/...`, mirroring the parameters used (feature scheme,
stride, lag, RFF params, PCCA cut).
