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

## Configuration

Each config is a JSON file with a `tica_rff` section (passed to `TicaRffModel`)
and a `coarse_grained_model` section (passed to `CoarseGrainedModel`).

Key fields in `tica_rff`:

- `traj_paths`: list of trajectory files, one per trajectory (`.npy`, or `.npz`
  with a `features` array). `stride` applies to all of them equally, so if
  your trajectories were saved at different native strides, downsample them
  to a common stride yourself before listing them here.
- `feat_scheme` / `mean_pooling`: **must be set consistently with the actual
  shape of your trajectory files.**
  - `"closest-heavy"`: pairwise closest-heavy-atom distances, already a fixed
    per-frame vector (`n_frames, n_features`). `mean_pooling` is not used.
  - `"BioEmu_L1_features"`: raw per-residue embeddings, shape
    `(n_frames, n_residues, n_channels)`. Set `mean_pooling: true` to mean-pool
    over residues into a fixed-size per-frame vector before TICA.
  - `"BioEmu_L1_features_pooled"`: embeddings that have *already* been
    mean-pooled over residues elsewhere, shape `(n_frames, n_channels)`. Set
    `mean_pooling: false` here — pooling again would error (the data no
    longer has a residue axis to pool over).

  In short: if your file still has a residue dimension, use
  `"BioEmu_L1_features"` with `mean_pooling: true`; if it's already been
  reduced to one vector per frame, use `"BioEmu_L1_features_pooled"` with
  `mean_pooling: false`. Using the wrong `feat_scheme` for your data's actual
  dimensionality will fail with a shape-mismatch error at startup.
- `dt`: physical time per raw (pre-stride) frame, in whatever time unit you
  want lags/timescales reported in.
- `lag`: TICA/EDMD lag, in units of (strided) frames.
- `d_tica`: number of TICA components to keep and lift with random Fourier
  features. `p`, `scaling`: number of random Fourier features and kernel
  bandwidth for that lift.
- `n_ev`: number of eigenvalues/eigenvectors kept from the whitened lagged
  covariance matrix. `tol_whitening`: relative-eigenvalue cutoff used when
  whitening (drops directions with eigenvalue below `tol_whitening *` the
  largest one).
- `n_eigenfunctions_to_plot`: how many leading EDMD eigenfunctions to render.
- `batchwise_tica`: compute covariances in streaming batches instead of
  holding all trajectories in memory. `symmetrize`: average covariance
  matrices with their transpose before eigendecomposition. `remove_mean`:
  subtract the mean before fitting.
- `load_intermediate_results` / `save_intermediate_results`: cache/reuse
  intermediate matrices (covariances, whitening, EDMD fit) under `Results/`.

`coarse_grained_model`:

- `n_pcca`: number of metastable states to resolve with PCCA+.
- `cut`: membership-probability threshold in `[0, 1]` for assigning a frame
  to a state; frames below the threshold for every state are left "residual".

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
