from pathlib import Path
import numpy as np

from koopman import TICA, EDMD
from util import load_from_file, save_to_file, set_attrs_from_dict, standardize_and_pool_features, plot_eigenfunctions
from basis import BasisSet, GaussianRFF


class TicaRffModel:
    """
    Standardize features, fit TICA, and fit a random-Fourier-feature EDMD on
    the TICA-projected data, for a single protein / feature scheme / lag
    combination.

    Less-obvious parameters:
        d_tica: number of TICA components to project onto and lift with RFF.
        p: number of random Fourier features sampled for the RFF basis.
        scaling: RFF kernel bandwidth (inverse length scale; larger = narrower kernel).
        tol_whitening: relative-eigenvalue cutoff used when whitening C0 (components
            with eigenvalue < tol_whitening * largest eigenvalue are dropped).
        batchwise_tica: compute covariance matrices in streaming batches instead of
            holding all trajectories in memory at once.
        symmetrize: average covariance matrices with their transpose before
            eigendecomposition (recommended when remove_mean/whitening introduces
            small numerical asymmetries).
        n_ev: number of eigenvalues/eigenvectors to keep from the whitened lagged
            covariance matrix.
        mean_pooling: for non-'closest-heavy' feature schemes, mean-pool standardized
            per-residue features into a single per-frame vector.
        n_eigenfunctions_to_plot: number of leading EDMD eigenfunctions to render in
            plot_eigenfunctions().
    """

    def __init__(
        self,
        dir_res,
        dt,
        traj_paths,
        lag,
        stride=1,
        feat_scheme='BioEmu_L1_features', # 'BioEmu_L1_features' or 'BioEmu_L1_features_pooled'
        mean_pooling=False, # after standardization
        remove_mean=True,
        load_intermediate_results=True,
        save_intermediate_results=True,
        batchwise_tica=True,
        symmetrize=True,
        tol_whitening=None, # whitening parameter, relative to the largest eigenvalue covariance matrix
        n_ev=10, # number of eigenvalues to compute and save for TICA
        n_eigenfunctions_to_plot=2,
        d_tica=9,
        p=300,
        scaling=5,
    ):
        
        self.dir_res = dir_res
        self.traj_paths = [Path(p) for p in traj_paths]

        self.stride = stride
        self.feat_scheme = feat_scheme
        self.mean_pooling = mean_pooling
        self.lag = lag

        self.remove_mean = remove_mean
        self.load_intermediate_results = load_intermediate_results
        self.save_intermediate_results = save_intermediate_results
        self.batchwise_tica = batchwise_tica
        self.symmetrize = symmetrize
        self.tol_whitening = tol_whitening
        self.n_ev = n_ev

        self.n_eigenfunctions_to_plot = n_eigenfunctions_to_plot
        self.d_tica = d_tica
        self.p = p
        self.scaling = scaling

        self.dt = self.stride * dt
        self.n_ev_path_extension = f'_n_ev={self.n_ev}'

        self.dir_tica = Path(f'{self.dir_res}/{self.feat_scheme}/stride={self.stride}')
        self.dir_tica.mkdir(exist_ok=True, parents=True)

        self.dir_tica_lag = self.dir_tica / Path(f'lag={self.lag}')
        self.dir_tica_lag.mkdir(exist_ok=True, parents=True)

        self.dir_edmd = Path(f'{self.dir_tica_lag}/rff_params={[self.d_tica, self.scaling, self.p]}')
        self.dir_edmd.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        data = [
            np.load(path)[::self.stride] if path.suffix == '.npy' else np.load(path)['features'][::self.stride]
            for path in self.traj_paths
        ]
        print('Loaded trajectory shapes:', [traj.shape for traj in data])

        if self.feat_scheme == 'closest-heavy':
            self.data_for_tica = data
        else:
            self.data_for_tica = standardize_and_pool_features(
                data=data,
                feat_scheme=self.feat_scheme,
                dir_base=self.dir_res,
                mean_pooling=self.mean_pooling,
            )
        return self.data_for_tica

    def fit_tica(self):
        self.tica_op = TICA(lag=self.lag)

        path_tica_results = self.dir_tica_lag / f'tica_lag={self.lag}.pickle'
        if path_tica_results.is_file():
            loaded = load_from_file(path_tica_results)
            set_attrs_from_dict(self.tica_op, loaded)
        else:
            self.tica_op.fit(
                X=self.data_for_tica,
                Y=None,
                remove_mean=self.remove_mean,
                mean=None,
                symmetrize=self.symmetrize,
                mode_whitening='cov',
                tol=self.tol_whitening,
                rank_whitening=None,
                batchwise=self.batchwise_tica,
                load_intermediate_results=self.load_intermediate_results,
                save_intermediate_results=self.save_intermediate_results,
                dir_base=self.dir_tica,
                n_ev=self.n_ev,
            )
            save_to_file(self.tica_op, path_tica_results, overwrite=True)
        return self.tica_op

    def project(self):
        mean = load_from_file(f"{self.dir_tica}/mean.npy")
        self.tica_projected_data = self.tica_op.transform(
            data=self.data_for_tica, dim=self.d_tica, mean=mean, weight_by_eval=True, batch_size=1000
        )
        return self.tica_projected_data

    def fit_rff(self):
        params_RFF = GaussianRFF(self.d_tica, self.p, self.scaling, dir_base=f'{self.dir_res}')
        self.Psi = BasisSet(params_RFF, basis_type='rff_real')

        self.tica_rff_op = EDMD(lag=self.lag, Phi=self.Psi)

        path_edmd_results = self.dir_edmd / Path(f'edmd_results_lag={self.lag}.pickle')

        if path_edmd_results.is_file():
            loaded = load_from_file(path_edmd_results)
            set_attrs_from_dict(self.tica_rff_op, loaded)
        else:
            self.tica_rff_op.fit(
                data=self.tica_projected_data,
                remove_mean=False,
                symmetrize=self.symmetrize,
                load_intermediate_results=False,
                save_intermediate_results=False,
                tol=1e-8,
                n_ev=self.n_ev,
            )
            save_to_file(self.tica_rff_op, path_edmd_results, overwrite=True)

        path_edmd_timescales = self.dir_edmd / Path(f'timescales_lag={self.lag}.npy')
        self.ts_ticarff = np.round(np.array(self.tica_rff_op.get_timescales(dt=self.dt)), 2)
        save_to_file(self.ts_ticarff, path_edmd_timescales)
        return self.tica_rff_op

    def plot_eigenfunctions(self):
        evecs_K = np.real(self.tica_rff_op.right_eigenvectors)[:, 1:]
        self.domain = np.real(np.vstack(self.tica_projected_data))

        plot_eigenfunctions(
            evecs_K, self.Psi, self.domain, W=None, k=self.n_eigenfunctions_to_plot,
            title=(
                f'd_tica={self.d_tica}, sigma={self.scaling}, p={self.p}, '
                f'lag={self.lag}, physical lag={round(self.lag * self.dt, 3)}, '
                f'timescales={self.ts_ticarff[1:self.n_eigenfunctions_to_plot+1]}'
            ),
            path=f'{self.dir_edmd}/eigenfunctions_{self.n_eigenfunctions_to_plot}{self.n_ev_path_extension}.png',
        )

    def run(self):
        self.load_data()
        self.fit_tica()
        self.project()
        self.fit_rff()
        self.plot_eigenfunctions()
        return self
