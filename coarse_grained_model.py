from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from util import load_from_file, save_to_file, set_attrs_from_dict
from pcca import PCCAoperator


class CoarseGrainedModel:
    """
    PCCA+ coarse-graining of a fitted TicaRffModel into metastable states.

    Parameters:
        n_pcca: number of metastable states to resolve.
        cut: PCCA+ membership-probability threshold in [0, 1]. A frame is assigned
            to a metastable state only if that state's membership (chi) value
            exceeds cut; frames that don't clear the threshold for any state are
            left in the "residual" set. Lower values assign more frames to states
            at the cost of less-confident assignments.
    """

    def __init__(self, tica_rff_model, n_pcca=4, cut=0.6):
        self.tica_rff_model = tica_rff_model
        self.n_pcca = n_pcca
        self.cut = cut

        self.dir_exports = tica_rff_model.dir_edmd / Path(f'pcca_{self.n_pcca}')
        self.dir_exports.mkdir(parents=True, exist_ok=True)

    def run_pcca(self):
        tica_rff_model = self.tica_rff_model

        right_evecs = tica_rff_model.tica_rff_op.right_eigenvectors
        lifted_traj = tica_rff_model.Psi(tica_rff_model.domain)
        dataTICARFF_stacked = lifted_traj @ right_evecs[:, :self.n_pcca]
        data_val_along_evec = dataTICARFF_stacked / np.max(np.abs(dataTICARFF_stacked), axis=0)
        data_val_along_evec = data_val_along_evec.real

        traj_lengths = [len(traj) for traj in tica_rff_model.tica_projected_data]

        path_pcca = self.dir_exports / f'pcca_{self.n_pcca}_states.pickle'
        self.pcca_op = PCCAoperator(nev=self.n_pcca, lag=tica_rff_model.lag, traj_lengths=traj_lengths, saving_dir=None)
        if path_pcca.is_file():
            set_attrs_from_dict(self.pcca_op, load_from_file(path_pcca))
        else:
            rescaled_eigen_data = self.pcca_op.rescale_eigen_projected_data(data_val_along_evec[:, :self.n_pcca], cplx=False)
            self.pcca_op.run_simplex(rescaled_eigen_data)
            self.pcca_op.compute_metastable_indices(cut=self.cut)
            save_to_file(self.pcca_op, path_pcca, overwrite=True)

        self.pcca_op.print_state_statistics(cut=self.cut)

        # Computed unconditionally (even when pcca_op was restored from an
        # older cached pickle that predates this attribute, or has
        # traj_lengths=None) so it's always available after run().
        self.pcca_op.compute_transition_matrix(cut=self.cut)
        self.transition_matrix = self.pcca_op.transition_matrix[self.cut]

        return self.pcca_op

    def plot_pcca(self):
        tica_rff_model = self.tica_rff_model
        domain = tica_rff_model.domain

        title_str = (
            f'lag={tica_rff_model.lag}, pcca_{self.n_pcca} timeseries, '
            f'{tica_rff_model.feat_scheme}, n_ev={tica_rff_model.n_ev}'
        )
        timescales_str = f'timescales:{tica_rff_model.ts_ticarff[1:self.n_pcca]}, physical lag={round(tica_rff_model.lag*tica_rff_model.dt, 3)}'

        # pcca tica clustering plot
        path_fig_pcca_traj = f'{self.dir_exports}/pcca_{self.n_pcca}_clustering{tica_rff_model.n_ev_path_extension}.png'
        fig_pcca = plt.figure(figsize=(10, 7))
        axs_pcca = fig_pcca.add_subplot(111, projection="3d")
        for i, idx_set in enumerate(self.pcca_op.indices[self.cut]):
            axs_pcca.scatter(domain[idx_set, 0], domain[idx_set, 1], domain[idx_set, 2], alpha=1, s=2, c=f'C{i}', label=f'State {i}', zorder=10)
        res_idx_set = self.pcca_op.residual[self.cut]
        axs_pcca.scatter(domain[res_idx_set, 0], domain[res_idx_set, 1], domain[res_idx_set, 2], alpha=0.4, s=2, c='grey', label='residual')
        axs_pcca.legend()
        axs_pcca.set_title(f'PCCA cut={self.cut}, {title_str}, {timescales_str}', fontsize=9, wrap=True)
        plt.tight_layout()
        plt.savefig(path_fig_pcca_traj)
        plt.close(fig_pcca)

        # pcca timeseries plot
        path_fig_pcca_timeseries = f'{self.dir_exports}/pcca_{self.n_pcca}_timeseries{tica_rff_model.n_ev_path_extension}.png'
        fig_pcca_timeseries, axs = plt.subplots(ncols=1, nrows=self.n_pcca + 1, figsize=(20, 8))
        for i in range(self.n_pcca):
            axs[i].plot(np.arange(len(self.pcca_op.chi[:, i])), self.pcca_op.chi[:, i], label=f'state {i}', c=f'C{i}')
            axs[i].legend()

        axs[-1].plot(
            np.arange(len(self.pcca_op.chi)),
            np.sum(self.pcca_op.chi, axis=1),
            label="sum",
            c="k",
        )
        axs[-1].legend()

        plt.suptitle(f'{title_str}, {timescales_str}', fontsize=9, wrap=True)

        plt.savefig(path_fig_pcca_timeseries)
        plt.close(fig_pcca_timeseries)

    def plot_transition_network(self):
        print(f'Transition matrix (row-stochastic, in %, cut={self.cut}):')
        print(self.transition_matrix)

        self.pcca_op.compute_state_colors(cut=self.cut)
        self.pcca_op.generate_transition_network(
            cut=self.cut,
            save_plot=True,
            show_plot=False,
            path=self.dir_exports,
            colors=self.pcca_op.state_colors[self.cut],
        )

    def run(self):
        self.run_pcca()
        self.plot_pcca()
        self.plot_transition_network()
        return self
