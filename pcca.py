# Class for performing PCCA+ clustering on a set of eigenvectors and storing the results

import numpy as np
import collections
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib import colormaps

# for generating kinetic network
import networkx as nx 
from deeptime.markov import TransitionCountEstimator

from util import save_to_file, load_from_file


class PCCAoperator():
    # class for performing PCCA+ clustering on a set of eigenvectors and storing the results

    # main attributes:
    # nev:                      number of metastable states
    # lag:                      lag time used to compute the eigenvectors
    # chi:                      membership matrix
    # cuts:                     set of cut values used to define metastable states (cut-offs on chi)
    # indices:                  dictionary mapping cut values to lists of arrays of indices per metastable state
    # residual:                 dictionary mapping cut values to arrays of indices not assigned to any metastable state
    # markov:                   dictionary mapping cut values to arrays of state assignments per original state
    # freq_per_state:           dictionary mapping cut values to arrays of frequencies per metastable state
    # free_energy_per_state:    dictionary mapping cut values to arrays of free energies per metastable state (use protein-specific temperature)

    # main functions:
    # run_simplex:                  perform PCCA+ clustering on a set of eigenvectors
    # compute_metastable_indices:   compute metastable state assignments for a given cut value
    # compute_state_statistics:     compute frequencies and free energies per metastable state for a given cut value
    # print_state_statistics:       print frequencies and free energies per metastable state for a given cut value
    # generate_contact_maps:        generate and save contact maps for each metastable state
    # compute_state_colors:         compute colors for each metastable state based on free energies
    # generate_transition_network:  generate and save a transition network plot between metastable states
    # export_xtc:                   export trajectory files for each metastable state and optionally for the residual states
    # get_random_indices:           get random indices per metastable state for a given cut value
    # get_xtc_paths:                get paths to exported trajectory files for each metastable state


    @staticmethod
    def getDir(nev, lag, **kwargs):
        return f'lag={lag}_nStates={nev}'

    def __init__(self, nev, lag, traj_lengths=None, saving_dir=None):
        self.nev = nev
        self.lag = lag
        self.saving_dir = Path(saving_dir) if saving_dir is not None else None

        self.colormap = None
        self.min_energy = None
        self.max_energy = None

        self.traj_lengths = traj_lengths

        self.chi = None
        self.cuts = set()

        self.indices = dict()
        self.residual = dict()
        self.markov = dict()
        self.image_paths = dict()

        self.N_per_state = dict()
        self.freq_per_state = dict()
        self.free_energy_per_state = dict()
        self.state_colors = dict()

        self.transition_counts = dict()
        self.transition_matrix = dict()

        # a list of sttributes that are saved when util.save_to_file is used
        self.savingList = {'nev', 'lag', 'chi', 'cuts', 'indices', 'residual', 'markov', 'image_paths', 'saving_dir', 'traj_lengths', 'transition_counts', 'transition_matrix'}

    def post_load(self):
        """
        Recompute derived per-cut statistics (frequencies, free energies)
        after restoring a PCCAoperator from a saved dict -- called
        automatically by util.set_attrs_from_dict when loading (which does
        not persist self.freq_per_state/self.free_energy_per_state directly,
        only the raw indices/markov assignments they're derived from).
        """
        assert self.cuts == set(self.indices.keys())
        self.N = self.chi.shape[0]
        for cut in self.cuts:
            self.compute_state_statistics(cut=cut)

    def run_simplex(self, V):
        """
        Fit PCCA+ membership vectors (self.chi) via the inner-simplex
        algorithm (_pcca_connected_isa) on the leading self.nev columns of V.

        Parameters:
            V: rescaled eigenvector-projected data, shape (n_frames, >=nev)
                (see rescale_eigen_projected_data). n_frames must equal
                sum(self.traj_lengths) if traj_lengths was provided in
                __init__; otherwise all of V is assumed to be a single
                trajectory and self.traj_lengths is set to [V.shape[0]] --
                get this wrong for multi-trajectory data and
                compute_transition_matrix will (incorrectly) count a
                transition across each trajectory boundary.
        """
        if self.traj_lengths is not None:
            assert V.shape[0] == sum(self.traj_lengths), "The number of rows in the eigenvector matrix must match the total number of frames in the trajectories."
        else:
            self.traj_lengths = [V.shape[0]]

        chi, pcca_transform = _pcca_connected_isa(V[:, :self.nev], n_clusters=self.nev)
        self.N = chi.shape[0]
        self.chi = chi
        self.pcca_transform = pcca_transform

    def rescale_eigen_projected_data(self, projectedData, cplx=True):
        """
        Prepare eigenvector-projected data for run_simplex: force the first
        column constant (PCCA+ requires the leading/stationary eigenvector to
        be constant), and, if the data is complex, rotate each remaining
        column's phase to minimize its imaginary part before discarding it.

        Parameters:
            projectedData: array (n_frames, n_eigenvectors).
            cplx: whether projectedData is complex-valued and needs the
                phase-rotation step; pass False for already-real data.

        Returns:
            rescaled copy of projectedData, same shape, real-valued input for
            run_simplex.
        """
        rescaledData = projectedData.copy()

        # if data is complex, rotate each dimension such that imaginary part is minimized
        if cplx:
            for ii in range(rescaledData.shape[1]):
                theta_opt = 0.0
                vi = rescaledData[:, ii]
                vmax = np.max(np.abs(np.imag(vi)))
                for theta in np.arange(0, 2 * np.pi, 0.05):
                    currVal = np.max(np.abs(np.imag(np.exp(1j * theta) * vi)))
                    if currVal < vmax:
                        theta_opt = theta
                        vmax = np.max(np.abs(np.imag(np.exp(1j * theta) * vi)))
                rescaledData[:, ii] = np.real( rescaledData[:, ii] * np.exp(1j * theta_opt) )

        rescaledData[:, 0] = np.mean(rescaledData[:, 0])

        return rescaledData


    def compute_metastable_indices(self, cut):
        """
        Assign each frame to a metastable state (or to the residual set) at a
        given membership threshold, and cache the result under self.cuts.

        cut: PCCA+ membership-probability threshold in [0, 1]. Frame i is
            assigned to state j if self.chi[i, j] >= cut; frames that don't
            clear the threshold for any state are left in the residual set
            (self.residual[cut]). Lower cut assigns more (less-confident)
            frames to states; this is the parameter referenced as `cut`
            throughout the rest of this class (compute_state_statistics,
            compute_transition_matrix, generate_transition_network, etc.).

        Results are cached per cut (self.indices[cut], self.residual[cut],
        self.markov[cut]) -- calling again with a cut already in self.cuts is
        a no-op.
        """
        if cut in self.cuts:
            return

        assert self.chi is not None

        ans = []
        res = np.ones(self.chi.shape[0], dtype=bool)
        markov = np.zeros(self.chi.shape[0], dtype=int)

        for j in range(self.nev):
            ind = np.where(self.chi[:, j] >= cut)[0]
            ans.append(ind)
            res[ind] = False 
            markov[ind] = j

        res = np.where(res)[0]
        
        # set value for pcca transition states to nev
        markov[res] = self.nev

        self.residual[cut] = res
        self.indices[cut] = ans
        self.markov[cut] = markov
        self.cuts.add(cut)

        self.compute_state_statistics(cut)

    

    def get_random_indices(self, cut, n_random_per_state, dir_base=None):

        self.compute_metastable_indices(cut)
        list_of_indices = self.indices[cut]

        random_indices_filename = Path(f'{n_random_per_state}_random_indices_per_state.npy')

        if dir_base is None:
            if not hasattr(self, 'saving_dir'):
                raise ValueError('Need saving dir')
            dir_base = self.saving_dir / Path(f'cut={cut}') / Path('random_indices')

        random_indices_path = dir_base / random_indices_filename

        if random_indices_path.is_file():
            print(f"Loading random indices for cut {cut} from file...")
            random_indices = load_from_file(random_indices_path)

        else:
            print(f"Generating random indices for cut {cut} and saving to file...")
            random_indices = [np.random.choice(idcs, min(len(idcs), n_random_per_state),replace=False) for idcs in list_of_indices]
            save_to_file(random_indices, filepath=random_indices_path)

        return random_indices


    def export_xtc(self, md_traj, cut, export_residual=False, all_per_state=False, n_random_per_state=None, dir_random=None, base_dir=None, return_paths=False):

        if base_dir is None:
            if not hasattr(self, 'saving_dir') or self.saving_dir is None:
                raise ValueError('Need saving dir')
            base_dir = self.saving_dir
        
        final_dir = Path(base_dir)
        final_dir.mkdir(parents=True, exist_ok=True)

        if not cut in self.indices.keys():
            self.compute_metastable_indices(cut)

        if all_per_state:
            paths_all = []
            filename_xtc = lambda i, n: Path(f"PCCA_State_{i}_N={n}.xtc")

            for i, idcs in enumerate(self.indices[cut]):
                traj_state = md_traj[idcs]
                traj_state.superpose(traj_state[-1])

                curr_path = final_dir / filename_xtc(i, len(idcs))
                paths_all.append(curr_path)

                # print(f"Saving PCCA traj {i} to file...", end=' ')
                traj_state.save_xtc(curr_path)
                # print(f"Done")

        if n_random_per_state is not None:
            paths_random = []
            random_indices = self.get_random_indices(cut=cut, n_random_per_state=n_random_per_state, dir_base=dir_random)
            filename_xtc = lambda i, n_samples : Path(f"PCCA_State_{i}_random={n_samples}.xtc")

            for i, idcs in enumerate(random_indices):
                curr_path = final_dir / filename_xtc(i, len(idcs))
                paths_random.append(curr_path)

                random_traj_state = md_traj[idcs]
                random_traj_state.superpose(random_traj_state[-1])

                random_traj_state.save_xtc(curr_path)

        if export_residual:
            idcs = self.residual[cut]
            traj_state = md_traj[idcs]
            traj_state.superpose(traj_state[-1])

            file_name_xtc_residue = lambda n: Path(f"PCCA_residue_N={n}.xtc")
            final_path = final_dir / file_name_xtc_residue(len(idcs))

            # print(f"Saving PCCA residue to file...", end=' ')
            traj_state.save_xtc(final_path)
            # print(f"Done")

        if return_paths:
            if all_per_state:
                if n_random_per_state is not None:
                    return paths_all, paths_random
                return all_per_state
            if n_random_per_state is not None:
                return paths_random
            
    def export_pdb(self, md_traj, cut, n_frames=10, dir_random=None, base_dir=None):

        list_of_indices = self.get_random_indices(cut=cut, n_random_per_state=n_frames, dir_base=dir_random)

        if base_dir is None:
            if not hasattr(self, 'saving_dir'):
                raise ValueError('Need saving dir')
            base_dir = self.saving_dir

        final_dir = Path(base_dir) /  Path(f'{n_frames}_random_per_state')
        final_dir.mkdir(parents=True, exist_ok=True)

        for state, idcs_state in enumerate(list_of_indices):
            for curr_idx in idcs_state:
                md_traj_frame = md_traj[curr_idx]
                curr_path = final_dir / Path(f'PCCA_State_{state}_frame={curr_idx}.pdb')
                md_traj_frame.save_pdb(curr_path)


    def get_feat_paths(self, cut, all_per_state=False, n_random_per_state=None, dir_random=None, base_dir=None):

        if base_dir is None:
            if not hasattr(self, 'saving_dir'):
                raise ValueError('Need saving dir')
            base_dir = self.saving_dir

        final_dir = base_dir / Path(f'cut={cut}')

        if all_per_state:
            paths_all = []
            filename_eigenfunctions = lambda i, n: Path(f'eigenfunction_state_{i}_N={n}.npy')
            for i, idcs in enumerate(self.indices[cut]):
                curr_path = final_dir / filename_eigenfunctions(i, len(idcs))
                paths_all.append(curr_path)

        if n_random_per_state is not None:
            paths_random = []
            random_indices = self.get_random_indices(cut=cut, n_random_per_state=n_random_per_state, dir_base=dir_random)
            filename_eigenfunctions = lambda i : Path(f'eigenfunction_state_{i}_random={n_random_per_state}.npy')

            for i, idcs in enumerate(random_indices):
                curr_path = final_dir / filename_eigenfunctions(i)
                paths_random.append(curr_path)

        if all_per_state:
            if n_random_per_state is not None:
                return paths_all, paths_random
            return all_per_state
        if n_random_per_state is not None:
            return paths_random


    def get_xtc_paths(self, cut, all_per_state=False, n_random_per_state=None, dir_random=None, base_dir=None):

        if base_dir is None:
            if not hasattr(self, 'saving_dir'):
                raise ValueError('Need saving dir')
            base_dir = self.saving_dir

        final_dir = base_dir / Path(f'cut={cut}')

        if all_per_state:
            paths_all = []
            filename_xtc = lambda i, n: Path(f"PCCA_State_{i}_N={n}.xtc")

            for i, idcs in enumerate(self.indices[cut]):
                curr_path = final_dir / filename_xtc(i, len(idcs))
                paths_all.append(curr_path)

        if n_random_per_state is not None:
            paths_random = []
            random_indices = self.get_random_indices(cut=cut, n_random_per_state=n_random_per_state, dir_base=dir_random)
            filename_xtc = lambda i : Path(f"PCCA_State_{i}_random={n_random_per_state}.xtc")

            for i, idcs in enumerate(random_indices):
                curr_path = final_dir / filename_xtc(i)
                paths_random.append(curr_path)

        if all_per_state:
            if n_random_per_state is not None:
                return paths_all, paths_random
            return all_per_state
        if n_random_per_state is not None:
            return paths_random


    def compute_state_statistics(self, cut, T=300):
        if not cut in self.cuts:
            self.compute_metastable_indices(cut=cut)
        res = self.residual[cut]
        indcs = self.indices[cut]

        N_per_state = [len(x) for x in indcs]
        N_per_state.append(len(res))

        # factor = - (scipy.constants.Boltzmann / 1000 * scipy.constants.Avogadro * T) ** -1
        factor = - (8.314462618 * T) / 1000
        freeEnergy = lambda p: factor * np.log(p)

        self.N_per_state[cut] = N_per_state
        self.freq_per_state[cut] = np.array([x / self.N for x in N_per_state])
        self.free_energy_per_state[cut] = np.array([freeEnergy(x) for x in self.freq_per_state[cut]])


    def print_state_statistics(self, cut):

        if not cut in self.cuts:
            self.compute_metastable_indices(cut=cut)

        indices = self.N_per_state[cut]
        freq_per_state = np.round(100 * self.freq_per_state[cut], 3)
        free_energy_per_state = np.round(self.free_energy_per_state[cut], 3)

        print(f'total N: {self.N}, cut: {cut}')

        for i, (N, ratio, energy) in enumerate(zip(indices, freq_per_state, free_energy_per_state)):
            empty_spaces1 = (8 - len(str(N))) * ' '
            empty_spaces2 = (8 - len(str(ratio))) * ' '
            if i == self.nev:
                print(f'residue: N: {N} {empty_spaces1} freq: {ratio} % {empty_spaces2} free energy: {energy}')
            else:
                print(f'state {i}: N: {N} {empty_spaces1} freq: {ratio} % {empty_spaces2} free energy: {energy}')
        print('\n')


    def generate_contact_maps(self,
                                traj,
                                cut,
                                n_res,
                                tol=0.3,
                                save_plot=False,
                                protein_prefix='',
                                final_dir=None,
                                cmap='viridis',
                                labelstep=5,
                                res_id_offset=1,
                                interpolation=None,
                                show_plot=False,
                                return_mat=False,
                                mode='frequency',
                                headlines=None,
                                batchwise=None):


        if save_plot or final_dir is not None:

            if final_dir is not None:
                save_plot = True
                folder_name = final_dir

            elif save_plot:
                if hasattr(self, 'saving_dir'):
                    folder_name = self.saving_dir / Path(f'cut={cut}')

            folder_name.mkdir(exist_ok=True, parents=True) 
            final_path = lambda i, ending: folder_name / Path(f'{protein_prefix}cut={cut}_contact_tol={tol}_interpolation={interpolation}_state{i}{ending}') # _{np.round(self.free_energy_per_state[cut][i], 2)

        ind_triu = np.triu_indices(n_res, k=3)
        ind_rows = ind_triu[0]
        ind_cols = ind_triu[1]

        results_mean = []
        results_variance = []
        
        # Precompute trajectory index boundaries to map global indices to specific trajectory chunks
        traj_lengths = np.array([t.shape[0] for t in traj])
        cum_lengths = np.insert(np.cumsum(traj_lengths), 0, 0)
        d = traj[0].shape[1]

        for ii, ind in enumerate(self.indices[cut]):
            
            if isinstance(batchwise, int):
                n_frames = len(ind)
                sum_x = np.zeros(d, dtype=np.float64)
                sum_x2 = np.zeros(d, dtype=np.float64)
                sum_m = np.zeros(d, dtype=np.float64)
                
                # Process the data batch-by-batch over global indices
                for b_start in range(0, n_frames, batchwise):
                    b_ind = ind[b_start:b_start+batchwise]
                    
                    b_traj_indices = np.searchsorted(cum_lengths, b_ind, side='right') - 1
                    b_local_indices = b_ind - cum_lengths[b_traj_indices]
                    
                    batch_data = np.empty((len(b_ind), d), dtype=traj[0].dtype)
                    for t_idx in np.unique(b_traj_indices):
                        mask = (b_traj_indices == t_idx)
                        batch_data[mask] = traj[t_idx][b_local_indices[mask]]
                        
                    sum_x += np.sum(batch_data, axis=0, dtype=np.float64)
                    sum_x2 += np.sum(batch_data**2, axis=0, dtype=np.float64)
                    
                    if mode == 'frequency':
                        if isinstance(tol, float):
                            batch_data_m = (batch_data <= tol).astype(float)
                        else:
                            batch_data_m = batch_data
                    else:
                        batch_data_m = batch_data
                        
                    sum_m += np.sum(batch_data_m, axis=0, dtype=np.float64)
                    
                var_ii = np.maximum(0, (sum_x2 / n_frames) - (sum_x / n_frames)**2)
                map_ii = sum_m / n_frames

            else:
                # Map global indices into local trajectory space
                traj_indices = np.searchsorted(cum_lengths, ind, side='right') - 1
                local_indices = ind - cum_lengths[traj_indices]
                
                contact_ii = np.empty((len(ind), d), dtype=traj[0].dtype)
                for t_idx in np.unique(traj_indices):
                    mask = (traj_indices == t_idx)
                    contact_ii[mask] = traj[t_idx][local_indices[mask]]
                
                # Transpose to preserve the original (d x m) internal assumption    
                contact_ii = contact_ii.T.copy()

                var_ii = np.var(contact_ii, axis=1)

                if mode == 'frequency':
                    if isinstance(tol, float):
                        contact_ii = (contact_ii <= tol).astype(float)

                map_ii = np.mean(contact_ii, axis=1)

            ii_mat = np.zeros((n_res, n_res))
            ii_var = np.zeros((n_res, n_res))

            for jj in range(map_ii.shape[0]):
                ii_mat[ind_rows[jj], ind_cols[jj]] = map_ii[jj]
                ii_var[ind_rows[jj], ind_cols[jj]] = var_ii[jj]

            ii_mat = ii_mat + ii_mat.T
            ii_var = ii_var + ii_var.T

            results_mean.append(ii_mat)
            results_variance.append(ii_var)

        if show_plot or save_plot:
            for ii, (ii_mat, ii_var) in enumerate(zip(results_mean, results_variance)):
                for m, ending in zip([ii_mat, ii_var], ['.png', '_variance.png']):


                    
                    norm = None
                    if mode == 'frequency':
                        vmin = 0
                        vmax = 1 if ending == '.png' else np.max(results_variance) 

                        if 'var' in ending:
                            norm = LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
                        
                        else:
                            norm = Normalize(vmin=0, vmax=vmax)

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.set_aspect('equal')
                    im = ax.imshow(m, interpolation=interpolation, norm=norm, cmap=cmap, origin='lower', aspect='equal')
                    ax.tick_params(axis='both', which='major', labelsize=14)

                    plt.xticks(np.arange(res_id_offset, res_id_offset + m.shape[1], labelstep))
                    plt.yticks(np.arange(res_id_offset, res_id_offset + m.shape[0], labelstep))
                    ax.tick_params(which='both', width=3, length=4)
                    # Make colorbar the same height as imshow
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=14) 
                    fig.tight_layout()

                    if headlines is not None:
                        ax.set_title(headlines[ii], fontsize=25)

                    fig.tight_layout()

                    if show_plot:
                        fig.show()

                    if save_plot:
                        # print('Saving contact map for state', ii)
                        fig.savefig(final_path(ii, ending), dpi=600)
                        save_to_file(ii_mat, final_path(ii, '.npy'))
                        save_to_file(ii_var, final_path(ii, '_variance.npy'))

                    plt.close()

        if return_mat:
            return np.array(results_mean), np.array(results_variance)


    def compute_state_colors(self, cut, include_transition=False, min_energy=None, max_energy=None, colormap='viridis'):
        """
        Map each metastable state's free energy to a color (self.state_colors[cut]),
        for use in generate_transition_network's node coloring.

        Parameters:
            cut: see compute_metastable_indices.
            include_transition: if True, also assign a color to the residual
                ("transition") state; otherwise only the nev real states get
                a color, matching generate_transition_network's default.
            min_energy, max_energy: fixed bounds for the color normalization
                (kJ/mol); default to the min/max free energy actually observed
                across states, so colors are comparable across cuts/systems
                only if you pass explicit bounds.
            colormap: any matplotlib colormap name.
        """
        if not cut in self.free_energy_per_state.keys():
            self.compute_state_statistics(cut=cut)
        free_energy_values = self.free_energy_per_state[cut]
        if not include_transition:
            free_energy_values = free_energy_values[:-1]

        if min_energy is not None:
            self.min_energy = min_energy
        else:
            self.min_energy = min(free_energy_values)

        if max_energy is not None:
            self.max_energy = max_energy
        else:
            self.max_energy = max(free_energy_values)

        normalized_energies = [(e - self.min_energy) / (self.max_energy - self.min_energy) for e in free_energy_values]

        self.colormap = colormaps[colormap]
        self.state_colors[cut] = [self.colormap(e) for e in normalized_energies]


    def compute_transition_matrix(self, cut, include_transition=False):
        """
        Build a state-to-state transition-count/probability matrix from the
        discretized PCCA state sequence self.markov[cut], and store it as
        self.transition_counts[cut] / self.transition_matrix[cut] (the
        latter row-stochastic, in percent, rounded to 3 decimals).

        Known caveats (not corrected here, by design):
        - The transition counts are estimated at a fixed 1-frame lag in the
          discretized state sequence, NOT at self.lag. Do not interpret this
          as "the transition matrix at lag=self.lag".
        - With include_transition=False (the default), frames belonging to
          the residual/transition region are spliced out of each
          trajectory's sequence before counting, which can fabricate
          apparent transitions between states that were never actually
          adjacent in time. Consider include_transition=True, or treat
          these numbers cautiously, until this is reviewed.

        Returns (countMatrix, markovMatrix, counts) so callers that also
        need the deeptime TransitionCountModel (e.g. for state_histogram)
        don't have to recompute it.
        """
        if self.traj_lengths is None:
            # Older PCCAoperator pickles (saved before traj_lengths was
            # persisted in run_simplex) may still have traj_lengths=None.
            # Fall back to the single-trajectory assumption run_simplex
            # itself uses.
            self.traj_lengths = [len(self.markov[cut])]

        nev = self.nev
        markovTraj = self.markov[cut]
        markovTrajs = [markovTraj[i:i+length] for i, length in zip(np.cumsum([0] + self.traj_lengths[:-1]), self.traj_lengths)]

        estimator = TransitionCountEstimator(lagtime=1, count_mode="sliding")

        if include_transition:
            counts = estimator.fit(markovTrajs).fetch_model()
        else:
            markovTrajs_no_transitions = [np.delete(traj, np.where(traj == nev)) for traj in markovTrajs]
            counts = estimator.fit(markovTrajs_no_transitions).fetch_model()

        countMatrix = counts.count_matrix
        rowSums = np.sum(countMatrix, axis=1)
        markovMatrix = np.round(100 * np.dot(np.diag(1 / rowSums), countMatrix), 3)

        self.transition_counts[cut] = countMatrix
        self.transition_matrix[cut] = markovMatrix

        return countMatrix, markovMatrix, counts

    def generate_transition_network(self,
                                    cut=0.6,
                                    include_transition=False,
                                    remove_self_edges=True,
                                    save_plot=True,
                                    show_plot=True,
                                    colors=None,
                                    path=None,
                                    path_extension='',
                                    include_legend=True,
                                    figsize=(8, 10),
                                    cbar_orientation='horizontal',
                                    return_path=False,
                                    cbar_bounds=None,
                                    alpha=0.8,
                                    factors=[1, 1, 1, 1]):
        """
        Draw a directed graph of metastable-state transitions: one node per
        state (sized by state population, colored by free energy), edges
        labeled with the transition matrix's per-state-pair percentages
        (see compute_transition_matrix, which this calls internally).

        Parameters:
            cut, include_transition: see compute_metastable_indices /
                compute_transition_matrix.
            remove_self_edges: if True, hide self-transition edges/loops and
                label each node with its occupancy percentage instead of its
                index; if False, show self-loops (labeled with their
                percentage) and label nodes by index.
            save_plot / show_plot: whether to write the figure to disk / call
                plt.show() (the figure is always closed afterward either way).
            colors: per-state colors, e.g. from compute_state_colors(cut).
                If not given, falls back to a broken placeholder
                (['C{i}' for i in ...] -- note the missing f-string, so every
                state ends up the same literal color); always pass colors=
                explicitly rather than relying on this default.
            path: directory to save the plot in (with save_plot=True); falls
                back to self.saving_dir if not given.
            path_extension: extra text inserted into the saved filename.
            cbar_bounds: (min, max) free-energy bounds for the colorbar;
                defaults to (self.min_energy, self.max_energy) as set by
                compute_state_colors.
            factors: [x_min, x_max, y_min, y_max] multipliers on the node
                radius used to pad the axis limits around the graph layout.
            return_path: if True, also return the path the plot was saved to.
        """
        labelsize = 8
        nodesize_scaling = 1.2

        if colors is not None:
            self.state_colors[cut] = colors
        else:
            self.state_colors[cut] = ['C{i}' for i in range(self.nev + 1)]

        nev = self.nev
        nStates = nev + 1 if include_transition else nev

        countMatrix, markovMatrix, counts = self.compute_transition_matrix(cut, include_transition=include_transition)

        # build graph G
        G = nx.from_numpy_array(countMatrix, create_using=nx.DiGraph())
        pos = nx.circular_layout(G)
        
        # NODES (size and color)
        hist = counts.state_histogram
        node_sizes = [np.log(hist[i] / 1000 + 3) * np.e ** 6.3 for i in range(nStates)]
        node_sizes = [nodesize_scaling * s for s in node_sizes]

        # EDGES (which, shape, label)
        edge_labels = {}
        edgeAttributes = collections.defaultdict(dict)
        for u, v in G.edges():
            edge_labels[(u, v)] = f"{markovMatrix[u, v]}%"
            edgeAttributes[(u,v)].update({'percentage': markovMatrix[u, v]})

        nx.set_edge_attributes(G, values=edgeAttributes)

        # Draw graph 
        fig, ax = plt.subplots(figsize=figsize)

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='w', alpha=0)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=self.state_colors[cut], alpha=alpha)

        onlySelfG = G.edge_subgraph(list(nx.selfloop_edges(G)))
        nonSelfG = G.edge_subgraph([edge for edge in G.edges() if edge[0] != edge[1]])

        # draw one-way edges as straight lines
        oneWayEdges = [edge for edge in nonSelfG.edges() if edge[::-1] not in nonSelfG.edges()]
        twoWayEdges = [edge for edge in nonSelfG.edges() if edge[::-1] in nonSelfG.edges()]

        oneWayG = nonSelfG.edge_subgraph(oneWayEdges)
        twoWayG = nonSelfG.edge_subgraph(twoWayEdges)

        selfEdgeLabels = {edge: edge_labels[edge] for edge in list(nx.selfloop_edges(G))}
        oneWayEdgeLabels = {edge: edge_labels[edge] for edge in oneWayEdges}
        twoWayEdgeLabels = {edge: edge_labels[edge] for edge in twoWayEdges}

        if remove_self_edges:
            nodeLabels = {i: f'{np.round(100 * self.freq_per_state[cut][i], 2)} %' for i in G.nodes()}

        else:
            nodeLabels = {i: i for i in G.nodes()}
            nx.draw_networkx_edges(onlySelfG,
                                   pos=pos,
                                   node_size=node_sizes,
                                   edge_color='gray',
                                   alpha=0.7,
                                   connectionstyle="arc3,rad=0.6",
                                   arrowstyle="-|>",
                                   arrowsize=30
                                   )

            nx.draw_networkx_edge_labels(onlySelfG,
                                         pos=pos,
                                         edge_labels=selfEdgeLabels,
                                         connectionstyle="arc3,rad=0.6",
                                         )
        
        # for dark node colors, use white font
        nodelabels_white = {k: v for k, v in nodeLabels.items() if self.freq_per_state[cut][k] >= 0.14}
        nodelabels_black = {k: v for k, v in nodeLabels.items() if self.freq_per_state[cut][k] < 0.14}

        nx.draw_networkx_labels(G, pos, nodelabels_black, alpha=0.9, font_color='black', font_size=labelsize)
        nx.draw_networkx_labels(G, pos, nodelabels_white, alpha=0.9, font_color='white', font_size=labelsize)


        nx.draw_networkx_edges(oneWayG,
                               pos=pos,
                               node_size=node_sizes,
                               edge_color='gray',
                               alpha=0.7,
                               arrowstyle="-|>",
                               arrowsize=15
                               )

        nx.draw_networkx_edges(twoWayG,
                               pos=pos,
                               node_size=node_sizes,
                               edge_color='gray',
                               alpha=0.7,
                               connectionstyle="arc3,rad=0.2",
                               arrowstyle="-|>",
                               arrowsize=15
                               )

        nx.draw_networkx_edge_labels(oneWayG,
                               pos=pos,
                               edge_labels=oneWayEdgeLabels,
                               font_size=labelsize
                               )

        nx.draw_networkx_edge_labels(twoWayG,
                               pos=pos,
                               edge_labels=twoWayEdgeLabels,
                               connectionstyle="arc3,rad=0.2",
                               font_size=labelsize
                               )

        
        if cbar_bounds is None:
            cbar_bounds = (self.min_energy, self.max_energy)
        else:
            self.min_energy = cbar_bounds[0]
            self.max_energy = cbar_bounds[1]

        if add_colorbar := (self.min_energy is not None and self.max_energy is not None and self.colormap is not None):
            sm = plt.cm.ScalarMappable(cmap=self.colormap, norm=Normalize(vmin=cbar_bounds[0], vmax=cbar_bounds[1]))
            sm.set_array([])
                
            if 20 < int(self.max_energy) <= 25:
                ticks = [5, 10, 15, 20]
            elif 15 < int(self.max_energy) <= 20:
                ticks = [5, 10, 15]
            elif 10 < int(self.max_energy) <= 15:
                ticks = [5, 10]
            else:
                ticks = np.linspace(1, int(self.max_energy) + 1, int(self.max_energy) + 1)
                
            cbar = fig.colorbar(sm, ax=ax, ticks=ticks, orientation=cbar_orientation, pad=0.15)
            cbar.solids.set(alpha=alpha)
            cbar.set_label(label="Free Energy (kJ/mol)")

            plt.axis("off")

        if include_legend:
            import matplotlib.patches as mpatches
            # sort legend by free energy
            # l = np.array(np.argsort(self.free_energy_per_state[cut][:-1]))
            # patches = [mpatches.Patch(color=x,label=y) for (x,y) in zip(np.array(node_colors)[l], [f'State {i}' for i in range(nev)])]
            # plt.legend(handles=patches)
            # add legend state{i}
            patches = [mpatches.Patch(color=self.state_colors[cut][i], label=f'State {i}') for i in range(nev)]
            plt.legend(handles=patches, loc='upper right', fontsize=labelsize)


        # determine the whitespace around the graph to make space for renders
        # Get x/y coordinates of all nodes
        x_vals, y_vals = zip(*pos.values())
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)

        max_node_size = max(node_sizes)
        node_radius = np.sqrt(max_node_size) / 250  

        factor_x_min, factor_x_max, factor_y_min, factor_y_max = factors
        ax.set_xlim(x_min - factor_x_min * node_radius, x_max + factor_x_max * node_radius)
        ax.set_ylim(y_min - factor_y_min * node_radius, y_max + factor_y_max * node_radius)
        
        plt.tight_layout()
        
        if save_plot is True:
            base_dir = Path(path) if path is not None else self.saving_dir
            if base_dir is None:
                ValueError('Please provide saving_dir for PCCA results.')
            saving_dir = base_dir
            saving_dir.mkdir(exist_ok=True, parents=True)

            filename = f'transition_network_cut={cut}'
            filename += path_extension
            filename += '_no_legend' if not include_legend else ''

            if include_transition:
                filename += '_with_transition.png'
            else:
                filename += '_no_transition.png'

            plt.savefig(saving_dir / Path(filename), dpi=600)

        if show_plot:
            plt.show()
        
        plt.close()
        
        if return_path:
            return saving_dir / Path(filename)




def _pcca_connected_isa(eigenvectors, n_clusters):
    """
    PCCA+ spectral clustering method using the inner simplex algorithm.

    Clusters the first n_cluster eigenvectors of a transition matrix in order to cluster the states.
    This function assumes that the state space is fully connected, i.e. the transition matrix whose
    eigenvectors are used is supposed to have only one eigenvalue 1, and the corresponding first
    eigenvector (evec[:,0]) must be constant.

    Parameters
    ----------
    eigenvectors : ndarray
        A matrix with the sorted eigenvectors in the columns. The stationary eigenvector should
        be first, then the one to the slowest relaxation process, etc.

    n_clusters : int
        Number of clusters to group to.

    Returns
    -------
    chi : ndarray (n x m)
        A matrix containing the probability or membership of each state to be assigned to each cluster.
        The rows sum to 1.

    rot_mat : ndarray (m x m)
        A rotation matrix that rotates the dominant eigenvectors to yield the PCCA memberships, i.e.:
        chi = np.dot(evec, rot_matrix)

    References
    ----------
    [1] P. Deuflhard and M. Weber, Robust Perron cluster analysis in conformation dynamics.
        in: Linear Algebra Appl. 398C M. Dellnitz and S. Kirkland and M. Neumann and C. Schuette (Editors)
        Elsevier, New York, 2005, pp. 161-184

    """
    (n, m) = eigenvectors.shape

    # do we have enough eigenvectors?
    if n_clusters > m:
        raise ValueError("Cannot cluster the (" + str(n) + " x " + str(m)
                         + " eigenvector matrix to " + str(n_clusters) + " clusters.")

    # check if the first, and only the first eigenvector is constant
    diffs = np.abs(np.max(eigenvectors, axis=0) - np.min(eigenvectors, axis=0))
    assert diffs[0] < 1e-6, "First eigenvector is not constant. This indicates that the transition matrix " \
                            "is not connected or the eigenvectors are incorrectly sorted. Cannot do PCCA."
    assert diffs[1] > 1e-6, "An eigenvector after the first one is constant. " \
                            "Probably the eigenvectors are incorrectly sorted. Cannot do PCCA."

    # local copy of the eigenvectors
    c = eigenvectors[:, list(range(n_clusters))]

    ortho_sys = np.copy(c)

    max_dist = 0.0

    # representative states
    ind = np.zeros(n_clusters, dtype=np.int32)

    # select the first representative as the most outlying point
    for (i, row) in enumerate(c):
        if np.linalg.norm(row, 2) > max_dist:
            max_dist = np.linalg.norm(row, 2)
            ind[0] = i

    # translate coordinates to make the first representative the origin
    ortho_sys -= c[ind[0], None]

    # select the other m-1 representatives using a Gram-Schmidt orthogonalization
    for k in range(1, n_clusters):
        max_dist = 0.0
        temp = np.copy(ortho_sys[ind[k - 1]])

        # select next farthest point that is not yet a representative
        for (i, row) in enumerate(ortho_sys):
            row -= np.dot(np.dot(temp, np.transpose(row)), temp)
            distt = np.linalg.norm(row, 2)
            if distt > max_dist and i not in ind[0:k]:
                max_dist = distt
                ind[k] = i
        ortho_sys /= np.linalg.norm(ortho_sys[ind[k]], 2)

    # obtain transformation matrix of eigenvectors to membership matrix
    rot_mat = np.linalg.inv(c[ind])
    # print "Rotation matrix \n ", rot_mat

    # compute membership matrix
    chi = np.dot(c, rot_mat)

    return chi, rot_mat