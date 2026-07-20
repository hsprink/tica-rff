# implements transfer operator classes for Koopman operator approximation
# includes TICA and RFF classes inheriting from base class TransferOperator
# contains functions to create/load operators and helper functions for data preparation 

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from scipy.linalg import svd, eig, eigh
from scipy.sparse.linalg import eigsh
import gc
from util import *
from basis import *

class TransferOperator(ABC):
    # base class for transfer operators
    # main functions:
    #   function fit approximates koopman operator from data
    #   function transform projects data onto leading eigenvectors
    #   function get_timescales computes implied timescales from eigenvalues
    # TICA and RFF inherit from this class

    def __init__(self, 
                 lag=1,
                 nev='all',
                 dt=None,
                 cplx=False
                 ):
        
        self.name = self.__class__.__name__
        self.lag = lag
        self.nev = nev
        self.dt = dt
        self.dim = None
        self.cplx = cplx
        self.trainingDataMean = None

        if not (isinstance(nev, int) or nev == 'all'):
            raise ValueError('nev should be all or an int.')

        self.classDir = Path(self.name)

        self.fileName = None
        self.set_file_name()

        self.operator = None
        self.eigenvalues = None
        self.timescales = None
        self.right_eigenvectors = None
        self.leftEigenvectors = None

        self.attrs_to_save = {'dt', 'lag', 'nev', 'eigenvalues', 'timescales', 'right_eigenvectors', 'trainingDataMean', 'whitening_transform'}
        self._update_attrs_to_save()

    def getDir(self, baseDir):
        return baseDir / self.classDir

    # def getPath(self, baseDir):
    #     return self.getDir(baseDir) / self.fileName

    @property
    def fitted(self):
        """Check if the operator has been fitted.
        """
        try:
            return self.eigenvalues is not None
        except (ValueError, AttributeError):
            return False

    @abstractmethod
    def set_file_name(self):
        pass

    @abstractmethod
    # this function can be used in child classes to add additional attributes to the set of attributes that are saved when saving the operator to file
    def _update_attrs_to_save(self):
        pass

    def fit(self,
                X,
                Y=None,
                remove_mean=False,
                mean=None,
                symmetrize=False,
                mode_whitening='cov',
                tol=None,       
                rank_whitening=None,
                batchwise=False,
                load_intermediate_results=False,
                save_intermediate_results=False,
                dir_base=None,
                n_ev=None,
                ):

        """
        params:

        X: array (m, d) or list of arrays (m_i, d)
        Y: array (m, d) or list of arrays (m_i, d) or None, if None, Y is set to X shifted by lag
        remove_mean: bool, whether to remove mean from data before fitting
        symmetrize: bool, whether to symmetrize covariance matrices before eigendecomposition
        mode_whitening: 'cov' or 'svd', whether to compute whitening transformation based on covariance eigendecomposition or SVD of data
        tol: float, tolerance for eigenvalue cutoff in whitening step (relative to largest eigenvalue)
        rank_whitening: int or None, maximal rank to retain in whitening step, if None, cut-off is determined solely by tol
        batchwise: bool, whether to compute covariance matrices in batches (only implemented for mode='cov')
        load_intermediate_results: bool, whether to try load precomputed results/matrices, requires dir_base. 
        save_intermediate_results: bool, whether to save results, requires dir_base. 
        dir_base: str or Path, base directory to load from / save to
        n_ev: int or None, number of eigenvalues/eigenvectors to compute in eigendecomposition of whitened lagged covariance matrix. If None, all eigenvalues/eigenvectors are computed.


        Apply AMUSE algorithm:

        1. Split data into two parts, separated by lag
            --> X, Y,   size (n, T-lag) := (n, m)

                NB: The goal is an eigendecomposition of M_TICA := C_0^-1 @ C_lag:
                    C_* denote the instantaneous/time-lagged covariance matrix:
                    C_0 = X @ X^T       C_lag = X @ Y^T

        2. Compute whitening transformation L:
            Whitened data WX is obtained using the reduced SVD of X :
            X = U @ s @ V^T --> WX = s^-1 @ U^T @ X = L^T @ X  (size (r, m))
                            --> WY = s^-1 @ U^T @ Y = L^T @ Y
            where L := U @ s^-1 (size (n, r))

            WX @ WX^T = L^T @ X @ X^T L = s^-1 U^T U s V^T V s U^T U s^-1 = Id

        3. Compute R := WX @ WY^T       (size (r, r))
            Note that (using SVD of X) WX = L^T @ X = V^T
            Thus R = L^T @ X @ Y^T @ L = V^T @ Y^T @ L


        4. Eigenvalue decomposition R @ W = d @ W.
            Sort and keep only the largest nev eigenvalues and corresponding eigenvectors.


        5. TICA-coordinates are given by Z = L @ W  (size (n, r) - reduced eVecs lifted to original space)

                ZT Z = WT LT L W = WT s^-2 W = (s-1 W)^T (s-1 W)

            Note (using SVD)   (X @ X^T)^-1 @ X = U @ s^-2 @ U^T @ U @ s @ V^T
                                                = U @ s^-1 @ V^T
                                                = L @ WX

            --> M_TICA @ Z =     C_0^-1   @  C_lag  @   Z
                            = (X @ X^T)^-1 @ X @ Y^T @ L @ W
                            = (X @ X^T)^-1 @ X @   WY^T  @ W
                            =    L  @    WX    @   WY^T  @ W
                            =    L  @          R         @ W
                            = d @ L @ W
                            = d @ Z

            --> (d, Z) yields an eigendecomposition of M_TICA.

            """
        # TODO take care of mean..?!?!
        assert mode_whitening in ['cov', 'svd'], "mode_whitening should be either 'cov' or 'svd'."
        if tol is None:
            if mode_whitening == 'cov':
                tol = 1e-8
            elif mode_whitening == 'svd':
                tol = 1e-4
            

        # TODO currently it is not possible to save only without trying to load.
        if save_intermediate_results:
            load_intermediate_results = True

        if load_intermediate_results:
            if dir_base is None:
                print('dir_base is None. should be provided if load_intermediate_results or save_intermediate_results is True.')
        else:
            # else set dir_base to None
            dir_base = None
            print('Fit mode: not loading or saving intermediate results, computing everything from scratch...')

        # dir_base holds lag-independent results (C0, mean); dir_base_lag holds
        # lag-dependent results (Ct, Ct_whitened, eigendecomposition of Ct_whitened).
        dir_base_lag = None
        if dir_base is not None:
            dir_base = Path(dir_base)
            dir_base_lag = dir_base / f'lag={self.lag}'
            if save_intermediate_results:
                dir_base_lag.mkdir(parents=True, exist_ok=True)

        # initialize other variables to come
        C0, Ct = None, None
        needs_C0 = True
        needs_Ct = True
        W = None
        Ct_whitened = None

        # check if s, U-1 exist
        s, U = load_or_compute_eigendecomposition(
            M=C0, 
            dir_base=dir_base,
            filename_without_extension_string='C0',
            mode='eigh',
            save=save_intermediate_results,
            verbose=load_intermediate_results,
        )

        needs_C0 = (s is None or U is None)
        # if load_intermediate_results:
        #     print('not found.') if needs_C0 else print('exists.')
        
        if not needs_C0:
            ##### U, s-1 exist, check if Ct is required
            # if load_intermediate_results:
            #     print('Now checking if whitened lagged covariance matrix is available on disk...', end=' ')

            s, U = filter_eigenvalues_and_vectors(s, U, tol=tol)
            W = compute_whitening_transform(U=U, s=s, rank=rank_whitening)

            # try loading Ct whitened from file, whitout computing it from Ct,
            Ct_whitened = load_or_compute_Ct_whitened(
                W=W,
                Ct=None,
                lag=self.lag,
                dir_base=dir_base_lag,
                symmetrize=symmetrize,
                save_results=save_intermediate_results,
                verbose=load_intermediate_results
            )
            needs_Ct = (Ct_whitened is None)
            # if load_intermediate_results:
            #     print('exists.') if not needs_Ct else print('not found.')

            # if needs_Ct is False, we have all the data we need to compute the eigenvalues/eigenvectors of Ct_whitened,
            # i.e. skip to eigendecomposition of Ct_whitened, otherwise we need to compute Ct and then whiten it before eigendecomposition.
            # However, if we need Ct, lets try to load or compute now
            
            if needs_Ct:
                if load_intermediate_results:
                    print('need to load or compute Ct.', end=' ')
                    
                if batchwise:
                    Ct = load_or_compute_covariance_from_batches(
                        dir_base=dir_base,
                        dir_base_lag=dir_base_lag,
                        mmap_trajs=X,
                        symmetrize=symmetrize,
                        lag=self.lag,
                        remove_mean=remove_mean,
                        mean=mean,
                        save_results=save_intermediate_results,
                        which='lagged',
                        batch_size=1000)

                else:
                    Ct = load_or_compute_covariance_in_memory(
                        dir_base=dir_base,
                        dir_base_lag=dir_base_lag,
                        X=X,
                        Y=Y,
                        lag=self.lag,
                        remove_mean=remove_mean,
                        mean=mean,
                        symmetrize=symmetrize,
                        # load_results=load_intermediate_results,
                        save_results=save_intermediate_results,
                        which='lagged',
    )
                    # X, Y = prepare_data(X, Y, remove_mean, lag=self.lag)
                    # Ct = compute_covariances(X, Y, symmetrize=symmetrize)[1]

                needs_Ct = False

                Ct_whitened = load_or_compute_Ct_whitened(
                    W=W,
                    Ct=Ct,
                    lag=self.lag,
                    dir_base=dir_base_lag,
                    symmetrize=symmetrize,
                    save_results=save_intermediate_results,
                    verbose=load_intermediate_results
                )
            
            # now we have all the data we need to compute the eigenvalues/eigenvectors of Ct_whitened,
                  

        elif needs_C0:
            # at this point we know that we need to load or compute C0. And probably also Ct. So for now assume that this is always the case.
            
            if load_intermediate_results:
                print("need C0: ", needs_C0, "need Ct: ", needs_Ct)
                
            
            if batchwise:
                # if batchwise, we load or compute in batches
                # whether we try loading results is determined by dir_base kwarg
                C0, Ct, mean = load_or_compute_covariance_from_batches(
                    dir_base=dir_base,
                    dir_base_lag=dir_base_lag,
                    mmap_trajs=X,
                    symmetrize=symmetrize,
                    lag=self.lag,
                    remove_mean=remove_mean,
                    mean=mean,
                    return_mean=True,
                    save_results=save_intermediate_results,
                    which='both',
                    batch_size=1000)

            else:
                C0, Ct, mean = load_or_compute_covariance_in_memory(
                    dir_base=dir_base,
                    dir_base_lag=dir_base_lag,
                    X=X,
                    Y=Y,
                    lag=self.lag,
                    remove_mean=remove_mean,
                    mean=mean,
                    return_mean=True,
                    symmetrize=symmetrize,
                    # load_results=load_intermediate_results,
                    save_results=save_intermediate_results,
                    which='both',
    )

            # if C0 was required earlier, then we have not yet computed the eigendecomposition of C0, so we need to do that now.
            s, U = load_or_compute_eigendecomposition(
                M=C0,
                dir_base=dir_base,
                filename_without_extension_string='C0',
                mode='eigh',
                save=save_intermediate_results,
                verbose=False
            )
            s, U = filter_eigenvalues_and_vectors(s, U, tol=tol)

            W = compute_whitening_transform(U=U, s=s, rank=rank_whitening)
            Ct_whitened = load_or_compute_Ct_whitened(
                W=W,
                Ct=Ct,
                lag=self.lag,
                dir_base=dir_base_lag,
                symmetrize=symmetrize,
                save_results=save_intermediate_results,
                verbose=load_intermediate_results
            )

        del C0, Ct
        gc.collect()

        rank_extension = f'_rank={rank_whitening}' if rank_whitening is not None else ''
        sym_extension = '_sym' if symmetrize else ''
        eigvals, eigvecs = load_or_compute_eigendecomposition(
            M=Ct_whitened,
            dir_base=dir_base_lag,
            filename_without_extension_string=f'Ct_whitened_lag={self.lag}{rank_extension}{sym_extension}',
            mode='eigh' if symmetrize else 'eig',
            n_ev=n_ev,
            save=save_intermediate_results,
            verbose=load_intermediate_results
        )
    
        eigvals, eigvecs = filter_eigenvalues_and_vectors(eigvals, eigvecs, tol=tol)
        
        self.Ct_whitened = Ct_whitened
        self.eigenvalues = eigvals[:rank_whitening]
        self.whitening_transform = W
        self.right_eigenvectors = W @ eigvecs[:, :rank_whitening]


    def get_timescales(self, N_non_trivial=None, skip_trivial=False, dt=None):
        """
        Compute implied timescales -dt * lag / log(eigenvalue) from the fitted
        eigenvalues.

        Parameters:
            N_non_trivial: if given, only return this many timescales after the
                (optional) leading-eigenvalue skip.
            skip_trivial: if True, drop the leading eigenvalue (normally ~1,
                corresponding to the stationary/non-decaying process).
            dt: physical time per frame; defaults to self.dt if not given.

        Returns:
            array of real-valued timescales, same units as dt.
        """
        if dt is None:
            dt = self.dt
        if dt is None:
            raise ValueError('need dt for timescale computation.')

        a = 0
        b = None

        if skip_trivial:
            a = 1

        if N_non_trivial:
            b = a + N_non_trivial

        if self.timescales is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                self.timescales = -dt * self.lag / np.log(self.eigenvalues)

        return np.real(self.timescales[a:b])
    


    def transform(self, data, mean=None, weight_by_eval=False, dim=None, conjugate=False, batch_size=None):
        """
        Project data onto the leading right eigenvectors.

        Parameters:
            data: array (m, d) or list of arrays (m_i, d).
            mean: mean to subtract before projecting (passed through rather than
                stored, since data may be projected in batches).
            weight_by_eval: if True, scale each eigenvector by its eigenvalue
                before projecting (kinetic map).
            dim: number of leading eigenvectors to project onto (None = all).
            conjugate: if True, use the complex-conjugate eigenvectors.
            batch_size: if given, project in batches of this many rows to bound
                memory use.

        Returns:
            projected data, same list/array structure as the input, or None if
            projection fails (e.g. eigenvectors not yet fitted).
        """
        to_project_on = self.right_eigenvectors[:, :dim].copy()

        if weight_by_eval:
            to_project_on *= self.eigenvalues[None, :dim]

        if conjugate:
            to_project_on = to_project_on.conj()

        try:
            projected_data = project_data_onto_vectors(data=data, V=to_project_on, dim=dim, mean=mean, batch_size=batch_size)
        except:
            projected_data = None
            print('Could not project data onto eigenvectors. Returning None.')
        return projected_data


    def normalize_eigenvectors(self):
        # normalize right eigenvectors to have unit norm
        self.right_eigenvectors /= np.linalg.norm(self.right_eigenvectors, axis=0)


    def get_kinetic_variance(self, cut_off_idx=0):
        # compute kinetic variance explained by leading eigenvalues

        sorted_eigenvalues = self.eigenvalues[np.argsort(np.abs(self.eigenvalues))[::-1]]
        if cut_off_idx == 0:
            while sorted_eigenvalues[cut_off_idx] > 0:
                cut_off_idx += 1

        squared_evals = self.eigenvalues[:cut_off_idx] ** 2
        denominator = np.sum(squared_evals)
        kinetic_variance = np.cumsum(squared_evals) / denominator

        return kinetic_variance

class TICA(TransferOperator):
    # TICA class inheriting from TransferOperator

    @staticmethod
    def getDir(nev):
        return f'nev={nev}'

    def set_file_name(self):
        self.fileName = self.getDir(nev=self.nev)
        self.path = self.classDir / self.fileName

    def _update_attrs_to_save(self):
        return
    

class EDMD(TransferOperator):
    """
    Extended DMD: fits TransferOperator.fit's AMUSE/TICA machinery on data
    lifted through a basis function Phi (e.g. random Fourier features) rather
    than on the raw data directly, approximating the Koopman operator in the
    lifted feature space.
    """

    def __init__(self,
                lag=1,
                 nev='all',
                 dt=None,
                 Phi=None):

        self.Phi = Phi # callable that takes data and returns lifted data, ndarray (m, d) to ndarray (m, n_features)

        super().__init__(lag=lag, nev=nev, dt=dt)
    
    def set_file_name(self):
        pass

    def _update_attrs_to_save(self):
        pass

    def fit(self, data, **params):
        liftedData = self.lift(data)
        super().fit(X=liftedData, **params)

    def lift(self, data):
        if isinstance(data, list):
            return [self.Phi(x) for x in data]
        return self.Phi(data)

    def transform(self, data, dim=None, weight_by_eval=False, conjugate=False):
        return super().transform(self.lift(data), dim=dim, weight_by_eval=weight_by_eval, conjugate=conjugate)


# ---- helper functions related to transfer operators ----
def get_lagtimes(parameters, compute_timelags=False):
    # helper function to get lagtimes from parameter object containing running parameters

    if compute_timelags:
        #  get initial and final lag and number of lagtime
        initialLag = parameters.initialLag
        finalLag = parameters.finalLag
        N_timelags = parameters.N_timelags

        # compute timelags as equidistant points between initialLag and finalLag
        range_to_cover = finalLag - initialLag
        avg_stride = range_to_cover / (N_timelags - 1)
        timelags = []
        for i in range(N_timelags):
            timelags.append(int(initialLag + i * avg_stride))

    else:
        # use provided timelags
        timelags = np.array(parameters.timelags)
        print('Using provided timelags:', timelags)

    return timelags


def split_by_lag(data, lag):

    """
    Parameters:
        data: array (m, d) or list of arrays (m_i, d)
        lag: int

    Returns:
        X: array (m-lag, d) or (sum of (m_i-lag), d)
        Y: array (m-lag, d) or (sum of (m_i-lag), d)
    """

    if isinstance(data, list) or isinstance(data, tuple) or (isinstance(data, np.ndarray) and data.ndim == 3):
        X = np.vstack([traj[:-lag] for traj in data])
        Y = np.vstack([traj[lag:] for traj in data])
    elif isinstance(data, np.ndarray):
        X = data[:-lag]
        Y = data[lag:]
    return X, Y








# from tica_large_systems

def load_or_compute_mean(filepath, data, elementwise=False, batch_size=None):
    # helper function to load mean from file if it exists, otherwise compute mean from data and save to file
    if os.path.exists(filepath):
        print('loading mean from file', end=' ')
        mean = load_from_file(filepath)
        print('done.')
    else:
        print('computing mean from data and saving', end=' ')
        mean = compute_mean(data, elementwise=elementwise, batch_size=batch_size)
        save_to_file(obj=mean, filepath=filepath)
        print('done.')

    return mean


def compute_sum_and_count(traj, batch_size=None):
    """
    Helper subroutine to compute the sum and row-count of a trajectory (ndarray of shape (m, d)) in batches if batch_size is provided.
    Can be reused by other statistical functions (like variance/std).
    """
    m = traj.shape[0]
    
    if batch_size is None:
        return np.sum(traj, axis=0), m
        
    total_sum = 0
    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        total_sum += np.sum(traj[start:end], axis=0)
        
    return total_sum, m



def compute_mean(data, elementwise=False, batch_size=None):
    '''
    Parameters:
    -----------
    data: array (m, d) or list of arrays (m_i, d)
        data to compute mean from.
    elementwise: bool
        If True, mean is computed per-trajectory.
        If False, mean is computed globally across all data points.
    batch_size: int or None,
        If int, mean is computed in chunks of size batch_size.
        If None, mean is computed from all data at once.
    '''

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = [data]
        elementwise = False
    if isinstance(data, list) or isinstance(data, tuple) or (isinstance(data, np.ndarray) and data.ndim == 3):
        for traj in data:
            if not isinstance(traj, np.ndarray):
                raise ValueError('All trajectories should be numpy arrays.')
    else:
            raise ValueError('Data should be a numpy array or a list of numpy arrays.')

    if elementwise:
        means = []
        for traj in data:
            traj_sum, traj_m = compute_sum_and_count(traj, batch_size)
            traj_mean = traj_sum / traj_m
            means.append(traj_mean.reshape(-1, 1))
        return means
        
    else:
        global_sum = 0
        global_m = 0
        
        for traj in data:
            traj_sum, traj_m = compute_sum_and_count(traj, batch_size)
            global_sum += traj_sum
            global_m += traj_m
            
        global_mean = global_sum / global_m
        return global_mean.reshape(-1, 1)




def remove_mean_fun(data, mean=None, return_mean=False, elementwise=False):
    """ Helper function to remove mean from data

    Parameters:
    -----------
    data: array (m, d) or list of arrays (m_i, d):
        data to remove mean from
    mean: array (d, 1), (1, d), (d,) or None
        mean to remove from data. If None, mean is computed from data.
    elementwise: bool
        whether to remove mean from each trajectory separately (True) 
        or to compute overall mean from all data and remove it from all trajectories (False)
    """    

    # --- 1. Standardize Input Data ---
    if isinstance(data, np.ndarray):
        data_list = [data]
        input_was_array = True
    elif isinstance(data, (list, tuple)):
        data_list = data
        input_was_array = False
    else:
        raise ValueError("Data should be a list of arrays or a single array.")

    k = len(data_list)
    d = data_list[0].shape[1]

    if mean is None:
        mean = compute_mean(data_list, elementwise=elementwise)

    # --- 3. Standardize the Mean Shape ---
    if isinstance(mean, (list, tuple)):
        assert len(mean) == k, "Length of mean list must match number of trajectories."
        mean_list = [np.reshape(m, (1, d)) for m in mean]
        
    elif isinstance(mean, np.ndarray) and mean.ndim == 3:
        assert mean.shape[0] == k, "First dimension of 3D mean array must match number of trajectories."
        mean_list = [np.reshape(mean[i], (1, d)) for i in range(k)]
        
    elif isinstance(mean, np.ndarray) and mean.shape in [(k, d), (d, k)] and k > 1:
        if mean.shape == (k, d):
            mean_list = [np.reshape(mean[i], (1, d)) for i in range(k)]
        else:
            mean_list = [np.reshape(mean[:, i], (1, d)) for i in range(k)]
            
    else:
        global_mean = np.reshape(mean, (1, d))
        mean_list = [global_mean] * k

    # --- 4. Remove Mean ---
    centered_data = []
    for i, traj in enumerate(data_list):
        # Broadcasting automatically handles (m, d) - (1, d)
        centered_data.append(traj - mean_list[i])

    # --- 5. Return in original format ---
    if input_was_array:
        return centered_data[0]
    
    if return_mean:
        return centered_data, mean_list[0] if not elementwise else mean_list
    return centered_data



def compute_cov_from_batches(data,
                          lag=None,
                          batch_size=1000,
                          compute_inst=True,
                          compute_lagged=True,
                          full_data_C0=False,
                          symmetrize=False,
                          remove_mean=False,
                          mean=None,
                          print_info=False):
    
    # TODO : symmetrize 

    '''
    Parameters:
    -----------
    data: list of arrays (m_i, d) or array (m, d)
        trajectory data to compute covariance from

    lag: int or None
        lag time for lagged covariance. If None, only instantaneous covariance on full data set is computed.
    
    batch_size: int
        size of batches to process data in. 
    
    compute_inst: bool
        whether to compute instantaneous covariance matrix (C0) or not. If False, only lagged covariance (Ct) is computed.

    compute_lagged: bool
        whether to compute lagged covariance matrix (Ct) or not. If False, only instantaneous covariance (C0) is computed.

    full_data_C0: bool
        whether to compute instantaneous covariance on full data set (True) or until -lag.
        If True, lagged covariance is still computed in batches if compute_lagged is True.

    mean: array (d,) or None
        mean to subtract from data before covariance computation. If None, mean is computed from data.
    
    remove_mean: bool
        whether to remove mean from data before covariance computation. If True, mean is computed from data if not provided as an argument, and subtracted from data.
    '''

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = [data]
    elif isinstance(data, list) or isinstance(data, tuple) or (isinstance(data, np.ndarray) and data.ndim == 3):
        for traj in data:
            if not isinstance(traj, np.ndarray):
                raise ValueError('All trajectories should be numpy arrays.')
            
    else:
        raise ValueError('Data should be a numpy array or a list of numpy arrays.')
    
    if lag is None:
        compute_inst = True
        full_data_C0 = True
        compute_lagged = False
    
    elif not isinstance(lag, int) or lag < 0:
        raise ValueError('Lag should be a non-negative integer.')

    assert compute_inst or compute_lagged, 'At least one of compute_inst and compute_lagged should be True.'


    # initialize
    feat_dim = data[0].shape[1]
    m = 0
    n_trajs = len(data)
    s = np.zeros(feat_dim)
    s_C0 = np.zeros(feat_dim)
    s_x_lagged = np.zeros(feat_dim)
    s_y_lagged = np.zeros(feat_dim)
    C0_batches = np.zeros((feat_dim, feat_dim))
    Ct_batches = np.zeros((feat_dim, feat_dim))

    for i, curr_traj in enumerate(data):

        curr_m = curr_traj.shape[0]
        m += curr_m
        if compute_lagged:
            if lag == 0:
                s_x_lagged += np.sum(curr_traj, axis=0)
                s_y_lagged += np.sum(curr_traj, axis=0)
            else:
                s_x_lagged += np.sum(curr_traj[:-lag], axis=0)
                s_y_lagged += np.sum(curr_traj[lag:], axis=0)
        prev_batch = None
        n_batches = int(np.ceil(curr_m / batch_size))

        print(f'processing trajectory {i+1}/{n_trajs} with {curr_m} frames') if print_info else None
        for i_batch, start in enumerate(range(0, curr_m, batch_size)):
            print(f'  processing batch {i_batch + 1} / {n_batches} starting at frame {start}') if print_info else None
            end = min(start + batch_size, curr_m)
            curr_batch = curr_traj[start:end]

            if (not full_data_C0) and (i_batch == n_batches - 1):
                X = curr_batch[:-lag]
            else:
                X = curr_batch

            if compute_inst:
                s_C0 += np.sum(X, axis=0)
            
            if compute_inst:
                C0_batches += np.dot(X.T, X)

            if compute_lagged:
                curr_batch_length = curr_batch.shape[0]
                if i_batch > 0 or batch_size >= curr_m:
                    if lag <= curr_batch_length:
                        x_i = prev_batch
                        y_i = np.vstack((prev_batch[lag:], curr_batch[:lag]))
                        Ct_batches += np.dot(x_i.T, y_i)

                        if symmetrize:
                            Ct_batches += np.dot(y_i.T, x_i)
                        
                        if i_batch == n_batches - 1:
                            x_i = curr_batch[:-lag]
                            y_i = curr_batch[lag:]
                            Ct_batches += np.dot(x_i.T, y_i)

                            if symmetrize:
                                Ct_batches += np.dot(y_i.T, x_i)
                            # s_y += np.sum(curr_batch[lag:], axis=0)
                    else:
                        # must be last batch -> take for Y everything until the end + what is necessary from prev batch
                        print('lag larger than current batch size, special handling')
                        print('lag', lag, 'prev batch shape', prev_batch.shape, 'curr batch shape', curr_batch.shape)
                        Y = np.vstack((prev_batch[lag:], curr_batch))
                        index_from_previous = Y.shape[0]
                        print('Y shape', Y.shape, 'index from previous', index_from_previous)
                        Ct_batches += np.dot(prev_batch[:index_from_previous].T, Y)

                prev_batch = curr_batch
    
            s += np.sum(curr_batch, axis=0)
    
    if mean is None:
        mean = s / m
        
    if full_data_C0:
        denom_x = m
    else:
        denom_x = m - n_trajs * lag

    if compute_lagged:
        denom_y = m - n_trajs * lag
    
    if compute_inst:
        mean_C0 = s_C0 / denom_x
        C0_batches = C0_batches / denom_x

        if remove_mean:
            C0_batches = (
                C0_batches
                - np.outer(mean, mean_C0)
                - np.outer(mean_C0, mean)
                + np.outer(mean, mean)
            )

    if compute_lagged:
        mean_x_lagged = s_x_lagged / denom_y
        mean_y_lagged = s_y_lagged / denom_y
        if symmetrize:
            Ct_batches = Ct_batches / (2 * denom_y)
        else:
            Ct_batches = Ct_batches / denom_y


        if remove_mean:
            Ct_batches = (
                Ct_batches
                - np.outer(mean, mean_y_lagged)
                - np.outer(mean_x_lagged, mean)
                + np.outer(mean, mean)
            )

    if compute_inst and not compute_lagged:
        return C0_batches, mean
    
    elif compute_lagged and not compute_inst:
        return Ct_batches, mean
    
    else:
        return C0_batches, Ct_batches, mean



def compute_covariances(X, Y=None, which='both', full_data_C0=False , symmetrize=False):
    '''
    Parameters:
    -----------
    X: array (m, d)
    Y: array (m, d) - should be the same length as X, but shifted by lag (i.e. Y[i] corresponds to X[i+lag])
    symmetrize: bool, whether to symmetrize covariance matrices (C0 and Ct) by averaging with their transposes.
    '''

    if which not in ['inst', 'lagged', 'both']:
        raise ValueError("which should be either 'inst' or 'lagged'.")

    if which in ['lagged', 'both'] and Y is None:
        raise ValueError("Y should be provided if which is 'lagged' or 'both'.")
    
    if symmetrize and Y is None:
        raise ValueError("Y should be provided if symmetrize=True.")

    if symmetrize:
        if which in ['inst', 'both']:
            C0 = (np.dot(X.T, X) + np.dot(Y.T, Y)) / (2 * (X.shape[0] - 1))
        
        if which in ['lagged', 'both']:
            Ct = (np.dot(X.T, Y) + np.dot(Y.T, X)) / (2 * (X.shape[0] - 1))
    
    else:
        if which in ['inst', 'both']:
            C0 = np.dot(X.T, X) / (X.shape[0] - 1)
        
        if which in ['lagged', 'both']:
            Ct = np.dot(X.T, Y) / (X.shape[0] - 1)

    if which == 'inst':
        return C0

    elif which == 'lagged':
        return Ct

    else:
        return C0, Ct

def load_covariance_from_file(dir_base, dir_base_lag=None, which='inst', load_mean=False, lag=None, symmetrize=False): #TODO add option CO_full_data for C0.
    """
    Load previously-saved covariance matrices/mean from disk, if present.

    Parameters:
        dir_base: directory holding the lag-independent C0.npy/mean.npy.
        dir_base_lag: directory holding the lag-dependent Ct_lag=....npy;
            defaults to dir_base if not given.
        which: 'inst' (C0 only), 'lagged' (Ct only), or 'both'.
        load_mean: also try to load mean.npy.
        lag, symmetrize: used to build the Ct filename (must match how it was saved).

    Returns:
        Whatever `which` selects (C0, Ct, or (C0, Ct)), each None if not found
        on disk; additionally paired with mean if load_mean is True. Never
        computes anything -- pure disk lookup.
    """
    if which not in ['inst', 'lagged', 'both']:
        raise ValueError("which should be either 'inst' or 'lagged'.")

    if dir_base_lag is None:
        dir_base_lag = dir_base

    sym_file_extension = '_sym' if symmetrize else ''
    filepath_C0 = f'{dir_base}/C0.npy'
    filepath_mean = f'{dir_base}/mean.npy'
    filepath_Ct = f'{dir_base_lag}/Ct_lag={lag}{sym_file_extension}.npy'

    C0 = None
    Ct = None
    mean = None

    if which in ['inst', 'both']:
        if os.path.exists(filepath_C0):
            print('loading C0 from file', end='')
            C0 = load_from_file(filepath_C0)
            print('... done')
    
    if which in ['lagged', 'both']:
        if os.path.exists(filepath_Ct):
            print('loading Ct from file', end='')
            Ct = load_from_file(filepath_Ct)
            print('... done')

    if load_mean:
        if os.path.exists(filepath_mean):
            print('loading mean from file', end='')
            mean = load_from_file(filepath_mean)
            print('... done')

    if which == 'inst':
        return (C0, mean) if load_mean else C0

    elif which == 'lagged':
        return (Ct, mean) if load_mean else Ct

    elif which == 'both':
        return (C0, Ct, mean) if load_mean else (C0, Ct)
    


def load_or_compute_covariance_in_memory( #TODO add option CO_full_data for C0.
    *,
    dir_base=None,
    dir_base_lag=None,
    X=None,
    Y=None,
    lag=None,
    remove_mean=False,
    return_mean=False,
    mean=None,
    symmetrize=False,
    save_results=False,
    which='both',
):
    """
    Load C0/Ct/mean from disk if available (see load_covariance_from_file),
    otherwise compute whichever of them are missing from X, Y (holding all
    trajectory data in memory at once), then optionally save the results.

    Parameters:
        dir_base / dir_base_lag: see load_covariance_from_file.
        X, Y: array (m, d) or list of arrays (m_i, d); Y defaults to X shifted
            by lag if not given (see prepare_data).
        lag: required if which is 'lagged' or 'both'.
        remove_mean: subtract mean from X, Y before computing covariances.
        return_mean: also return the mean (computed if needed and not None).
        which: 'inst' (C0), 'lagged' (Ct), or 'both'.
        save_results: persist any newly-computed matrices to dir_base/dir_base_lag.

    Returns:
        C0 / Ct / (C0, Ct), each optionally paired with mean if return_mean.
    """
    if which not in ['inst', 'lagged', 'both']:
        raise ValueError("which should be 'inst', 'lagged' or 'both'.")

    if dir_base_lag is None:
        dir_base_lag = dir_base

    compute_inst = which in ['inst', 'both']
    compute_lagged = which in ['lagged', 'both']

    C0 = None
    Ct = None
    need_mean = (remove_mean or return_mean) and mean is None

    if dir_base is not None:
        loaded = load_covariance_from_file(
            dir_base=dir_base,
            dir_base_lag=dir_base_lag,
            which=which,
            load_mean=need_mean,
            lag=lag,
            symmetrize=symmetrize,
        )

        if which == 'inst':
            C0, mean = loaded if need_mean else (loaded, None)
        elif which == 'lagged':
            Ct, mean = loaded if need_mean else (loaded, None)
        else:
            C0, Ct, mean = loaded if need_mean else loaded[:2] + (None,)

    needs_C0 = compute_inst and C0 is None
    needs_Ct = compute_lagged and Ct is None
    needs_mean = need_mean and mean is None

    if needs_C0 or needs_Ct:
        res_prepre_data = prepare_data(
            X,
            Y,
            return_mean=needs_mean,
            remove_mean=remove_mean,
            mean=mean,
            lag=lag,
        )
        if needs_mean:
            Xp, Yp, mean = res_prepre_data
        else:
            Xp, Yp = res_prepre_data 

        if needs_C0 and needs_Ct:
            C0, Ct = compute_covariances(
                Xp,
                Yp,
                which='both',
                symmetrize=symmetrize,
            )
        elif needs_C0:
            C0 = compute_covariances(
                Xp,
                Yp,
                which='inst',
                symmetrize=symmetrize,
            )
        elif needs_Ct:
            Ct = compute_covariances(
                Xp,
                Yp,
                which='lagged',
                symmetrize=symmetrize,
            )

        if save_results and dir_base is not None:
            dir_base = Path(dir_base)
            dir_base.mkdir(parents=True, exist_ok=True)

            sym_file_extension = '_sym' if symmetrize else ''

            if needs_C0:
                save_to_file(obj=C0, filepath=dir_base / 'C0.npy')

            if needs_Ct:
                dir_base_lag = Path(dir_base_lag)
                dir_base_lag.mkdir(parents=True, exist_ok=True)
                save_to_file(
                    obj=Ct,
                    filepath=dir_base_lag / f'Ct_lag={lag}{sym_file_extension}.npy',
                )

            if needs_mean:
                save_to_file(obj=mean, filepath=dir_base / 'mean.npy')

    if which == 'inst':
        return C0 if not return_mean else (C0, mean)

    if which == 'lagged':
        return Ct if not return_mean else (Ct, mean)

    return (C0, Ct) if not return_mean else (C0, Ct, mean)





    


def load_or_compute_covariance_from_batches(dir_base=None,
                                dir_base_lag=None,
                                mmap_trajs=None,
                                lag=None,
                                full_data_C0=True,
                                remove_mean=False,
                                return_mean=False,
                                mean=None,
                                symmetrize=False,
                                save_results=True,
                                batch_size=1000,
                                which='both'):
    """
    Helper function to load or compute mean and covariance matrices C0 and/or Ct from trajectory data in batches.

    Parameters:
    -----------
    dir_base / dir_base_lag: see load_covariance_from_file; if dir_base is None,
        computes from scratch without trying to load/save anything.
    mmap_trajs: array or list of arrays, trajectory data (may be memory-mapped)
        to compute from if not found on disk.
    full_data_C0: bool, whether C0 is accumulated over all frames (True) or with
        the last batch trimmed to match the lagged-covariance frame count (False).
    remove_mean: bool, whether to remove mean from data before covariance computation.
                  If True, mean is computed from data if not provided as an argument, and subtracted from data before covariance computation.
    mean: array (d,) or None, mean to subtract from data before covariance computation. If None and remove_mean is True, mean is computed from data.
    return_mean: bool, whether to return mean along with covariance matrices. Also returned if None!
    save_results: bool, whether to persist any newly-computed matrices to disk.
    batch_size: int, number of frames processed per batch.
    which: {'inst', 'lagged', 'both'}
        Which covariance matrix/matrices to load or compute.
        'inst' returns (C0, mean), 'lagged' returns (Ct, mean), and
        'both' returns (C0, Ct, mean).
    """

    if which not in ['inst', 'lagged', 'both']:
        raise ValueError("which should be either 'inst', 'lagged' or 'both'.")

    compute_inst = which in ['inst', 'both']
    compute_lagged = which in ['lagged', 'both']

    if compute_lagged and lag is None:
        raise ValueError("lag should be provided if which is 'lagged' or 'both'.")

    if dir_base_lag is None:
        dir_base_lag = dir_base

    def _save_if_requested(obj, filepath):
        if obj is None:
            return
        if save_results:
            save_to_file(obj=obj, filepath=filepath)

    if dir_base is None:
        # if dir_base is not provided, compute from scratch without trying to load from file
        cov_label = {'inst': 'C0', 'lagged': 'Ct', 'both': 'C0 and Ct'}[which]
        print(f'dir_base is None, computing {cov_label} from batches without loading/saving to file')
        computed_results = compute_cov_from_batches(data=mmap_trajs,
                                            lag=lag,
                                            compute_inst=compute_inst,
                                            symmetrize=symmetrize,
                                            full_data_C0=full_data_C0,
                                            compute_lagged=compute_lagged,
                                            remove_mean=remove_mean,
                                            mean=mean,
                                            batch_size=batch_size,
                                            print_info=True)
        return computed_results if return_mean else computed_results[:-1] if which == 'both' else computed_results[0]

    # dir_base is not None, try loading:
    filepath_C0 = f'{dir_base}/C0.npy'
    filepath_mean = f'{dir_base}/mean.npy'
    sym_file_extension = '_sym' if symmetrize else ''
    filepath_Ct = f'{dir_base_lag}/Ct_lag={lag}{sym_file_extension}.npy'

    C0 = None
    Ct = None

    # bool whether mean needs to be loaded for downstram calculations
    wants_mean = remove_mean and (mean is None)

    # load covariance matrices and mean from file if they exist. This return None for matrices/mean that do not exist, which is handled in the following code.
    loaded = load_covariance_from_file(dir_base=dir_base, dir_base_lag=dir_base_lag, which=which, load_mean=wants_mean, lag=lag, symmetrize=symmetrize)

    if which == 'inst':
        C0, loaded_mean = loaded if wants_mean else (loaded, None)
    elif which == 'lagged':
        Ct, loaded_mean = loaded if wants_mean else (loaded, None)
    else:
        C0, Ct, loaded_mean = loaded if wants_mean else (*loaded, None)


    if wants_mean:
        mean = loaded_mean

    # at this point mean was either provided and is not None, or it was loaded from file if it existed, or it is still None.
    # If it needs to be computed depends on wants_mean, e.g.
    # mean = None, remove_mean=False -> mean does not need to be computed
    # mean = None, remove_mean=True -> mean needs to be computed, only in this case overwrite mean.
    # mean != None, remove_mean=False -> mean does not need to be computed, it is provided as an argument
    # mean != None, remove_mean=True -> mean does not need to be computed, it is provided as an argument

    needs_C0 = compute_inst and C0 is None
    needs_Ct = compute_lagged and Ct is None
    needs_mean = wants_mean and mean is None

    # either mean is provided as an argument or it is loaded from file, if it exists.
    # If mean is still None, it will be computed later if necessary (i.e. if C0 or Ct do not exist and need to be computed from batches)

    if needs_C0 or needs_Ct:
        computed_covariances = compute_cov_from_batches(data=mmap_trajs,
                                                        lag=lag,
                                                        compute_inst=needs_C0,
                                                        full_data_C0=full_data_C0,
                                                        compute_lagged=needs_Ct,
                                                        symmetrize=symmetrize,
                                                        remove_mean=remove_mean,
                                                        mean=mean,
                                                        batch_size=batch_size,
                                                        print_info=True)
    
        if needs_C0 and needs_Ct:
            C0, Ct, mean = computed_covariances
        elif needs_C0:
            C0, mean = computed_covariances
        elif needs_Ct:
            Ct, mean = computed_covariances

    _save_if_requested(obj=C0, filepath=filepath_C0)
    _save_if_requested(obj=Ct, filepath=filepath_Ct)
    _save_if_requested(obj=mean, filepath=filepath_mean)

    if which == 'inst':
        return (C0, mean) if return_mean else C0

    elif which == 'lagged':
        return (Ct, mean) if return_mean else Ct

    else:
        return (C0, Ct, mean) if return_mean else (C0, Ct)
    

def compute_whitening_transform(U, s, rank=None):
    # helper function to compute whitening matrix W from eigendecomposition of C0 (U, s) and rank for truncation.
    U = U[:, :rank]
    s = s[:rank]
    W = U * (s ** (-0.5))[None, :]
    return W

def compute_Ct_whitened(W, Ct):
    """Whiten the lagged covariance matrix: Ct_whitened = W^T @ Ct @ W."""
    return np.dot(np.dot(W.T, Ct), W)

def load_or_compute_Ct_whitened(W, Ct, lag, dir_base=None, symmetrize=False, save_results=True, verbose=False):
    """
    Load Ct_whitened from disk if a matching (or higher-rank) file exists,
    otherwise compute it from W and Ct and optionally save it.

    Since rank == W.shape[1] only truncates the whitened matrix, a
    previously-saved Ct_whitened computed at a larger rank can be reused by
    truncating it to the requested rank -- this function searches dir_base
    for the smallest file with rank >= the requested rank (or a 'full'-rank
    file) before falling back to recomputing from W and Ct.

    Parameters:
        W: whitening transform, shape (d, rank).
        Ct: lagged covariance matrix, or None (only valid if a cached file exists).
        lag: used to build the cache filename.
        dir_base: directory to search/save in; if None, always recomputes
            without touching disk.
        symmetrize: used to build the cache filename (must match how it was saved).
        save_results: persist a newly-computed Ct_whitened to disk.
        verbose: print cache hits/misses.

    Returns:
        Ct_whitened, shape (rank, rank), or None if neither a cache file nor
        Ct is available.
    """
    rank = W.shape[1]

    if dir_base is None:
        print('dir_base is None, computing whitening and Ct_whitened without loading/saving to file') if verbose else None
        if Ct is None:
            return None
        return compute_Ct_whitened(W, Ct)
    
    dir_base = Path(dir_base)
    rank_extension = f'_rank={rank}' if rank is not None else 'full_'
    sym_file_extension = '_sym' if symmetrize else ''
    exact_path = dir_base / f'Ct_whitened_lag={lag}{rank_extension}{sym_file_extension}.npy'
    
    Ct_whitened = None
    print(f'Checking for existing Ct_whitened files in {dir_base}...') if verbose else None
    # 1. Check for exact match
    if exact_path.exists():
        print(f'    loading Ct_whitened from {exact_path}') if verbose else None
        Ct_whitened = load_from_file(exact_path)

    # 2. If rank is specified, look for ANY larger rank (or 'full')
    elif rank is not None:
        if verbose:
            print(f'    no exact match for Ct_whitened with rank={rank} found. Looking for larger rank files to load and truncate...')
        best_larger_rank = float('inf')
        best_larger_file = None
        
        # Regex to match filenames and extract the rank (if it exists)
        # pattern = re.compile(rf'^Ct_whitened_lag={lag}(?:rank=(\d+)_|full_){sym_file_extension}\.npy$')
        pattern = re.compile(
            rf'^Ct_whitened_lag={lag}_(?:rank=(\d+)|full){re.escape(sym_file_extension)}\.npy$'
)

        if dir_base.exists():
            for file_path in dir_base.iterdir():
                match = pattern.match(file_path.name)
                if match:
                    # If it's 'full', group(1) is None. If 'rank=N', group(1) is 'N'
                    file_rank_str = match.group(1)
                    file_rank = int(file_rank_str) if file_rank_str is not None else float('inf')
                    
                    # Find the smallest rank that is still larger than the requested rank
                    if file_rank > rank and file_rank < best_larger_rank:
                        print(f'    found candidate file {file_path.name} for loading and truncation.') if verbose else None
                        best_larger_rank = file_rank
                        best_larger_file = file_path
        
        # If we found a suitable larger file, load and truncate it
        if best_larger_file is not None:
            source_rank_str = "full" if best_larger_rank == float('inf') else str(best_larger_rank)
            if verbose:
                print(f'    loading Ct_whitened (rank {source_rank_str}) from {best_larger_file.name} and truncating to {rank}...', end=' ')
            R_larger = load_from_file(best_larger_file)
            Ct_whitened = R_larger[:rank, :rank]
            print('Done.') if verbose else None

            
    if Ct_whitened is None:
        if Ct is None:
            print('Ct is None, cannot compute Ct_whitened. Returning W and None for Ct_whitened.') if verbose else None
        else:
            print('Didnt find a larger rank, computing Ct_whitened from whitening and Ct...', end=' ') if verbose else None
            Ct_whitened = compute_Ct_whitened(W, Ct)
            if save_results:
                print('Saving Ct_whitened to file.', end=' ')
                save_to_file(obj=Ct_whitened, filepath=exact_path)
            print('Done.') if verbose else None

    return Ct_whitened


def load_or_compute_eigendecomposition(M,
                                       dir_base=None,
                                       filename_without_extension_string=None,
                                       mode='eig',
                                       n_ev=None,
                                       save=True,
                                       verbose=False
                                       ):
    
    """
    Load the eigendecomposition of M from disk if a matching (or larger n_ev)
    file exists, otherwise compute and optionally cache it.

    Since n_ev only truncates to the leading eigenpairs, a previously-saved
    decomposition computed with a larger (or unrestricted) n_ev can be reused
    by truncating it -- this function searches dir_base for the smallest
    n_ev-tagged file that covers the request (or a file with no n_ev tag,
    i.e. computed with n_ev=None) before falling back to recomputing from M.

    Parameters:
        M: matrix to decompose, or None (only valid if a cache file exists).
        dir_base, filename_without_extension_string: cache location; if either
            is None, always recomputes without touching disk.
        mode: passed to eigendecomposition() ('eig' or 'eigh').
        n_ev: number of leading eigenpairs to keep (None = all).
        save: persist a newly-computed decomposition to disk.
        verbose: print cache hits/misses.

    Returns:
        (s, U): eigenvalues and eigenvectors (see eigendecomposition()), or
        (None, None) if M is None and no cache file was found.
    """
    loaded = False
    save = save and (dir_base is not None) and (filename_without_extension_string is not None)
    n_ev_extension = f'n_ev={n_ev}_' if n_ev is not None else ''

    print(f'Checking if eigendecomposition can be loaded from file for {filename_without_extension_string} ...') if verbose else None

    if dir_base is not None and filename_without_extension_string is not None:

        filepath_evecs = Path(dir_base) / Path(f'{filename_without_extension_string}_{n_ev_extension}evecs.npy')
        filepath_evals = Path(dir_base) / Path(f'{filename_without_extension_string}_{n_ev_extension}evals.npy')

        filepath_evecs_to_load = filepath_evecs
        filepath_evals_to_load = filepath_evals

        if not (os.path.exists(filepath_evecs_to_load) and os.path.exists(filepath_evals_to_load)) and n_ev is not None:
            print('    no exact match for eigendecomposition with n_ev={n_ev} found. Looking for larger n_ev files to load and truncate...') if verbose else None

            pattern = re.compile(
                rf'^{re.escape(filename_without_extension_string)}_n_ev=(\d+)_evecs\.npy$'
            )

            larger_decompositions = []

            if Path(dir_base).exists():
                for file_path in Path(dir_base).iterdir():
                    match = pattern.match(file_path.name)

                    if match:
                        available_n_ev = int(match.group(1))

                        corresponding_evals = (
                            Path(dir_base)
                            / f'{filename_without_extension_string}_n_ev={available_n_ev}_evals.npy'
                        )

                        if available_n_ev >= n_ev and corresponding_evals.exists():
                            larger_decompositions.append(
                                (available_n_ev, file_path, corresponding_evals)
                            )

            if larger_decompositions:
                _, filepath_evecs_to_load, filepath_evals_to_load = min(
                    larger_decompositions,
                    key=lambda item: item[0]
                )
            else:
                filepath_evecs_without_n_ev = (
                    Path(dir_base)
                    / f'{filename_without_extension_string}_evecs.npy'
                )
                filepath_evals_without_n_ev = (
                    Path(dir_base)
                    / f'{filename_without_extension_string}_evals.npy'
                )

                if (
                    filepath_evecs_without_n_ev.exists()
                    and filepath_evals_without_n_ev.exists()
                ):
                    filepath_evecs_to_load = filepath_evecs_without_n_ev
                    filepath_evals_to_load = filepath_evals_without_n_ev

        if os.path.exists(filepath_evecs_to_load) and os.path.exists(filepath_evals_to_load):
            print(f'    loading eigendecomposition from {filepath_evecs_to_load}...', end=' ') if verbose else None
            U = load_from_file(filepath_evecs_to_load)
            s = load_from_file(filepath_evals_to_load)

            if n_ev is not None:
                U = U[:, :n_ev]
                s = s[:n_ev]

            loaded = True
            if verbose:
                print('Done.')

    if not loaded:
        if M is None:
            if verbose:
                print('Matrix is None, cannot compute eigendecomposition. Returning None for eigenvalues and eigenvectors.')
            return None, None
        if verbose:
            print(f'computing eigendecomp for {filename_without_extension_string}...', end=' ')
        s, U = eigendecomposition(M, mode=mode, n_ev=n_ev)
        s, U = filter_eigenvalues_and_vectors(s, U)
        if verbose:
            print('Done.')

        if save:
            if verbose:
                print(f'Saving under {dir_base}/{filename_without_extension_string}...')
            save_to_file(obj=s, filepath=filepath_evals) 
            save_to_file(obj=U, filepath=filepath_evecs)
            if verbose:
                print('Done.')
    
    return s, U



def eigendecomposition(M, mode=None, n_ev=None, tol=0.0):
    """
    Returns s, U where M @ U = U @ diag(s), sorted by descending absolute
    eigenvalue.

    Parameters:
        mode: 'eigh' for a symmetric/Hermitian matrix (real eigenvalues,
            faster); anything else uses the general (possibly complex)
            eigendecomposition.
        n_ev: if given, only compute the n_ev eigenpairs with largest
            magnitude (via a sparse/iterative solver) instead of the full
            spectrum.
        tol: convergence tolerance for the sparse solver, only used when
            mode='eigh' and n_ev is given (eigsh's `tol`, 0 = machine precision).
    """
    if mode == 'eigh' and n_ev is not None:
        s, U = eigsh(M, k=n_ev, which='LM', tol=tol)

    elif n_ev is not None:
        from scipy.sparse.linalg import eigs
        s, U = eigs(M, k=n_ev, which="LM")

    elif mode == 'eigh':
        s, U = eigh(M)

    else:
        s, U = eig(M)
        
    idx = np.argsort(np.abs(s))[::-1]
    return s[idx], U[:, idx]


def filter_eigenvalues_and_vectors(d, V, tol=1e-10, print_info=False):
    """
    Sort eigenpairs by descending magnitude, clean up numerical noise
    (real-if-close), and drop eigenvalues below a relative cutoff.

    Parameters:
        d, V: eigenvalues and eigenvectors (columns of V), as returned by
            eigendecomposition().
        tol: relative-eigenvalue cutoff -- eigenpairs with
            eigenvalue / largest_eigenvalue < tol are dropped. This is what
            determines the whitening rank when used on C0's eigendecomposition.

    Returns:
        (d_filtered, V_filtered), sorted descending and truncated to the
        eigenpairs that passed the cutoff.
    """
    d = np.asarray(d)
    V = np.asarray(V)

    # Convert legacy object arrays without forcing real data to complex.
    if d.dtype == object:
        d = np.asarray(d.tolist())

    if V.dtype == object:
        V = np.asarray(V.tolist())

    # Convert complex arrays back to real when imaginary parts are numerical noise.
    d = np.real_if_close(d, tol=1000)
    V = np.real_if_close(V, tol=1000)

    idx = np.argsort(-np.abs(d))
    d_sorted = d[idx]
    V_sorted = V[:, idx]

    largest = np.real(d_sorted[0])
    keep = np.real(d_sorted) / largest >= tol

    return d_sorted[keep], V_sorted[:, keep]


def prepare_data(X, Y=None, remove_mean=False, return_mean=False, mean=None, lag=None):
    """
    Optionally remove the mean from X (and Y), and derive Y from X via
    split_by_lag if Y is not already given.

    Parameters:
        X: array (m, d) or list of arrays (m_i, d).
        Y: instantaneous/time-lagged counterpart of X; if None, both X and Y
            are derived from splitting X at `lag`.
        remove_mean: subtract mean before splitting/pairing.
        mean: mean to subtract; if None and remove_mean is True, computed from
            the data (jointly from X if Y is None, separately from X and Y
            otherwise -- see the printed warning in that branch).
        lag: required if Y is None.

    Returns:
        (X, Y) or (X, Y, mean) if return_mean is True. When Y was None and a
        mean was computed, mean is a single array; when Y was given and a
        mean was computed for each of X and Y separately, mean is the tuple
        (mean_X, mean_Y).
    """
    assert Y is not None or lag is not None, 'Either Y or lag should be provided.'

    if Y is None:
        if remove_mean:
            res = remove_mean_fun(X, mean=mean, return_mean=return_mean)
            if return_mean:
                X, mean = res
            else:
                X = res
        X, Y = split_by_lag(X, lag)

    else:
        if remove_mean:
            if mean is None:
                print('Warning: mean is None, but remove_mean is True. Taking setwise means of X and Y.')
            res_X = remove_mean_fun(X, mean=mean, return_mean=return_mean)
            res_Y = remove_mean_fun(Y, mean=mean, return_mean=return_mean)
            if return_mean:
                X, mean_X = res_X
                Y, mean_Y = res_Y
                mean = (mean_X, mean_Y)
            else:
                X = res_X
                Y = res_Y

    if return_mean:
        return X, Y, mean
    return X, Y







            # elif mode == 'svd':
            #     raise NotImplementedError('Batchwise SVD whitening not implemented yet.')
            #     # whitening step based on SVD of data

            #     U, s, Vh = svd(X * (m-1)**-0.5, full_matrices=False)

            #     ind = np.where(s / s[0] >= tol**0.5)[0]
            #     r = ind.shape[0]

            #     s = s[:r]
            #     U = U[:, :r]
            #     Vh = Vh[:r, :]

            #     if self.cplx:
            #         L = U.conj() * (s ** (-1))[None, :]
            #         R = Vh @ (Y.conj().T * (m-1)**-0.5) @ L

            #     else:
            #         L = U * (s ** (-1))[None, :]
            #         R = Vh @ (Y.T * (m-1)**-0.5) @ L
            # else: