# ---- helper functions for general use

from pathlib import Path
import glob
import os
import json
import pickle
import numpy as np
import mdtraj as md     # used in the Protein class to load the topology file
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List, Union, Optional, Any

def set_attrs_from_dict(target, attributes, post_load=True, *init_args, **init_kwargs):
    """
    Set attributes of an object from a dict.
    
    If `target` is a class, a new instance is created:
        obj = target(*init_args, **init_kwargs)
    and returned.

    If `target` is an object instance, its attributes are modified in-place.
    """

    # Determine instance or create one
    if isinstance(target, type):  
        # target is a class → instantiate
        obj = target(*init_args, **init_kwargs)
        return_new_instance = True
    else:
        # target is an instance → modify in-place
        obj = target
        return_new_instance = False


    # Assign attributes
    for key, value in attributes.items():
        setattr(obj, key, value)


    # Optional post_load
    if post_load and hasattr(obj, "post_load"):
        method = getattr(obj, "post_load")
        if callable(method):
            method()

    # Return new object only if we created one
    return obj if return_new_instance else None

def save_to_file(obj, filepath=None, attrs_to_save=None, overwrite=False):
    """
    Save data of an object to a file, creating directories as needed.

    - attrs_to_save: list of attribute names to save (if None, save all attributes).
    - If obj is a dict or list, it is saved directly.
    - If filepath ends with .npy → use np.save
    """

    if filepath is None:
        if hasattr(obj, 'final_path'):
            filepath = getattr(obj, 'final_path')
        else:
            raise ValueError("Provide filepath or use an object with attribute 'final_path'.")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.is_file() and not overwrite:
        return

    # Handle .npy separately
    if filepath.suffix == '.npy':
        arr = np.asarray(obj)

        if arr.dtype == object:
            raise TypeError(
                f"Refusing to save object array to {filepath}. "
                "For numeric data, convert to float64/complex128 before saving."
            )

        np.save(filepath, arr, allow_pickle=False)
        return

    to_save = None
    # Build data to save

    if isinstance(obj, (dict, list)):
        to_save = obj
    
    # if a list/tuple/set of attribute names is given, take only those
    elif isinstance(attrs_to_save, (list, tuple, set)):
        pass

    # if 'all': save all attributes
    elif attrs_to_save == 'all':
        to_save = obj.__dict__

    else:
        # now attrs_to_save is None or anything stupid.
        # try to take list from object itself, or set too all
        if hasattr(obj, 'attrs_to_save'):
            print('saving attrs_to_save from obj')
            attrs_to_save = obj.attrs_to_save
        elif attrs_to_save is None:
            to_save = obj.__dict__
        else:
            raise ValueError('attrs_to_save is not usable.')
            
    if to_save is None:
        # Only pick specific attributes
        try:
            to_save = {attr: getattr(obj, attr) for attr in attrs_to_save}
        except AttributeError as e:
            print(f"Error: One of the attributes {attrs_to_save} does not exist.")
            raise e

    with open(filepath, 'wb') as f:
        pickle.dump(to_save, f)

def load_from_file(filepath):
    filepath = Path(filepath)
    suffix = filepath.suffix

    try:
        if suffix == '.npy':
            return np.load(filepath, allow_pickle=False)

        if suffix == '.json':
            with open(filepath, 'r') as f:
                return json.load(f)

        if suffix == '.pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)

        raise ValueError(f"Unsupported file type: {suffix}")

    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        raise

# Helper function turning parameters into dictionary keys.
# It is used to check if results are already loaded.
def freeze(obj):
    if isinstance(obj, dict):
        return frozenset((k, freeze(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return tuple(freeze(i) for i in obj)
    return obj

# helper class to organize parameters loaded from a .json file
class Parameter():
    """ Generate an instance of a parameter class that turns a .json file into "usable" form."""
    def __init__(self, paramsFile='params.json', **kwargs):
        if kwargs:
            self.paramsDict = kwargs
        else:
            self.paramsFile = paramsFile
            self.paramsDict = load_from_file(filepath=paramsFile)
        set_attrs_from_dict(self, self.paramsDict)

    @classmethod
    def transformParam(cls, x, dataType):
        return dataType(x)

    @classmethod
    def toKey(cls, x):
        return cls.transformParam(x, str)


def project_data_onto_vectors(data, V, dim=None, mean=None, batch_size=None):
    ''' Project data onto top dim components of V. If batch_size is provided, 
        data is projected in batches to save memory.
        
    Input:
        data: array or list of arrays containing trajectories to project (n_frames, n_features)
        V: array of shape (n_features, n_components) containing vectors to project onto
        dim: number of components to project onto (if None, project onto all)
        mean: array of shape (n_features,) or list of such arrays containing mean 
              to subtract from data
        batch_size: int, size of batches for projection (if None, project all at once)
    '''
    if V.shape[1] < dim:
        raise ValueError(f"Requested projection dimension {dim} exceeds available components {V.shape[1]} in V.")

    # --- 1. Standardize the Input Data ---
    d = V.shape[0]
    if isinstance(data, (list, tuple)):
        for traj in data:
            assert traj.shape[1] == d, "Data dimension does not match projection vectors."
        input_was_array = False
    elif isinstance(data, np.ndarray):
        assert data.shape[1] == d, "Data dimension does not match projection vectors."
        data = [data]
        input_was_array = True
    else:
        raise ValueError("Data should be a list of arrays or a single array.")

    k = len(data)

    # --- 2. Standardize the Mean ---
    # We want mean_list to be a list of exactly length 'k', where each element is an array of shape (1, d) or None.
    if mean is not None:    
        print('subtracting mean from data before projection')
        if isinstance(mean, (list, tuple)):
            # Case A: Element-wise list (directly from compute_mean)
            assert len(mean) == k, "Length of mean list must match number of trajectories."
            mean_list = [np.reshape(m, (1, d)) for m in mean]
            
        elif isinstance(mean, np.ndarray) and mean.ndim == 3:
            # Case B: Element-wise 3D array (loaded from np.load)
            assert mean.shape[0] == k, "First dimension of 3D mean array must match number of trajectories."
            mean_list = [np.reshape(mean[i], (1, d)) for i in range(k)]
            
        elif isinstance(mean, np.ndarray) and mean.shape in [(k, d), (d, k)] and k > 1:
            # Case C: Element-wise 2D array
            if mean.shape == (k, d):
                mean_list = [np.reshape(mean[i], (1, d)) for i in range(k)]
            else: # shape is (d, k)
                mean_list = [np.reshape(mean[:, i], (1, d)) for i in range(k)]

        else:
            # Case D: Global mean (1D array or 2D column vector)
            global_mean = np.reshape(mean, (1, d))
            mean_list = [global_mean] * k

    else:
        mean_list = [None] * k

    projected_data = []
    
    V_proj = V[:, :dim] 

    for i, curr_traj in enumerate(data):
        curr_mean = mean_list[i]
        
        if batch_size is None:

            if curr_mean is not None:
                curr_data_to_project = curr_traj - curr_mean
            else:
                curr_data_to_project = curr_traj
                
            curr_proj_data = np.dot(curr_data_to_project, V_proj)

        else:
            # Batch-wise processing
            curr_proj_data = []
            for start in range(0, curr_traj.shape[0], batch_size):
                end = min(start + batch_size, curr_traj.shape[0])
                batch = curr_traj[start:end]

                if curr_mean is not None:
                    curr_batch_to_project = batch - curr_mean 
                else:
                    curr_batch_to_project = batch

                curr_proj_batch = np.dot(curr_batch_to_project, V_proj)
                curr_proj_data.append(curr_proj_batch)
                
            curr_proj_data = np.vstack(curr_proj_data)
            
        projected_data.append(curr_proj_data)
    
    if input_was_array:
        return projected_data[0]
    
    return projected_data


# --- Protein Class ---
class Protein:
    '''
    Protein class to hold metadata about a protein and its trajectories.
    Responsible for loading trajectory data and related information based on a configuration file.
    Expects a .json` with information about name etc.
    '''

    def __init__(self, filepath, group_md_files_by_first=True):
        
        loaded_dict = load_from_file(filepath)
        set_attrs_from_dict(self, loaded_dict)

        # a list of necessary keys for initialization.
        keys_required = ['name', 'dir_MD', 'filename_MD']
        for key in keys_required:
            assert hasattr(self, key), f"JSON must contain '{key}' key for the protein."

        # determine where results should be stored (e.g., for features, TICA, RFF, etc.) based on the name and feature scheme

        self.topology = None
        if hasattr(self, 'path_to_topology'):
            try:
                self.topology = md.load_topology(self.path_to_topology)
            except Exception as e:
                print(f'Could not load topology file {self.path_to_topology} for {self.name}. Error: {e}')

        
        self.md_paths = get_filenames_matching_pattern(dir_base=self.dir_MD, pattern=self.filename_MD, group_by_first_var=group_md_files_by_first)
        self.n_trajs = len(self.md_paths)
        self.n_files_per_traj = [len(self.md_paths[i]) for i in range(self.n_trajs)]

        self._compute_residue_indices()

        print(f'Initialized Protein {self.name} with {self.n_trajs} trajectories, each having {self.n_files_per_traj} files.') 

    

    def __str__(self):
        return f'{self.name}'
    
    def _compute_residue_indices(self):
        ind_triu = np.triu_indices(self.n_residues, k=3)
        self.res_ids_to_idx = {tuple(p): i for i, p in enumerate(zip(ind_triu[0], ind_triu[1]))}

    def get_subdirectory(self, feat_scheme, random=None):
        # returns the subdirectory name for storing features based on the feature scheme (and optionally random projection parameters)
        if random is None:
            return Path(self.name) / Path(feat_scheme)
        else:
            return Path(f'{self.name}_r={random}') / Path(feat_scheme)
    

    def traj_file_name(self, feat_scheme, i_traj, suffix='npy', full_traj=True, i_chunk=None):
        if full_traj:
            return Path(f'{self.name}_{feat_scheme}_full_trajectory{i_traj}.{suffix}')
        else:
            assert i_chunk is not None, "i_chunk must be provided for chunked trajectories."
            return Path(f'{self.name}_{feat_scheme}_trajectory{i_traj}_{i_chunk:03d}.{suffix}')
    
    def featurize(self,
                  feat_scheme='closest-heavy',
                  stride=1,
                  batch_size=10000,
                  save_chunks=False,
                  save_full=True):
        
        print(f'Featurizing trajectories for {self.name} using scheme "{feat_scheme}" with stride {stride} and batch size {batch_size}...')

        md_trajs_paths = self.md_paths # list of lists of file paths for each trajectory
        assert isinstance(md_trajs_paths, list) and all(isinstance(t, list) for t in md_trajs_paths), "md_paths should be a list of lists (one per trajectory)."

        dir_final = Path(self.dir_feat) / self.get_subdirectory(feat_scheme)
        Path(dir_final).mkdir(exist_ok=True, parents=True)

        for i_traj, paths_for_traj_i in enumerate(md_trajs_paths):
            x = []
            if save_full:
                path_full_traj_i = dir_final / self.traj_file_name(feat_scheme=feat_scheme, full_traj=True, i_traj=i_traj)
                if path_full_traj_i.is_file():
                    print(f'Full trajectory for trajectory {i_traj} already exists at {path_full_traj_i}. Skipping featurization for this trajectory.')
                    continue

            print(f'Loading data for trajectory {i_traj + 1} / {self.n_trajs}...', end='')
            traj = md.load(paths_for_traj_i, top=self.topology, stride=stride)
            print('Done')
            L = len(traj)
            for i_chunk in range((L // batch_size) + 1):
                print(f'Featurize chunk {i_chunk + 1} / {(L // batch_size) + 1}')
                ftraj, _ = md.compute_contacts(traj[i_chunk * batch_size: (i_chunk+1) * batch_size], contacts='all', scheme=feat_scheme)
                x.append(ftraj)

                if save_chunks:
                    np.save(dir_final / self.traj_file_name(feat_scheme=feat_scheme, i_traj=i_traj, full_traj=False, i_chunk=i_chunk), ftraj)
            
            print(f'Finished featurizing trajectory {i_traj + 1} / {self.n_trajs}.', flush=True)

            if save_full:
                full_traj = np.concatenate(x, axis=0)
                np.save(path_full_traj_i, full_traj)


    def load_MD_data(self, stride=1):
        md_list = []
        for i, traj_paths in enumerate(self.md_paths):
            print(f'Loading trajectory {i}...', end=' ')
            traj_obj = md.load(traj_paths, top=self.topology, stride=stride)
            md_list.append(traj_obj)
            print('Done')
        # Join trajectories into single mdtraj object and align trajectory to first frame
        print('Joining and superposing trajectories...')
        md_traj = md.join(md_list)
        md_traj.superpose(md_traj[0])
        return md_traj
    


    def project_onto_random_matrix(self, feat_scheme, r, batch_size=None, dir_random=None, stride=1, save=False):
        ## TODO check filename and trajwise??
        # get mmap trajs
        mmap_trajs = self.load_feat_trajs(feat_scheme=feat_scheme, stride=stride, mmap_mode=True)
        d = np.sum([mmap_trajs[i].shape[0] for i in range(len(mmap_trajs))])  # total number of frames across all trajectories

        # check if random projection matrix already exists, if not generate it
        dir_random = Path(dir_random) if (dir_random is not None and os.path.exists(dir_random)) else self.dir_feat / self.get_subdirectory(feat_scheme)
        Omegas = [get_random_projection_matrix(d=traj.shape[1], r=r, dir_base=dir_random) for traj in mmap_trajs]
        
        Y = [project_data_onto_vectors(data=mmap_trajs[i], V=Omegas[i], batch_size=batch_size) for i in range(len(mmap_trajs))]

        from scipy.linalg import qr
        Q, R = qr(np.vstack(Y).T, mode='economic')
        print(Q.shape, R.shape)

        projected_data = project_data_onto_vectors(data=Y, V=Q, batch_size=batch_size)
        

        if save:
            dir_final = Path(self.dir_feat) / self.get_subdirectory(feat_scheme, random=r)
            for i_traj, projected_traj in projected_data:
                np.save(dir_final / self.traj_file_name(feat_scheme=feat_scheme, full_traj=True, i_traj=i_traj), projected_traj)

        return projected_data



    def load_feat_trajs(self,
                        feat_scheme,
                        stride=1,
                        mmap_mode=False,
                        remove_mean=False,
                        return_mean=False,
                        list_of_trajs=None,
                        range_of_residues=None):
                        
        # range_of_residues: tuple (start, end) to specify a range of residues to load
        
        dir_feat = Path(self.dir_feat) / self.get_subdirectory(feat_scheme)
        X = []
        res_pair_indices = None

        list_of_trajs = np.array(list_of_trajs) if list_of_trajs is not None else np.arange(self.n_trajs)

        if range_of_residues is not None:
            assert isinstance(range_of_residues, tuple) or isinstance(range_of_residues, list), "range_of_residues should be a tuple of (start, end)."
            assert len(range_of_residues) == 2, "range_of_residues should be a tuple of (start, end)."
            assert isinstance(range_of_residues[0], int) and isinstance(range_of_residues[1], int), "range_of_residues values should be integers."
            assert 0 <= range_of_residues[0] < range_of_residues[1], "range_of_residues should satisfy 0 <= start < end."
            assert range_of_residues[1] <= self.n_residues, f"range_of_residues end should be less than or equal to number of residues ({self.n_residues})."

            from itertools import combinations
            # get a list of all desired residues pairs
            l = list(range(range_of_residues[0], range_of_residues[1]))
            l_pairs = list(combinations(l, 2))
            res_pair_indices = [self.res_ids_to_idx[p] for p in l_pairs if p in self.res_ids_to_idx]
            res_pair_indices = sorted(res_pair_indices)  # sort indices to maintain consistent order
            res_pair_indices = np.array(res_pair_indices)  # convert to numpy array for indexing
            assert len(res_pair_indices) > 0, "No valid residue pairs found in the specified range. Please check the range_of_residues and the protein's residue count."

        
        if mmap_mode:
            if return_mean or remove_mean:
                print('Warning: return_mean and remove_mean are not compatible with mmap_mode. Ignoring both.')
            return_mean, remove_mean = False, False

        print(f'Loading trajectory data (/{len(list_of_trajs)}) from {dir_feat}', end=' ', flush=True)
        
        for i in list_of_trajs:
            print(f'{i}', end=' ', flush=True)
            file_name_npy = dir_feat / self.traj_file_name(feat_scheme, i, suffix='npy')
            file_name_npz = dir_feat / self.traj_file_name(feat_scheme, i, suffix='npz')
            if mmap_mode:
                if res_pair_indices is not None:
                    if file_name_npy.is_file():
                        curr_traj = np.load(file_name_npy, mmap_mode='r')[::stride, res_pair_indices]
                    elif file_name_npz.is_file():
                        with np.load(file_name_npz, mmap_mode='r') as data:
                            curr_traj = data['features'][::stride, res_pair_indices]
                    else:
                        raise FileNotFoundError(f'No .npy or .npz file found for trajectory {i} in {dir_feat}.')
                
                else:
                    if file_name_npy.is_file():
                        curr_traj = np.load(file_name_npy, mmap_mode='r')[::stride]
                    elif file_name_npz.is_file():
                        with np.load(file_name_npz, mmap_mode='r') as data:
                            curr_traj = data['features'][::stride]
                    else:
                        raise FileNotFoundError(f'No .npy or .npz file found for trajectory {i} in {dir_feat}.')

            if not mmap_mode:
                if file_name_npy.is_file():
                    if res_pair_indices is not None:
                        curr_traj = np.load(file_name_npy)[::stride, res_pair_indices]
                    else:
                        curr_traj = np.load(file_name_npy)[::stride]
                elif file_name_npz.is_file():
                    with np.load(file_name_npz) as data:
                        if res_pair_indices is not None:
                            curr_traj = data['features'][::stride, res_pair_indices]
                        else:
                            curr_traj = data['features'][::stride]
                else:
                    raise FileNotFoundError(f'No .npy or .npz file found for trajectory {i} in {dir_feat}.')
            X.append(curr_traj)
        print('Done')

        if return_mean or remove_mean:
            print('Compute mean...', end=' ', flush=True)
            stacked = np.vstack([traj.data for traj in X])
            overall_mean = np.mean(stacked, axis=0).reshape(1, -1)

            if remove_mean:
                print('Remove mean...', end=' ', flush=True)
                for traj in X:
                    traj -= overall_mean
                print('Done')

            if return_mean:
                return X, overall_mean

        return X

import os
import re
import glob
from collections import defaultdict

def get_filenames_matching_pattern(dir_base, pattern, group_by_first_var=True):
    """
    Searches a directory for files matching a pattern, extracts variable values,
    and organizes the results into a consistent 2D list format.

    Returns:
    - ordered_filenames: A nested list of filenames.
      Format is always 2D.
      If group_by_first_var=True:  [[files for i=0], [files for i=1], ...]
      If group_by_first_var=False: [[file1], [file2], [file3], ...] sorted by pattern variables.
    """
    # 1. Split template into text parts and placeholders
    parts = re.split(r'(\{.*?\})', pattern)
    
    glob_pattern = ""
    regex_pattern = "^"
    var_names = []

    # 2. Build the Glob and Regex
    for part in parts:
        if part.startswith('{') and part.endswith('}'):
            var_name = part.strip('{}').split(':')[0]
            var_names.append(var_name)
            glob_pattern += "*"         
            regex_pattern += r"(\d+)"   
        else:
            glob_pattern += part
            regex_pattern += re.escape(part)
            
    regex_pattern += "$"
    regex = re.compile(regex_pattern)

    # Maintain the order of discovered unique variables (e.g., ['run', 'rep'])
    unique_vars = list(dict.fromkeys(var_names))

    # 3. Find candidate files in the file system
    search_path = os.path.join(dir_base, glob_pattern)
    found_files = glob.glob(search_path)
    
    # 4. Extract variables and pair them directly with the actual filepath
    valid_files = [] 
    for filepath in found_files:
        rel_path = os.path.relpath(filepath, dir_base).replace(os.sep, '/')
        match = regex.match(rel_path)
        
        if match:
            file_vars = {}
            is_valid = True
            
            for name, value_str in zip(var_names, match.groups()):
                val_int = int(value_str)
                if name in file_vars and file_vars[name] != val_int:
                    is_valid = False
                    break
                file_vars[name] = val_int
            
            if is_valid:
                valid_files.append((filepath, file_vars))

    # Failsafe for a static string without variables
    if not unique_vars:
        return [[f[0]] for f in sorted(valid_files)]

    ordered_filenames = []

    # 5. Group and format (Always produces a 2D list)
    if group_by_first_var:
        # Scenario 1: Group files by the very first variable (e.g., 'i')
        var_first = unique_vars[0] 
        grouped_by_first = defaultdict(list)
        
        for filepath, file_vars in valid_files:
            grouped_by_first[file_vars[var_first]].append((filepath, file_vars))
            
        # Sort the main groups by the first variable
        for i_val in sorted(grouped_by_first.keys()):
            group = grouped_by_first[i_val]
            
            # Sort the inner list by the remaining variables (e.g., 'j')
            group.sort(key=lambda x: tuple(x[1][v] for v in unique_vars[1:]))
            
            # Extract just the filepaths strings
            ordered_filenames.append([f[0] for f in group])
            
    else:
        # Scenario 2: Treat each file as a single trajectory, no grouping.
        # First, sort globally by ALL unique variables in the order they appear.
        valid_files.sort(key=lambda x: tuple(x[1][v] for v in unique_vars))
        
        # Wrap each individual file in a list to keep the output 2D
        for filepath, file_vars in valid_files:
            ordered_filenames.append([filepath])
            
    return ordered_filenames


def get_random_projection_matrix(d, r, dir_base=None):
    '''
    Get or generate a random projection matrix of shape (d, r).
    If a matrix with the same parameters already exists in the specified directory, it is loaded instead of generated.'''

    filepath = None
    Omega = None
    save = True
    if dir_base is not None:
        dir_base = Path(dir_base)
        dir_base.mkdir(parents=True, exist_ok=True)
        filepath = Path(dir_base) / Path(f'random_projection_matrix_{d,r}.npy')

        if os.path.exists(filepath):
            save = False
            print('loading existing random projection matrix')
            Omega = np.load(filepath)
        
    if Omega is None:
        print('generating new random projection matrix')
        Omega = np.random.normal(size=(d, r))
    
    if filepath is not None and save is True:
        print(f'saving random projection matrix to {filepath}')
        save_to_file(Omega, filepath=filepath)

    return Omega


def plot_eigenfunctions(evecs_K, lifting, domain, W=None, k=1, title=None, efunc_approx=None, path=None):
    """
    Plot the leading eigenfunctions evaluated on domain: a 1D scatter (value
    vs. domain) if domain is 1-dimensional, or 3D scatter plots colored by
    eigenfunction value (using the first 3 domain coordinates) otherwise.

    Parameters:
        evecs_K: right eigenvectors of the (lifted) Koopman/transfer operator,
            shape (n_features, n_eigenfunctions) -- used to combine lifted
            features into eigenfunction values (see evaluate_eigenfunctions).
        lifting: callable mapping domain points to lifted features (e.g. a
            BasisSet instance).
        domain: points to evaluate the eigenfunctions on, shape (n, d).
        W: optional whitening transform applied to the lifted features before
            combining with evecs_K (see evaluate_eigenfunctions).
        k: number of leading eigenfunctions to plot.
        title: extra text appended to the figure's suptitle.
        efunc_approx: precomputed eigenfunction values to plot instead of
            recomputing via evaluate_eigenfunctions (must match domain).
        path: if given, save the figure here (in addition to closing it --
            the figure is never shown interactively).
    """
    d = domain.shape[1] if domain.ndim == 2 else 1

    if efunc_approx is None:
        eigenfunction_approx = evaluate_eigenfunctions(evecs_K, lifting, domain, W)
    else:
        eigenfunction_approx = efunc_approx

    n_eigenfunctions_to_plot = min(k, eigenfunction_approx.shape[1])
    if d == 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(n_eigenfunctions_to_plot):
            if eigenfunction_approx[0, i] < 0:
                eigenfunction_approx[:, i] *= -1
            ax.scatter(domain, eigenfunction_approx[:, i], label=f'Eigenfunction {i}')
    elif d >= 3:
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(
            2, n_eigenfunctions_to_plot,
            height_ratios=[10, 1]
        )

        # --- top row: 3D eigenfunctions ---
        axs = []
        for i in range(n_eigenfunctions_to_plot):
            ax = fig.add_subplot(gs[0, i], projection='3d')
            axs.append(ax)

        # --- shared normalization ---
        vmin = -1 #eigenfunction_approx[:, :n_eigenfunctions_to_plot].min()
        vmax = 1 #eigenfunction_approx[:, :n_eigenfunctions_to_plot].max()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = 'viridis'

        # --- plot ---
        for i, ax in enumerate(axs):
            ax.scatter(
                domain[:, 0],
                domain[:, 1],
                domain[:, 2],
                c=np.real(eigenfunction_approx[:, i]),
                s=3,
                cmap=cmap,
                norm=norm
            )
            ax.set_title(f'Eigenfunction {i}')

        # --- bottom row: ONE axis spanning all columns ---
        cax = fig.add_subplot(gs[1, :])
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation='horizontal',
            #label='Normalized eigenfunction'
        )

    final_title = 'Normalized Eigenfunctions (first 3 coordinates of domain shown)' if d > 3 else 'Normalized Eigenfunctions'
    if title is not None:
        final_title += f', {title}'
    fig.suptitle(final_title)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if path is not None:
        fig.savefig(path)
    plt.close(fig)


def evaluate_eigenfunctions(evecs_K, lifting, domain, W=None):
    """
    Evaluate eigenfunctions on domain: lift domain through `lifting`, apply
    the whitening transform W (if given), project onto evecs_K, then
    normalize each eigenfunction to [-1, 1] by its max absolute value.

    Parameters:
        evecs_K: right eigenvectors, shape (n_features, n_eigenfunctions) --
            n_features must match W's output dimension if W is given, or
            lifting's output dimension otherwise.
        lifting: callable mapping domain points to lifted features.
        domain: points to evaluate on, shape (n, d).
        W: optional whitening transform, shape (n_features_lifted, n_features).

    Returns:
        real-valued array, shape (n, n_eigenfunctions), each column scaled to
        have max absolute value 1.
    """
    lifted_domain = lifting(domain)  # shape (len(domain), n_features)
    if W is not None:
        lifted_domain = lifted_domain @ W
    eigenfunction_approx = lifted_domain @ evecs_K  # shape (len(domain), n_eigenfunctions)
    eigenfunction_approx = np.real(eigenfunction_approx)
    eigenfunction_approx /= np.max(np.abs(eigenfunction_approx), axis=0)
    return eigenfunction_approx


def standardize_and_pool_features(
    data,
    feat_scheme,
    dir_base,
    mean_pooling,
    chunk_size=256,
):
    """
    Optionally mean-pool raw BioEmu_L1_features trajectories over residues,
    then standardize (per-channel zero mean / unit variance, cached to disk).

    Pooling is done *before* standardizing (when applicable) rather than
    after: standardization statistics are computed over whatever `data` is
    at that point, so pooling first means every frame -- regardless of how
    many residues its source protein has -- contributes equally to the
    statistics. Standardizing raw per-residue values first would instead
    weight the statistics by each trajectory's residue count, letting
    larger proteins dominate the normalization applied to every trajectory,
    including smaller ones. This matters whenever trajectories from
    differently-sized proteins are combined in one call.

    Parameters:
        data: list of trajectory arrays, one per trajectory. Shape
            (n_frames, n_residues, n_channels) for 'BioEmu_L1_features', or
            (n_frames, n_channels) if the scheme is already pooled.
            n_residues may differ between trajectories (e.g. different-sized
            proteins) -- only n_channels must match across all of them.
        feat_scheme: 'BioEmu_L1_features' (per-residue) or
            'BioEmu_L1_features_pooled' (already pooled) -- determines the
            expected array dimensionality and the caching subdirectory.
        dir_base: directory under which cached stats/standardized trajectories
            are stored, at f'{dir_base}/{feat_scheme}/processed_features(_pooled)/'.
        mean_pooling: if True and the scheme isn't already pooled, mean-pool
            each trajectory over its own residues before standardizing.
        chunk_size: number of frames processed per chunk when computing
            standardization statistics / standardizing (bounds memory use).

    Returns:
        list of standardized (and optionally mean-pooled) trajectory arrays,
        one per input trajectory, in the same order as `data`. If
        mean_pooling is used, trajectories from differently-sized proteins
        all end up with the same n_channels-dimensional per-frame vector and
        can be combined in one TICA fit despite differing residue counts.
    """
    expected_ndim = 3 if feat_scheme == 'BioEmu_L1_features' else 2

    if not data:
        raise ValueError("No trajectories were provided.")

    # Validate input dimensionality and channel count. n_residues (axis 1 of
    # the 3D/unpooled case) may differ per trajectory -- only n_channels
    # (the last axis) needs to match across all of them.
    n_channels = data[0].shape[-1]

    for traj_idx, traj in enumerate(data):
        if traj.ndim != expected_ndim:
            raise ValueError(
                f"Trajectory {traj_idx} has shape {traj.shape}, but "
                f"{feat_scheme} expects {expected_ndim} dimensions."
            )

        if traj.shape[-1] != n_channels:
            raise ValueError(
                f"Trajectory {traj_idx} has {traj.shape[-1]} channels, "
                f"but the first trajectory has {n_channels}."
            )

    pool_before_standardizing = mean_pooling and expected_ndim == 3

    if pool_before_standardizing:
        print("Mean-pooling each trajectory over its own residues before "
              "standardizing (so standardization weights every frame "
              "equally, regardless of residue count)...")
        data = [traj.mean(axis=1, dtype=np.float32) for traj in data]
        print("Pooled trajectory shapes:", [traj.shape for traj in data])

    # From this point on, `data` is 2D per trajectory whenever pooling
    # happened above (or the scheme was already pooled coming in).
    stats_ndim = 2 if pool_before_standardizing else expected_ndim

    cache_subdir = "processed_features_pooled" if pool_before_standardizing else "processed_features"
    processed_dir = Path(dir_base) / feat_scheme / cache_subdir
    processed_dir.mkdir(parents=True, exist_ok=True)

    stats_file = processed_dir / "standardization_stats.npz"

    expected_files = [
        processed_dir / f"traj_{traj_idx}_standardized_flat.npy"
        for traj_idx in range(len(data))
    ]

    # Both feature schemes are ultimately stored as:
    #
    #     n_frames x flattened_feature_dimension
    #
    # For pooled features, flattened_feature_dimension == n_channels.
    def get_output_shape(traj):
        flattened_dim = int(np.prod(traj.shape[1:]))
        return traj.shape[0], flattened_dim

    # Shape used for broadcasting channel-wise statistics:
    #
    # Unpooled: (1, 1, n_channels)
    # Pooled:   (1, n_channels)
    stats_shape = (1,) * (stats_ndim - 1) + (n_channels,)

    cache_available = stats_file.exists()

    # Check that the statistics file is compatible.
    if cache_available:
        try:
            with np.load(stats_file) as stats:
                required_keys = {"mean", "std", "std_safe"}

                if not required_keys.issubset(stats.files):
                    print(f"Statistics file is missing keys: {required_keys - set(stats.files)}")
                    cache_available = False

                elif stats["mean"].shape != stats_shape:
                    print(f"Mean shape mismatch: {stats['mean'].shape} != {stats_shape}")
                    cache_available = False

                elif stats["std"].shape != stats_shape:
                    print(f"Standard-deviation shape mismatch: {stats['std'].shape} != {stats_shape}")
                    cache_available = False

                elif stats["std_safe"].shape != stats_shape:
                    print(f"Safe-standard-deviation shape mismatch: {stats['std_safe'].shape} != {stats_shape}")
                    cache_available = False

        except (OSError, ValueError) as exc:
            print(f"Could not read statistics cache: {exc}")
            cache_available = False

    # Check whether all cached trajectories exist and have the
    # expected shapes.
    if cache_available:
        print("Per-channel standardized data available.")

        for traj, cache_file in zip(data, expected_files):
            if not cache_file.exists():
                print(f"Missing cached trajectory: {cache_file}")
                cache_available = False
                break

            cached = np.load(cache_file, mmap_mode="r")
            expected_shape = get_output_shape(traj)

            if cached.shape != expected_shape:
                print(f"Cache shape mismatch for {cache_file}: {cached.shape} != {expected_shape}")
                cache_available = False
                del cached
                break

            del cached

    if cache_available:
        print("Loading cached standardized features.")

        with np.load(stats_file) as stats:
            mean = stats["mean"].copy()
            std_safe = stats["std_safe"].copy()

        data_for_tica_base = [
            np.load(cache_file, mmap_mode="r")
            for cache_file in expected_files
        ]

    else:
        print("Cached standardized features not found or incompatible; computing them.")

        channel_sum = np.zeros(n_channels, dtype=np.float64)
        channel_sq_sum = np.zeros(n_channels, dtype=np.float64)
        count = 0

        # Compute channel-wise statistics.
        #
        # For (frames, residues, channels), reduction_axes=(0, 1).
        # For (frames, channels),           reduction_axes=(0,).
        for traj_idx, traj in enumerate(data):
            print(f"Computing statistics for trajectory {traj_idx + 1}/{len(data)}: {traj.shape}")

            for start in range(0, traj.shape[0], chunk_size):
                stop = min(start + chunk_size, traj.shape[0])

                chunk = np.asarray(traj[start:stop], dtype=np.float64)

                reduction_axes = tuple(range(chunk.ndim - 1))

                channel_sum += chunk.sum(axis=reduction_axes)

                # Sums the squared values over every dimension except
                # the final channel dimension. This works for both
                # 2-D and 3-D feature arrays.
                channel_sq_sum += np.einsum("...c,...c->c", chunk, chunk, optimize=True)

                count += int(np.prod(chunk.shape[:-1]))

        mean_1d = channel_sum / count

        variance_1d = np.maximum(channel_sq_sum / count - mean_1d**2, 0.0)

        std_1d = np.sqrt(variance_1d)

        small_std = std_1d < 1e-10
        print("Near-constant channels:", np.where(small_std)[0])

        std_safe_1d = np.where(small_std, 1.0, std_1d)

        # Reshape for broadcasting (stats_shape already accounts for whether
        # `data` was pooled to 2D above):
        #
        # Unpooled (n_frames, n_residues, n_channels): stats shape (1, 1, n_channels)
        # Pooled or already-pooled (n_frames, n_channels): stats shape (1, n_channels)
        mean = mean_1d.reshape(stats_shape)
        std = std_1d.reshape(stats_shape)
        std_safe = std_safe_1d.reshape(stats_shape)

        np.savez(
            stats_file,
            mean=mean,
            std=std,
            std_safe=std_safe,
            count=count,
            feat_scheme=feat_scheme,
        )

        mean_float32 = mean.astype(np.float32, copy=False)
        std_safe_float32 = std_safe.astype(np.float32, copy=False)

        data_for_tica_base = []

        # Standardize and store each trajectory as a 2-D array.
        for traj_idx, (traj, cache_file) in enumerate(zip(data, expected_files)):
            n_frames = traj.shape[0]
            flattened_dim = int(np.prod(traj.shape[1:]))

            print(f"Standardizing trajectory {traj_idx + 1}/{len(data)}: {traj.shape}")

            standardized_flat = np.lib.format.open_memmap(
                cache_file,
                mode="w+",
                dtype=np.float32,
                shape=(n_frames, flattened_dim),
            )

            for start in range(0, n_frames, chunk_size):
                stop = min(start + chunk_size, n_frames)

                chunk = np.asarray(traj[start:stop], dtype=np.float32)

                standardized_chunk = (chunk - mean_float32) / std_safe_float32

                standardized_flat[start:stop] = standardized_chunk.reshape(stop - start, flattened_dim)

            standardized_flat.flush()
            del standardized_flat

            data_for_tica_base.append(np.load(cache_file, mmap_mode="r"))

    return data_for_tica_base