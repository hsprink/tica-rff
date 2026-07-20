# this module contains helper functions and classes to generate and store RFF parameters (omega and b)
# and to create basis functions from these parameters

from pathlib import Path
import numpy as np
from util import set_attrs_from_dict, save_to_file, load_from_file


# ============================================================
# Registry and Basis Set Construction
# ============================================================

# global registry: maps basis_type strings to constructor functions
BASIS_REGISTRY = {}

def register_basis(name):
    """Decorator to register a basis constructor."""
    def wrapper(func):
        BASIS_REGISTRY[name] = func
        return func
    return wrapper

@register_basis("rff_real")
def make_rff_real(params):
    # Depending on how you pass params, you can adapt this.
    # Assumes params has .matrix and .bias (like the ParameterRFF classes below)
    matrix = getattr(params, 'matrix', params.get('M') if isinstance(params, dict) else None)
    bias = getattr(params, 'bias', params.get('b') if isinstance(params, dict) else None)
    return basisfunction_rff_real(matrix, bias)

@register_basis("rff_cplx")
def make_rff_cplx(params):
    matrix = getattr(params, 'matrix', params.get('M') if isinstance(params, dict) else None)
    return basisfunction_rff_cplx(matrix)

@register_basis("polynomial")
def make_polynomial(params):
    degree = params.get("degree", 1)
    def featurize(x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return np.hstack([x**k for k in range(1, degree+1)])
    return featurize

@register_basis("gaussian")
def make_gaussian_lift(params):
    centers = params.get("centers")
    sigma = params.get("sigma", 1.0)
    def featurize(x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return np.exp(- (x - centers)**2 / (2 * sigma**2))
    return featurize

@register_basis("gaussian_normalized")
def make_gaussian_normalized(params):
    centers = params.get("centers")
    sigma = params.get("sigma", 1.0)
    def featurize(x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        unnormalized = np.exp(- (x - centers)**2 / (2 * sigma**2)) # shape (m, n_features)
        s = np.sum(unnormalized, axis=1, keepdims=True) # shape (m, 1)
        return unnormalized / s
    return featurize

@register_basis("stump")
def make_stump_lift(params):
    p = params.get('p')
    a = params.get('a')
    # Use existing 't' if provided, otherwise sample uniformly
    t = params.get('t', np.random.uniform(-a, a, p) if a is not None else None)

    def featurize(x):
        lifted_x = np.zeros((x.shape[0], p))
        for i in range(p):
            lifted_x[:, i] = np.sign(x - t[i])
        return lifted_x
    return featurize

@register_basis("bins")
def make_bins(params):
    a = params.get("a")
    delta = params.get("delta")
    u = params.get("u")
    def featurize(x):
        lifted = []
        for dlt, uu in zip(delta, u):
            idx = np.floor((x + a - uu) / dlt).astype(int)
            lifted.append(idx)
        return lifted
    return featurize


class BasisSet:
    """
    General wrapper: maps (params, basis_type) -> feature map f(x).
    """

    def __init__(self, params, basis_type, name=None):
        if basis_type not in BASIS_REGISTRY:
            raise ValueError(f"Unknown basis type: {basis_type}")

        self.params = params
        self.basis_type = basis_type
        self.name = name or basis_type

        # build featurizer
        constructor = BASIS_REGISTRY[basis_type]
        self.featurize = constructor(params)

    def __call__(self, x):
        """
        Apply the featurizer to x. If x is a list (one array per trajectory),
        featurize each trajectory independently and return a list of the same
        length; otherwise apply directly to the single array.
        """
        if isinstance(x, list):
            return [self.featurize(d) for d in x]
        else:
            return self.featurize(x)
    

def kernel_approximation(x0, x, rff_dictionary, dict_type=None):
    lifted_x0 = rff_dictionary(x0)
    lifted_x = rff_dictionary(x)

    if dict_type == 'bins':
        ans = 0
        for curr_feat_x0, curr_feat_x in zip(lifted_x0, lifted_x):
            ans += 1.0 if curr_feat_x0 == curr_feat_x else 0.0
        return ans / len(lifted_x0)
    else:
        return float(np.dot(lifted_x0, lifted_x.T) / lifted_x0.shape[1])

def kernel(kernel_type, sigma):
    if kernel_type == 'gaussian':
        def function(x, y):
            return np.exp(- (x - y)**2 / (2 * sigma**2))
    elif kernel_type == 'laplacian':
        def function(x, y):
            return np.exp(-np.abs(x - y)/sigma)
    elif kernel_type == 'cauchy':
        def function(x, y):
            return 1 / (1 + (((x - y) / sigma) **2))
    else:
        raise ValueError("Unknown kernel type.")
    return function


# ============================================================
# Base class for RFF parameters
# ============================================================

class ParameterRFF():
    """
    Base class for kernel-specific RFF basis sets. Used for generating and filing RFF parameters (omega and b).
    But can also be used without file handling to just generate parameters on the fly.
    """

    def __init__(self, d, p, scaling, dir_base=None, generate_new=False):
        """
        Parameters:
            d: input dimension (number of features per sample being lifted).
            p: number of random Fourier features to sample.
            scaling: kernel bandwidth parameter, meaning depends on the
                subclass's sample_spectral_matrix (e.g. for GaussianRFF, the
                spectral matrix is scaled by 1/scaling, so larger scaling
                means a narrower/more local kernel).
            dir_base: if given, cache the generated (matrix, bias) under this
                directory, keyed by (d, p, scaling); if None, always generate
                fresh parameters without touching disk.
            generate_new: if True, ignore any cached parameters and regenerate.
        """
        self.d = d
        self.p = p
        self.scaling = scaling
        self.dir_base = dir_base

        self.matrix = None
        self.bias   = None

        if dir_base is not None:
            self.set_path_structure()
            self.load_or_generate(generate_new)
        else:
            self.generate_parameter()
 
    def set_path_structure(self):
        self.filename = self.get_filename(**self.parameters_for_filename)
        self.path_final = Path(self.dir_base) / Path(self.foldername) / Path(self.filename)
        self.path_final.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_filename(**kwargs):
        name_parts = [f"{k}={v}" for k, v in kwargs.items()]
        return "_".join(name_parts) + ".pickle"

    def sample_spectral_matrix(self):
        raise NotImplementedError

    def generate_parameter(self):
        print(f"Generating matrix and bias vector for {self.__class__.__name__}, d={self.d}, p={self.p} ... ", end="")
        self.matrix = self.sample_spectral_matrix()
        self.bias = np.random.uniform(0, 2*np.pi, size=(self.p, 1))
        print("Done.")

    def load_or_generate(self, generate_new=False):
        if (not generate_new) and self.path_final.is_file():
            data = load_from_file(self.path_final)
            set_attrs_from_dict(self, data)
            print(f"Loaded matrix and bias for d={self.d}, p={self.p} from {self.path_final}")
        else:
            self.generate_parameter()
            save_to_file(self, filepath=self.path_final, attrs_to_save=self.parameters_for_saving)

class GaussianRFF(ParameterRFF):
    """
    Random Fourier Features for the Gaussian/RBF kernel: samples the
    spectral matrix Omega from N(0, scaling^-2 I), matching Bochner's theorem
    for exp(-||x-y||^2 / (2*scaling^2)).
    """

    foldername = "gaussian_rff"

    def __init__(self, d, p, scaling, dir_base=None, generate_new=False):
        self.parameters_for_filename = dict(kernel="gaussian", d=d, p=p, scaling=scaling)
        self.parameters_for_saving   = {"matrix", "bias", "d", "p", "scaling"}
        super().__init__(d, p, scaling, dir_base, generate_new)

    def sample_spectral_matrix(self):
        return (1.0 / self.scaling) * np.random.randn(self.d, self.p)

class LaplacianRFF(ParameterRFF):
    foldername = "laplacian_rff"

    def __init__(self, d, p, scaling, dir_base=None, generate_new=False):
        self.parameters_for_filename = dict(kernel="laplacian", d=d, p=p, scaling=scaling)
        self.parameters_for_saving   = {"matrix", "bias", "d", "p", "scaling"}
        super().__init__(d, p, scaling, dir_base, generate_new)

    def sample_spectral_matrix(self):
        scale = 1.0 / self.scaling
        return scale * np.random.standard_cauchy(size=(self.d, self.p))
    
class CauchyRFF(ParameterRFF):
    foldername = "cauchy_rff"

    def __init__(self, d, p, scaling, dir_base=None, generate_new=False):
        self.parameters_for_filename = dict(kernel="cauchy", d=d, p=p, scaling=scaling)
        self.parameters_for_saving   = {"matrix", "bias", "d", "p", "scaling"}
        super().__init__(d, p, scaling, dir_base, generate_new)

    def sample_spectral_matrix(self):
        scale = 1.0 / self.scaling
        rng = np.random.default_rng()
        return rng.laplace(loc=0, scale=scale, size=(self.d, self.p))


# ============================================================
# Basis function builders
# ============================================================

def basisfunction_rff_cplx(spectral_matrix):
    d, p = spectral_matrix.shape
    def featurize(x):
        assert x.ndim in [1, 2], "Input x must be 1D or 2D array."
        if x.ndim == 1:
            assert d == 1, "If input x is 1D, then d must be 1."
            x = x[:, np.newaxis]
        ans = np.exp(-1j * np.dot(x, spectral_matrix))
        return ans
    return featurize


def basisfunction_rff_real(spectral_matrix, offset_vector):
    """
    Build a real-valued random-Fourier-feature map
    phi(x) = sqrt(2) * cos(x @ Omega + b), which approximates the shift-
    invariant kernel associated with Omega's sampling distribution (e.g.
    Gaussian) via Bochner's theorem, without needing complex arithmetic.

    Parameters:
        spectral_matrix: Omega, shape (d, p) -- samples from the kernel's
            spectral density (see e.g. GaussianRFF.sample_spectral_matrix).
        offset_vector: b, shape (p, 1) -- phase offsets, uniform on [0, 2*pi);
            needed so E[phi(x) phi(y)^T] recovers the kernel exactly.

    Returns:
        featurize(x): callable mapping x, shape (n, d), to lifted features,
        shape (n, p).
    """
    d, p = spectral_matrix.shape
    assert p in offset_vector.shape

    def featurize(x):
        assert x.ndim in [1, 2], "Input x must be 1D or 2D array."
        if x.ndim == 1:
            assert d == 1, "If input x is 1D, then d must be 1."
            x = x[:, np.newaxis]
        lifted_x = np.sqrt(2) * np.cos(np.dot(x, spectral_matrix) + offset_vector.T)
        return lifted_x

    return featurize
    