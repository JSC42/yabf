"""
Module defining the API for Samplers
"""

import os
import warnings

import attr
import numpy as np
import pypolychord as ppc
from attr.validators import instance_of
from cached_property import cached_property
from emcee import EnsembleSampler
from getdist import MCSamples
from pypolychord.priors import UniformPrior
from pypolychord.settings import PolyChordSettings
from scipy.optimize import minimize

from yabf.core import yaml
from yabf.core.likelihood import LikelihoodInterface
from . import mpi
from .plugin import plugin_mount_factory


@attr.s
class Sampler(metaclass=plugin_mount_factory()):
    likelihood = attr.ib(validator=instance_of(LikelihoodInterface))
    _output_dir = attr.ib(default="", validator=instance_of(str))
    _output_prefix = attr.ib(validator=instance_of(str))
    _save_full_config = attr.ib(default=True, converter=bool)
    sampler_kwargs = attr.ib(default={})

    def __attrs_post_init__(self):
        self.mcsamples = None

        # Save the configuration
        if self._save_full_config:
            with open(self.config_filename, 'w') as fl:
                yaml.dump(self.likelihood, fl)

    @_output_prefix.default
    def _op_default(self):
        return self.likelihood.name

    @cached_property
    def output_dir(self):
        dir = os.path.join(self._output_dir, os.path.dirname(self._output_prefix))

        # Try to create the directory.
        try:
            os.makedirs(dir)
            warnings.warn("Proposed output directory '{}' did not exist. Created it.".format(dir))
        except (FileExistsError, FileNotFoundError):
            pass
        return dir

    @cached_property
    def output_file_prefix(self):
        return os.path.basename(self._output_prefix)

    @cached_property
    def config_filename(self):
        return os.path.join(self.output_dir, self.output_file_prefix) + "_config.yml"

    @cached_property
    def nparams(self):
        return self.likelihood.total_active_params

    @cached_property
    def _sampler(self):
        return self._get_sampler(**self.sampler_kwargs)

    @cached_property
    def _sampling_fn(self):
        return self._get_sampling_fn(self._sampler)

    def _get_sampler(self, **kwargs):
        """
        Return an object that contains the sampling settings and potentially a
        method to actually perform sampling

        This could actually be nothing, and the class could rely purely
        on the :func:`_get_sampling_fn` method to create the sampler.
        """
        return None

    def _get_sampling_fn(self, sampler):
        pass

    def sample(self, **kwargs):
        samples = self._sample(self._sampling_fn, **kwargs)

        mpi.sync_processes()

        self.mcsamples = self._samples_to_mcsamples(samples)
        return self.mcsamples

    def _sample(self, sampling_fn, **kwargs):
        pass

    def _samples_to_mcsamples(self, samples):
        """Return posterior samples, with shape (<...>, NPARAMS, NITER)"""
        pass


class emcee(Sampler):

    @cached_property
    def nwalkers(self):
        return self.sampler_kwargs.pop("nwalkers", self.nparams * 2)

    def _get_sampler(self, **kwargs):
        # This is bad, but I have to access this before passing kwargs,
        # otherwise nwalkers is passed twice.
        if 'nwalkers' in kwargs:
            del kwargs['nwalkers']

        return EnsembleSampler(
            log_prob_fn=self.likelihood,
            ndim=self.nparams,
            nwalkers=self.nwalkers,
            **kwargs
        )

    def _get_sampling_fn(self, sampler):
        return sampler.run_mcmc

    def _sample(self, sampling_fn, downhill_first=False, bounds=None, restart=False,
                **kwargs):

        if not restart:
            try:
                sampling_fn(None, **kwargs)
                return self._sampler
            except:
                pass

        if not downhill_first:
            refs = np.array(self.likelihood.generate_refs(n=self.nwalkers))
        else:
            res = run_map(self.likelihood, bounds=bounds)
            refs = np.random.multivariate_normal(res.x, res.hess_inv,
                                                 size=(self.nwalkers, len(res.x)))

        sampling_fn(
            refs.T,
            **kwargs
        )

        return self._sampler

    def _samples_to_mcsamples(self, samples):
        # 'samples' is whatever is returned by _sample, in this case it is
        # the entire EnsembleSampler
        return MCSamples(
            samples=samples.get_chain(flat=False),
            names=[a.name for a in self.likelihood.child_active_params],
            labels=[p.latex for p in self.likelihood.child_active_params]
        )


class polychord(Sampler):
    @staticmethod
    def _flat_array(elements):
        lst = []

        if hasattr(elements, "__len__"):
            for e in elements:
                lst += polychord._flat_array(e)
        else:
            lst.append(elements)

        return lst

    @cached_property
    def __derived_sample(self):
        # a bad hack to get a sample of derived quantities
        # in order to understand the shape of the derived quantities
        return self.likelihood()[1]

    @cached_property
    def derived_shapes(self):
        return [getattr(d, "shape", 0) for d in self.__derived_sample]

    @cached_property
    def nderived(self):
        # this is a bad hack
        return len(self._flat_array(self.__derived_sample))

    @cached_property
    def log_prior_volume(self):
        prior_volume = 0
        for p in self.likelihood.child_active_params:
            if np.isinf(p.min) or np.isinf(p.max):
                raise ValueError("Polychord requires bounded priors")
            prior_volume += np.log(p.max - p.min)
        return prior_volume

    @cached_property
    def prior(self):
        # Determine proper prior.
        def pr(hypercube):
            ret = []
            for p, h in zip(self.likelihood.child_active_params, hypercube):
                ret.append(UniformPrior(p.min, p.max)(h))
            return ret

        return pr

    @cached_property
    def posterior(self):
        def posterior(p):
            lnl, derived = self.likelihood(p)
            return max(lnl + self.log_prior_volume, 0.99 * np.nan_to_num(-np.inf)), \
                   np.array(self._flat_array(derived))

        return posterior

    @staticmethod
    def _index_to_string(*indx):
        return "_" + "_".join([str(i) for i in indx])

    @staticmethod
    def _index_to_latex(*indx):
        return r"_{" + ",".join([str(i) for i in indx]) + r"}"

    def get_derived_paramnames(self):
        """Creates a list of tuples specifying derived parameter names"""
        names = []
        for name, shape in zip(self.likelihood.derived, self.derived_shapes):
            if shape == 0:
                names.append((name, name))
            else:
                names.extend([
                    (name + self._index_to_string(*ind),
                     name + self._index_to_latex(*ind))
                    for ind in np.ndindex(*shape)])
        return names

    def _make_paramnames_files(self, mcsamples):
        paramnames = [(p.name, p.latex) for p in self.likelihood.child_active_params]

        # also have to add derived...
        paramnames += self.get_derived_paramnames()
        mcsamples.make_paramnames_files(paramnames)

    def _get_sampler(self, **kwargs):
        if "file_root" in kwargs:
            warnings.warn("file_root was defined in sampler_kwargs, but is replaced by output_prefix")
            del kwargs['file_root']

        if "base_dir" in kwargs:
            warnings.warn("base_dir was defined in sampler_kwargs, but is replaced by output_prefix")
            del kwargs['base_dir']

        return PolyChordSettings(
            self.nparams, self.nderived, base_dir=self.output_dir,
            file_root=self.output_file_prefix, **kwargs
        )

    def _get_sampling_fn(self, sampler):
        return ppc.run_polychord

    def _sample(self, sampling_fn, **kwargs):
        settings = self._sampler

        return sampling_fn(
            self.posterior, self.nparams, self.nderived,
            settings=settings, prior=self.prior
        )

    def _samples_to_mcsamples(self, samples):
        if mpi.am_single_or_primary_process:
            self._make_paramnames_files(samples)
            # do initialization...
            samples.posterior

        mpi.sync_processes()
        return samples.posterior


def run_map(likelihood, x0=None, bounds=None):
    """
    Run a maximum a-posteriori fit.
    """

    def objfunc(p):
        return -likelihood.logp(params=p)

    if x0 is None:
        x0 = np.array([apar.fiducial for apar in likelihood.child_active_params])

    if bounds is None:
        bounds = []
        for apar in likelihood.child_active_params:
            bounds.append((apar.min if apar.min > -np.inf else None,
                           apar.max if apar.max < np.inf else None))

    elif not bounds:
        bounds = None

    res = minimize(objfunc, x0, bounds=bounds)
    return res
