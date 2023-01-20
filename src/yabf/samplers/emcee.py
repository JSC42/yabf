"""The emcee sampler wrapper into yabf."""
import numpy as np
from cached_property import cached_property
from emcee import EnsembleSampler
from emcee import backends # JSC
from getdist import MCSamples
import os

from ..core import mpi
from ..core.samplers import Sampler, run_map


class emcee(Sampler):
    @cached_property
    def nwalkers(self):
        return self.sampler_kwargs.pop("nwalkers", self.nparams * 2)

    def _get_sampler(self, **kwargs):
        # This is bad, but I have to access this before passing kwargs,
        # otherwise nwalkers is passed twice.
        if "nwalkers" in kwargs:
            del kwargs["nwalkers"]
        # Preparing HDF backend
        ChainFile = str(self._output_dir) + '/' + str(self._output_prefix) + '_EMCEE_Chains.h5'
        try:
            os.remove(ChainFile)
        except FileNotFoundError:
            pass
        backend = backends.HDFBackend(ChainFile)

        return EnsembleSampler(
            log_prob_fn=self.likelihood,
            ndim=self.nparams,
            nwalkers=self.nwalkers,
            backend=backend,
            **kwargs,
        )

    def _get_sampling_fn(self, sampler):
        return sampler.run_mcmc

    def _sample(
        self,
        sampling_fn,
        downhill_first=False,
        bounds=None,
        restart=False,
        refs=None,
        **kwargs,
    ):

        if not restart:
            try:
                sampling_fn(None, **kwargs)
                return self._sampler
            except Exception:
                pass

        if refs is None:
            if not downhill_first:
                refs = np.array(self.likelihood.generate_refs(n=self.nwalkers)).T
            else:
                res = run_map(self.likelihood, bounds=bounds)
                refs = np.random.multivariate_normal(
                    res.x, res.hess_inv, size=(self.nwalkers,)
                )

        sampling_fn(refs, **kwargs)

        return self._sampler

    def _samples_to_mcsamples(self, samples):
        # 'samples' is whatever is returned by _sample, in this case it is
        # the entire EnsembleSampler
        return MCSamples(
            samples=samples.get_chain(flat=False),
            names=[a.name for a in self.likelihood.child_active_params],
            labels=[p.latex for p in self.likelihood.child_active_params],
        )
