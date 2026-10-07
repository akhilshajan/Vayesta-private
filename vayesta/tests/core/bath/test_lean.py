"""Lean path (bath_options["lean"]=True, core/bath/lean.py) against the standard path."""

import functools
import unittest

import numpy as np
import pyscf.gto
import pyscf.scf

import vayesta
import vayesta.ewf
from vayesta.tests.common import TestCase


@functools.cache
def water_chain(n=8, basis="sto-3g"):
    """Chain of n waters, 2.9 A apart, so that rcut 5 A leaves a far environment."""
    atoms = []
    for i in range(n):
        x = 2.9 * i
        atoms += [("O", (x, 0, 0)), ("H", (x + 0.76, 0.59, 0)), ("H", (x - 0.24, -0.9, 0.1))]
    mol = pyscf.gto.M(atom=atoms, basis=basis, verbose=0)
    mf = pyscf.scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def run(mf, lean, bathtype, atoms, eta=1e-5, **kwargs):
    opts = dict(bathtype=bathtype, threshold=eta, rcut=5.0, lean=lean, **kwargs)
    if bathtype == "mp2coupled":
        opts["coupled_weights"] = "scs"
    emb = vayesta.ewf.EWF(mf, solver="MP2", bath_options=opts)
    with emb.iao_fragmentation() as f:
        for a in atoms:
            f.add_atomic_fragment(a)
    emb.kernel()
    return emb


class LeanPathTest(TestCase):
    atoms = (9, 10, 12, 13)  # O, H of the 4th water and O, H of the 5th: far environment on both sides

    def _compare(self, mf, bathtype, e_tol, sv_tol):
        std = run(mf, False, bathtype, self.atoms)
        lean = run(mf, True, bathtype, self.atoms)
        ovlp = mf.get_ovlp()
        self.assertAlmostEqual(lean.e_corr, std.e_corr, delta=e_tol)
        for fs, fl in zip(std.fragments, lean.fragments):
            cs, cl = fs.cluster, fl.cluster
            self.assertIsNone(fl.c_env)
            self.assertIsNone(fl._dmet_bath.c_env_occ)
            c = np.hstack((fl.c_frag, fl._dmet_bath.c_dmet))
            self.assertAllclose(c.T @ ovlp @ c, np.eye(c.shape[-1]), atol=1e-12, rtol=0)
            self.assertEqual(cl.norb_active, cs.norb_active)
            self.assertEqual(cl.nocc_active, cs.nocc_active)
            # same active space
            sv = np.linalg.svd(cs.c_active.T @ ovlp @ cl.c_active, compute_uv=False)
            self.assertGreater(sv.min(), 1 - sv_tol)
            self.assertAlmostEqual(fl.results.e_corr, fs.results.e_corr, delta=e_tol)
            # the far environment is implicit: counts add up to the full environment
            n_frozen_std = cs.c_frozen_occ.shape[-1], cs.c_frozen_vir.shape[-1]
            n_frozen_lean = (cl.c_frozen_occ.shape[-1] + cl.lean_nocc_far, cl.c_frozen_vir.shape[-1] + cl.lean_nvir_far)
            self.assertEqual(n_frozen_lean, n_frozen_std)
            # full frozen density from the mean field (standard: from the explicit environment, which
            # is occupied only to within the DMET threshold)
            self.assertAllclose(cl.make_frozen_rdm1(), cs.make_frozen_rdm1(), atol=1e-4, rtol=0)
            # the lean frozen density + the active density reproduce the mean-field density exactly
            dm_act = 2 * cl.c_active_occ @ cl.c_active_occ.T
            self.assertAllclose(cl.make_frozen_rdm1() + dm_act, mf.make_rdm1(), atol=1e-10, rtol=0)
        return std, lean

    def test_onesided_sto3g(self):
        self._compare(water_chain(8, "sto-3g"), "mp2", e_tol=1e-9, sv_tol=1e-6)

    def test_coupled_sto3g(self):
        std, lean = self._compare(water_chain(8, "sto-3g"), "mp2coupled", e_tol=1e-9, sv_tol=1e-6)
        # dMP2 / dSCS-MP2 corrections, as used by the production pipeline
        for fs, fl in zip(std.fragments, lean.fragments):
            ds = fs._bath_factory_occ.get_delta_mp2(fs.cluster)
            dl = fl._bath_factory_occ.get_delta_mp2(fl.cluster)
            for key in ("e_mp2_full", "d_mp2", "d_scs"):
                self.assertAlmostEqual(dl[key], ds[key], delta=1e-9)

    def test_coupled_631g(self):
        self._compare(water_chain(8, "6-31g"), "mp2coupled", e_tol=1e-8, sv_tol=1e-6)

    def test_r2_blocks(self):
        """The blocked <r^2> contraction does not depend on the block size."""
        mf = water_chain(8, "sto-3g")
        e = [run(mf, True, "mp2coupled", self.atoms[:1], lean_r2_block=b).e_corr for b in (7, 1000)]
        self.assertAlmostEqual(e[0], e[1], delta=1e-12)

    def test_default_is_standard(self):
        """Without bath_options['lean'] nothing changes: fragments keep their explicit environment."""
        emb = run(water_chain(8, "sto-3g"), False, "mp2", self.atoms[:1])
        self.assertFalse(emb.opts.bath_options["lean"])
        self.assertIsNotNone(emb.fragments[0].c_env)
        self.assertIsNotNone(emb.fragments[0]._dmet_bath.c_env_occ)

    def test_unsupported(self):
        mf = water_chain(8, "sto-3g")
        with self.assertRaises(ValueError):  # rcut is required
            emb = vayesta.ewf.EWF(mf, solver="MP2", bath_options=dict(bathtype="mp2", threshold=1e-5, lean=True))
            with emb.iao_fragmentation() as f:
                f.add_atomic_fragment(0)
            emb.kernel()
        with self.assertRaises(NotImplementedError):  # DMET-only bath
            emb = vayesta.ewf.EWF(mf, solver="MP2", bath_options=dict(bathtype="dmet", rcut=5.0, lean=True))
            with emb.iao_fragmentation() as f:
                f.add_atomic_fragment(0)
            emb.kernel()


if __name__ == "__main__":
    unittest.main()
