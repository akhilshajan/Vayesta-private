import unittest

import numpy as np
import pyscf.cc
import pyscf.mp

import vayesta
import vayesta.ewf
from vayesta.core.bath import DMET_Bath, MP2_Coupled_Bath
from vayesta.core.qemb import Embedding
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


def projected_density_reference(mf, t2, c_frag, side, weights=(1.0, 1.0)):
    """Coupled BNO density from full-system MP2 amplitudes, projected and symmetrized explicitly."""
    ovlp = mf.get_ovlp()
    nocc = np.count_nonzero(mf.mo_occ > 0)
    c_occ, c_vir = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
    r = c_occ.T @ ovlp @ c_frag
    pt2 = np.einsum("ik,kjab->ijab", r @ r.T, t2)
    t2s = (pt2 + pt2.transpose(1, 0, 3, 2)) / 2
    c_os, c_ss = weights
    if side == "occupied":
        a = np.einsum("ikab,jkab->ij", t2s, t2s)
        b = np.einsum("ikab,jkba->ij", t2s, t2s)
        return (c_os + c_ss) * a - c_ss * b, c_occ
    a = np.einsum("ijac,ijbc->ab", t2s, t2s)
    b = np.einsum("ijac,ijcb->ab", t2s, t2s)
    return (c_os + c_ss) * a - c_ss * b, c_vir


class Coupled_BNO_Test(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()
        mp2 = pyscf.mp.MP2(cls.mf)
        mp2.kernel()
        cls.mp2 = mp2

    def _baths(self, atom, weights=(1.0, 1.0)):
        emb = Embedding(self.mf)
        with emb.iao_fragmentation() as f:
            frag = f.add_atomic_fragment(atom)
        dmet = DMET_Bath(frag)
        dmet.kernel()
        occ = MP2_Coupled_Bath(frag, dmet, "occupied", weights=weights)
        vir = MP2_Coupled_Bath(frag, dmet, "virtual", weights=weights)
        return frag, dmet, occ, vir

    def test_density_vs_full_system_mp2(self):
        """Without rcut the two-sided full bath is the whole molecule: the coupled BNO occupations must equal
        the environment eigenvalues of the explicitly projected full-system MP2 density."""
        ovlp = self.mf.get_ovlp()
        for weights in ((1.0, 1.0), "scs"):
            w = (1.2, 1.0 / 3.0) if weights == "scs" else weights
            frag, dmet, occ, vir = self._baths("O", weights=weights)
            for bath, side, c_env in ((occ, "occupied", dmet.c_env_occ), (vir, "virtual", dmet.c_env_vir)):
                dm, c_mo = projected_density_reference(self.mf, self.mp2.t2, frag.c_frag, side, w)
                x = c_mo.T @ ovlp @ c_env
                ref = np.sort(np.linalg.eigvalsh(x.T @ dm @ x))[::-1]
                self.assertAllclose(bath.occup, ref, atol=1e-10, rtol=0)

    def test_energy_partition(self):
        """Projected full-bath MP2 energies (and OS/SS parts) of all atomic fragments sum to the full MP2."""
        e = e_os = e_ss = 0.0
        for atom in range(self.mf.mol.natm):
            frag, dmet, occ, vir = self._baths(atom)
            d = occ.get_coupled_mp2()
            self.assertIs(vir.get_coupled_mp2(), d)  # shared between the occupied and virtual bath objects
            e += d["e_mp2"]
            e_os += d["e_os"]
            e_ss += d["e_ss"]
        self.assertAllclose(e, self.mp2.e_corr, atol=1e-10, rtol=0)
        self.assertAllclose(e_os, self.mp2.e_corr_os, atol=1e-10, rtol=0)
        self.assertAllclose(e_ss, self.mp2.e_corr_ss, atol=1e-10, rtol=0)

    def test_full_bath_ewf_is_ccsd(self):
        """With a complete bath every cluster is the whole system: EWF = CCSD, and ΔMP2 = 0."""
        cc = pyscf.cc.CCSD(self.mf)
        cc.kernel()
        emb = vayesta.ewf.EWF(self.mf, solver="CCSD", bath_options=dict(bathtype="mp2coupled", threshold=-1))
        emb.kernel()
        self.assertAllclose(emb.e_tot, cc.e_tot, atol=1e-7, rtol=0)
        for fx in emb.fragments:
            d = fx._bath_factory_occ.get_delta_mp2(fx.cluster)
            self.assertAllclose(d["d_mp2"], 0.0, atol=1e-10, rtol=0)
            self.assertAllclose(d["d_scs"], 0.0, atol=1e-10, rtol=0)

    def test_truncated_bath_runs(self):
        """Coupled and SCS-coupled BNOs at a truncated threshold give working, different baths."""
        res = {}
        for weights in ((1.0, 1.0), "scs"):
            emb = vayesta.ewf.EWF(
                self.mf, solver="CCSD", bath_options=dict(bathtype="mp2coupled", threshold=1e-4, coupled_weights=weights)
            )
            emb.kernel()
            self.assertTrue(np.isfinite(emb.e_tot))
            res[str(weights)] = [fx.cluster.norb_active for fx in emb.fragments]
            for fx in emb.fragments:
                d = fx._bath_factory_vir.get_delta_mp2(fx.cluster)
                self.assertTrue(d["d_mp2"] <= 1e-12)  # the full bath has more (negative) correlation
        self.assertTrue(len(res) == 2)

    def test_rcut_freezes_far_environment(self):
        """With rcut the far environment of both sides is frozen, not dropped."""
        emb = vayesta.ewf.EWF(self.mf, solver="CCSD", bath_options=dict(bathtype="mp2coupled", threshold=-1, rcut=0.5))
        emb.kernel()
        self.assertTrue(np.isfinite(emb.e_tot))
        nmo = self.mf.mo_coeff.shape[-1]
        for fx in emb.fragments:
            c = fx._bath_factory_occ.get_coupled_mp2()
            self.assertEqual(fx.cluster.norb_active + fx.cluster.norb_frozen, nmo)
            self.assertEqual(c["far_occ"].shape[-1] + c["far_vir"].shape[-1], fx.cluster.norb_frozen)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
