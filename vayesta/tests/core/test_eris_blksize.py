import unittest

import numpy as np

import vayesta.core.eris as eris
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class CderiBlockSizeTests(TestCase):
    """The DF AO->MO loops must give identical integrals for every block of auxiliary functions."""

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_ccpvdz_df.rhf()
        nocc = np.count_nonzero(cls.mf.mo_occ > 0)
        cls.c_occ = cls.mf.mo_coeff[:, :nocc]
        cls.c_vir = cls.mf.mo_coeff[:, nocc:]
        cls.naux = cls.mf.with_df.get_naoaux()

    def tearDown(self):
        eris.CDERI_BLOCK_BYTES = 1e9

    def test_blksize_rule(self):
        # Trp-cage size: 920 AOs -> about 1 GB / (920^2 * 8 B) = 147 aux functions per block, not the whole aux basis
        self.assertEqual(eris.cderi_blksize(8 * 920 * 920), 147)
        self.assertEqual(eris.cderi_blksize(8 * 920 * 920, blksize=10), 10)
        self.assertEqual(eris.cderi_blksize(1e12), 1)
        eris.CDERI_BLOCK_BYTES = 8 * 920 * 920 * 5
        self.assertEqual(eris.cderi_blksize(8 * 920 * 920), 5)

    def test_cderi_df_independent_of_blocks(self):
        ref, ref_neg = eris.get_cderi_df(self.mf, (self.c_occ, self.c_vir), blksize=self.naux)
        self.assertIsNone(ref_neg)
        self.assertEqual(ref.shape, (self.naux, self.c_occ.shape[-1], self.c_vir.shape[-1]))
        for blksize in (1, 7, self.naux - 1, None):
            cderi, _ = eris.get_cderi_df(self.mf, (self.c_occ, self.c_vir), blksize=blksize)
            self.assertAllclose(cderi, ref, atol=1e-13, rtol=0)
        # the module-level budget alone must also split the loop into many blocks with the same result
        nao = self.mf.mol.nao
        eris.CDERI_BLOCK_BYTES = 3 * 8 * (nao * nao + nao * self.c_occ.shape[-1] + self.c_occ.shape[-1] * self.c_vir.shape[-1])
        cderi, _ = eris.get_cderi_df(self.mf, (self.c_occ, self.c_vir))
        self.assertAllclose(cderi, ref, atol=1e-13, rtol=0)

    def test_cderi_reproduces_eris(self):
        cderi, _ = eris.get_cderi_df(self.mf, (self.c_occ, self.c_vir), blksize=3)
        ovov = np.einsum("Lia,Ljb->iajb", cderi, cderi)
        ref = self.mf.with_df.ao2mo((self.c_occ, self.c_vir, self.c_occ, self.c_vir), compact=False)
        self.assertAllclose(ovov.reshape(ref.shape), ref, atol=1e-12, rtol=0)

    def test_cderi_exspace_independent_of_blocks(self):
        ex = np.einsum("ai,bj->ijab", self.c_occ, self.c_vir).reshape(-1, *self.c_occ.shape[:1] * 2)
        ref, _ = eris.get_cderi_df_exspace(self.mf, ex, blksize=self.naux)
        for blksize in (1, 11, None):
            cderi, _ = eris.get_cderi_df_exspace(self.mf, ex, blksize=blksize)
            self.assertAllclose(cderi, ref, atol=1e-13, rtol=0)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
