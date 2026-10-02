"""Coupled MP2 bath natural orbitals.

One-sided MP2 BNOs (``MP2_BNO_Bath``) are built separately for each side: the occupied BNOs from an
MP2 calculation whose virtual space is frozen at the DMET cluster, and the virtual BNOs the other way
round. Pairs in which *both* the occupied and the virtual index lie in the environment never enter
either selection.

Coupled BNOs are built from ONE fragment-projected MP2 calculation in which the occupied AND the
virtual environment are active (the environment within ``rcut`` if set, otherwise all of it):

    t2s = (P t2 + (P t2)^T) / 2,   P = R R^T,   R = C_occ^T S C_frag    (projection on the 1st occ. index)
    A_occ = sum_kab t2s_ikab t2s_jkab,   B_occ = sum_kab t2s_ikab t2s_jkba     (and likewise A_vir, B_vir)
    D = (c_os + c_ss) A - c_ss B       weights (1, 1): the MP2 density 2A - B ("coupled")
                                       weights (1.2, 1/3): SCS-weighted density ("SCS-coupled")

The natural orbitals of D in the environment are truncated with the usual BNO threshold. The same
calculation gives the fragment-projected MP2 energy of the full bath and its opposite-/same-spin parts,
so ``get_delta_mp2`` returns the ΔMP2 and ΔSCS-MP2 corrections for a cluster at no extra cost.

The full t2 is never stored: only the fragment-projected amplitudes R^T t2 of size
n(frag) x n(occ) x n(vir)^2, built block-wise over the first occupied index.

Restricted (RHF) only.
"""

import numpy as np

from vayesta.core.util import brange, dot, einsum, fix_orbital_sign, hstack, time_string, timer
from vayesta.core.types import Cluster
from vayesta.core.bath.bno import MP2_BNO_Bath

SCS_WEIGHTS = (1.2, 1.0 / 3.0)


def parse_coupled_weights(weights):
    """(c_os, c_ss) from a tuple or one of the names "mp2" (= (1, 1)) and "scs" (= Grimme's (1.2, 1/3))."""
    if isinstance(weights, str):
        name = weights.lower()
        if name == "mp2":
            return (1.0, 1.0)
        if name == "scs":
            return SCS_WEIGHTS
        raise ValueError("Unknown coupled_weights: %r (use 'mp2', 'scs' or a (c_os, c_ss) tuple)" % weights)
    c_os, c_ss = weights
    return (float(c_os), float(c_ss))


def coupled_densities(tx, rfrag):
    """Direct (A) and exchange (B) parts of the occupied and virtual densities of the symmetrized,
    fragment-projected amplitudes t2s = (P t2 + (P t2)^T)/2, P = R R^T, computed from tx = R^T t2 only.

    Parameters
    ----------
    tx : (n(frag), n(occ), n(vir), n(vir)) array
        Fragment-projected amplitudes R^T t2 (first occupied index projected).
    rfrag : (n(occ), n(frag)) array
        Overlap R = C_occ^T S C_frag.

    Returns
    -------
    a_occ, b_occ : (n(occ), n(occ)) arrays
    a_vir, b_vir : (n(vir), n(vir)) arrays
        The density for weights (c_os, c_ss) is (c_os + c_ss) A - c_ss B.
    """
    r = rfrag
    g = r.T @ r
    # U_ijab = sum_x R_ix tx_xjab ;  V_ijab = U_jiba ;  t2s = (U + V) / 2
    y = einsum("ky,xkab->xyab", r, tx)                      # Y_xy,ab = sum_k R_ky tx_xkab
    gtx = einsum("xy,yjab->xjab", g, tx)
    m = einsum("xkab,ykab->xy", tx, tx)
    ms = einsum("xkab,ykba->xy", tx, tx)
    # Occupied: A = sum t2s_ikab t2s_jkab ; B = sum t2s_ikab t2s_jkba
    uv = r @ einsum("xyab,yjba->xj", y, tx)
    a_occ = 0.25 * (r @ m @ r.T + uv + uv.T + einsum("xiab,xjab->ij", gtx, tx))
    uvs = r @ einsum("xyab,yjab->xj", y, tx)
    b_occ = 0.25 * (r @ ms @ r.T + uvs + uvs.T + einsum("xiba,xjab->ij", gtx, tx))
    # Virtual: A = sum t2s_ijac t2s_ijbc ; B = sum t2s_ijac t2s_ijcb
    a_vir = 0.25 * (einsum("xjac,xjbc->ab", gtx, tx) + einsum("xyac,yxcb->ab", y, y)
                    + einsum("xyca,yxbc->ab", y, y) + einsum("xica,xicb->ab", gtx, tx))
    b_vir = 0.25 * (einsum("xjac,xjcb->ab", gtx, tx) + einsum("xyac,yxbc->ab", y, y)
                    + einsum("xyca,yxcb->ab", y, y) + einsum("xica,xibc->ab", gtx, tx))
    return a_occ, b_occ, a_vir, b_vir


class MP2_Coupled_BNO_Bath(MP2_BNO_Bath):
    """Coupled (two-sided) MP2 bath natural orbitals, see module docstring.

    Parameters (in addition to MP2_BNO_Bath)
    ----------
    weights : tuple or str, optional
        (c_os, c_ss) weights of the density the BNOs are taken from: (1, 1) or "mp2" for coupled BNOs
        (default), (1.2, 1/3) or "scs" for SCS-coupled BNOs.
    """

    def __init__(self, fragment, dmet_bath, occtype, *args, weights=(1.0, 1.0), **kwargs):
        self.weights = parse_coupled_weights(weights)
        super().__init__(fragment, dmet_bath, occtype, *args, **kwargs)

    # --- environment, split by rcut (both sides, independent of self.occtype)
    def _env_split(self, occtype):
        dmet = self.dmet_bath
        c_env = dmet.c_env_occ if occtype == "occupied" else dmet.c_env_vir
        opts = self._get_bath_opts()
        rcut = opts.get("rcut", None)
        if rcut is None:
            return c_env, np.zeros((c_env.shape[0], 0))
        from vayesta.core.bath.r2bath import R2_Bath_RHF

        c_near, c_far = R2_Bath_RHF(self.fragment, dmet, occtype).get_bath(rcut, unit=opts.get("unit", "Ang"))
        own = self.occtype
        self.occtype = occtype  # _add_ligand_env reads self.occtype
        try:
            c_near, c_far = self._add_ligand_env(c_near, c_far)
        finally:
            self.occtype = own
        return c_near, c_far

    # --- the shared two-sided MP2 (one per fragment and DMET bath)
    def _projected_t2(self, actspace, fock, max_memory=1e9):
        """Fragment-projected amplitudes tx = R^T t2, overlap R, and the projected MP2 energy split
        E_MP2 = 2a - b = OS + SS with OS = a, SS = a - b."""
        eris, cderi, cderi_neg = self.get_eris_or_cderi(actspace)
        nocc = actspace.c_active_occ.shape[-1]
        nvir = actspace.c_active_vir.shape[-1]
        rfrag = dot(actspace.c_active_occ.T, self.base.get_ovlp(), self.c_frag)
        mo_energy = self._get_mo_energy(fock, actspace)
        eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
        nfrag = rfrag.shape[-1]
        tx = np.zeros((nfrag, nocc, nvir, nvir))
        gx = np.zeros_like(tx)
        blksize = max(1, int(max_memory / max(nocc * nvir * nvir * 8, 1)))
        for blk in brange(0, nocc, blksize):
            if eris is not None:
                gijab = eris[blk].transpose(0, 2, 1, 3)
            else:
                gijab = einsum("Lia,Ljb->ijab", cderi[:, blk], cderi)
                if cderi_neg is not None:
                    gijab -= einsum("Lia,Ljb->ijab", cderi_neg[:, blk], cderi_neg)
            t2blk = gijab / (eia[blk][:, None, :, None] + eia[None, :, None, :])
            tx += einsum("ix,ijab->xjab", rfrag[blk], t2blk)
            gx += einsum("ix,ijab->xjab", rfrag[blk], gijab)
        a = einsum("xjab,xjab->", tx, gx)
        b = einsum("xjab,xjba->", tx, gx)
        return tx, rfrag, float(a), float(a - b)

    def _two_sided_space(self, c_occ, c_vir, c_frozen_occ, c_frozen_vir, fock):
        """Canonicalized active space (needed for the MP2 denominators) and the rotations back."""
        c_occ_can, r_occ = self.fragment.canonicalize_mo(c_occ, fock=fock)
        c_vir_can, r_vir = self.fragment.canonicalize_mo(c_vir, fock=fock)
        return Cluster.from_coeffs(c_occ_can, c_vir_can, c_frozen_occ, c_frozen_vir), r_occ, r_vir

    def get_coupled_mp2(self):
        """Two-sided full-bath MP2 for this fragment; computed once and shared by the occupied and the
        virtual bath objects of the fragment (cached on the fragment for its current DMET bath and rcut)."""
        key = (self._get_bath_opts().get("rcut", None), self._get_bath_opts().get("unit", "Ang"))
        cache = getattr(self.fragment, "_coupled_bno_cache", None)
        if cache is not None and cache["dmet_bath"] is self.dmet_bath and cache["key"] == key:
            return cache
        t0 = timer()
        dmet = self.dmet_bath
        occ_near, occ_far = self._env_split("occupied")
        vir_near, vir_far = self._env_split("virtual")
        fock = self.base.get_fock_for_bath()
        actspace, r_occ, r_vir = self._two_sided_space(
            hstack(dmet.c_cluster_occ, occ_near), hstack(dmet.c_cluster_vir, vir_near), occ_far, vir_far, fock
        )
        self.log.info(
            "Coupled MP2 bath: two-sided full bath  [occ= %d (DMET %d + env %d)  vir= %d (DMET %d + env %d)]",
            actspace.c_active_occ.shape[-1], dmet.c_cluster_occ.shape[-1], occ_near.shape[-1],
            actspace.c_active_vir.shape[-1], dmet.c_cluster_vir.shape[-1], vir_near.shape[-1],
        )
        tx, rfrag, e_os, e_ss = self._projected_t2(actspace, fock)
        a_occ, b_occ, a_vir, b_vir = coupled_densities(tx, rfrag)
        del tx
        # canonical basis -> (DMET cluster | env-near) basis:  mo_coeff @ rot = mo_canon
        rot = lambda dm, r: dot(r, dm, r.T)  # noqa: E731
        cache = dict(
            dmet_bath=self.dmet_bath, key=key,
            a_occ=rot(a_occ, r_occ), b_occ=rot(b_occ, r_occ), a_vir=rot(a_vir, r_vir), b_vir=rot(b_vir, r_vir),
            env_occ=occ_near, env_vir=vir_near, far_occ=occ_far, far_vir=vir_far,
            ncluster_occ=dmet.c_cluster_occ.shape[-1], ncluster_vir=dmet.c_cluster_vir.shape[-1],
            e_os=e_os, e_ss=e_ss, e_mp2=e_os + e_ss,
        )
        self.fragment._coupled_bno_cache = cache
        self.log.info("Coupled MP2 bath: E(MP2, full bath)= %.10f  [OS= %.10f  SS= %.10f]", e_os + e_ss, e_os, e_ss)
        self.log.timing("Time coupled MP2 bath: %s", time_string(timer() - t0))
        return cache

    def make_bno_coeff(self, eris=None):
        """Natural orbitals of the weighted two-sided density in this side's environment."""
        if eris is not None:
            raise NotImplementedError("MP2_Coupled_BNO_Bath computes its own integrals")
        d = self.get_coupled_mp2()
        c_os, c_ss = self.weights
        side = "occ" if self.occtype == "occupied" else "vir"
        dm = (c_os + c_ss) * d["a_" + side] - c_ss * d["b_" + side]
        ncl = d["ncluster_" + side]
        dm = dm[ncl:, ncl:]
        c_env = d["env_" + side]
        # truncate_bno / c_env read these, as for the rcut-screened one-sided bath
        self._c_env_active = c_env
        self._c_env_frozen = d["far_" + side]
        if dm.shape[0] == 0:
            return c_env, np.zeros(0), d["e_mp2"]
        r_bno, n_bno = self._diagonalize_dm(dm)
        c_bno = fix_orbital_sign(dot(c_env, r_bno))[0]
        self.e_os, self.e_ss = d["e_os"], d["e_ss"]
        return c_bno, n_bno, d["e_mp2"]

    def get_active_space(self, c_active=None):
        raise NotImplementedError("MP2_Coupled_BNO_Bath uses a two-sided active space; see get_coupled_mp2")

    def get_finite_bath_correction(self, c_active, c_frozen):
        raise NotImplementedError(
            "The one-sided finite-bath correction (FBC) is not defined for coupled BNOs; "
            "use get_delta_mp2(cluster) for the two-sided ΔMP2 / ΔSCS-MP2 correction."
        )

    def get_delta_mp2(self, cluster, scs_weights=SCS_WEIGHTS):
        """ΔMP2 and ΔSCS-MP2 corrections of a cluster: the fragment-projected MP2 energy of the
        two-sided full bath minus that of the cluster's active space (both sides relaxed).

        Returns
        -------
        dict with e_mp2/e_os/e_ss of the full bath and of the cluster, d_mp2 = ΔMP2 and
        d_scs = c_os ΔOS + c_ss ΔSS (Grimme's weights by default).
        """
        full = self.get_coupled_mp2()
        fock = self.base.get_fock_for_bath()
        actspace = self._two_sided_space(cluster.c_active_occ, cluster.c_active_vir,
                                         cluster.c_frozen_occ, cluster.c_frozen_vir, fock)[0]
        _, _, e_os, e_ss = self._projected_t2(actspace, fock)
        d_os, d_ss = full["e_os"] - e_os, full["e_ss"] - e_ss
        return dict(e_mp2_full=full["e_mp2"], e_os_full=full["e_os"], e_ss_full=full["e_ss"],
                    e_mp2_cluster=e_os + e_ss, e_os_cluster=e_os, e_ss_cluster=e_ss,
                    d_mp2=d_os + d_ss, d_scs=scs_weights[0] * d_os + scs_weights[1] * d_ss)
