"""Lean embedding: DMET bath and rcut near/far split without the full environment.

Opt-in with ``bath_options["lean"] = True`` (default False; without it nothing in this module is used).
RHF, molecules, and an MP2-type bath (``bathtype="mp2"`` or ``"mp2coupled"``) with ``rcut`` set.

The standard path builds the explicit environment orbitals of every fragment (n(AO) x n(env), ~5-8 GB each for
31,795 AOs) and diagonalizes n(env) x n(env) matrices (DMET bath: environment 1-DM; rcut split: <(r-R)^2> over the
whole occupied / virtual environment). The lean path never forms them:

DMET bath
    SVD of the occupied MOs projected on the fragment, C_occ^T S C_frag = U s V^T. Each occupied orbital
    phi_k = C_occ U_k = s_k f_k + sqrt(1 - s_k^2) b_k (f_k = C_frag V_k) has an environment partner
        b_k = (phi_k - s_k f_k) / sqrt(1 - s_k^2)
    which is an eigenvector of the environment 1-DM with eigenvalue 1 - s_k^2; all other eigenvalues are exactly
    0 or 1. The bath is the b_k with dmet_threshold <= 1 - s_k^2 <= 1 - dmet_threshold, i.e. the same orbitals and
    the same threshold test as DMET_Bath_RHF.make_dmet_bath. Cost n(AO) n(occ) n(frag).

rcut near/far split
    The environment of one side (occupied or virtual) is the MO space of that side minus the DMET cluster orbitals.
    Candidate directions = the environment orbitals with weight on the AOs of the atoms within rcut + lean_buffer
    of the fragment (SVD of Q C^T S[:, local AOs], Q removes the cluster; singular values > lean_sv_tol).
    <(r-R)^2> is then diagonalized inside the candidate space only. Its matrix is built exactly, from the AO
    integrals in row blocks (~lean_r2_block AOs x all AOs) contracted with the candidates block by block, so the
    n(AO)^2 matrix is never stored. (The candidates are combinations of delocalized MOs with tails on every atom,
    so restricting the integrals to "local" AOs is not safe.) near = sqrt(eig) <= rcut, as in R2_Bath_RHF.
    Validated against R2_Bath_RHF on trypsin (STO-3G): buffer 3 A with sv_tol 1e-6 reproduces the near spaces and
    clusters exactly (COUPLED_PIPELINE/validation, job 6574324). Other basis sets need their own check.

Far environment
    Never formed. Clusters carry only the explicit near-space orbitals in their frozen space; the far counts are
    stored on the cluster (lean_nocc_far / lean_nvir_far) and make_frozen_rdm1() uses the mean-field density.
"""

import numpy as np
from vayesta.core.util import fix_orbital_sign, hstack, time_string, timer
from vayesta.core.bath.dmet import DMET_Bath_RHF
from vayesta.core.bath.r2bath import BOHR, _to_bohr


def lean_enabled(base):
    """True if the embedding object `base` runs the lean path."""
    try:
        return bool(base.opts.bath_options.get("lean", False))
    except AttributeError:
        return False


def check_lean_supported(fragment):
    """Raise for settings the lean path does not implement (it never falls back silently)."""
    base = fragment.base
    opts = base.opts.bath_options
    if not base.is_rhf:
        raise NotImplementedError("bath_options['lean']: RHF only")
    if getattr(base.mol, "dimension", 0) != 0:
        raise NotImplementedError("bath_options['lean']: molecules only (no PBC)")
    if opts.get("rcut", None) is None:
        raise ValueError("bath_options['lean'] requires bath_options['rcut'] (the near/far split)")
    for occtype in ("occupied", "virtual"):
        btype = fragment._get_bath_option("bathtype", occtype)
        if btype not in ("mp2", "mp2coupled"):
            raise NotImplementedError("bath_options['lean']: bathtype '%s' not supported (mp2, mp2coupled)" % btype)
    if opts.get("ligand_atoms", None) is not None:
        raise NotImplementedError("bath_options['lean']: ligand_atoms not supported yet")
    if opts.get("addbuffer", False):
        raise NotImplementedError("bath_options['lean']: addbuffer not supported")
    if fragment.sym_parent is not None:
        raise NotImplementedError("bath_options['lean']: symmetry-derived fragments not supported")


class LeanDMET_Bath_RHF(DMET_Bath_RHF):
    """DMET bath from the SVD of C_occ^T S C_frag; no environment orbitals (c_env_occ / c_env_vir stay None)."""

    def __init__(self, fragment, dmet_threshold=1e-8):
        super().__init__(fragment, dmet_threshold=dmet_threshold)
        self.n_env_occ = None
        self.n_env_vir = None

    def kernel(self):
        self.log.info("Making DMET Bath (lean)")
        self.log.info("-----------------------")
        self.log.changeIndentLevel(1)
        t0 = timer()
        c_dmet, n_dmet = self.make_lean_dmet_bath()
        self.base._check_orthonormal(self.c_frag, c_dmet, mo_name="cluster MO")
        tol = self.base.opts.bath_options["occupation_tolerance"]
        c_cluster_occ, c_cluster_vir = self.diagonalize_cluster_dm(self.c_frag, c_dmet, tol=tol)
        c_cluster_occ = self.fragment.canonicalize_mo(c_cluster_occ)[0]
        c_cluster_vir = self.fragment.canonicalize_mo(c_cluster_vir)[0]
        nocc = self.base.nocc
        nmo = self.base.mo_coeff.shape[-1]
        self.n_env_occ = nocc - c_cluster_occ.shape[-1]
        self.n_env_vir = nmo - nocc - c_cluster_vir.shape[-1]
        self.log.info("Cluster orbitals:  n(occ)= %3d  n(vir)= %3d", c_cluster_occ.shape[-1], c_cluster_vir.shape[-1])
        self.log.info("Environment (implicit):  n(occ-Env)= %d  n(vir-Env)= %d", self.n_env_occ, self.n_env_vir)
        self.log.timing("Time for DMET bath (lean):  %s", time_string(timer() - t0))
        self.log.changeIndentLevel(-1)
        self.c_dmet = c_dmet
        self.n_dmet = n_dmet
        self.c_env_occ = None
        self.c_env_vir = None
        self.c_cluster_occ = c_cluster_occ
        self.c_cluster_vir = c_cluster_vir

    def make_lean_dmet_bath(self):
        c_frag = self.c_frag
        c_occ = self.base.mo_coeff_occ
        ovlp = self.base.get_ovlp()
        r = np.dot(c_occ.T, np.dot(ovlp, c_frag))  # n(occ) x n(frag)
        u, sv, vh = np.linalg.svd(r, full_matrices=False)
        sv = np.clip(sv, 0.0, 1.0)
        lam = 1.0 - sv**2  # environment 1-DM eigenvalue (in [0,1]) of each partner orbital
        tol = self.dmet_threshold
        mask = np.logical_and(lam >= tol, lam <= 1 - tol)
        # Same order as make_dmet_bath: descending environment occupation
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(-lam[idx], kind="stable")]
        phi = np.dot(c_occ, u[:, idx])
        f = np.dot(c_frag, vh.T[:, idx])
        c_dmet = (phi - f * sv[idx]) / np.sqrt(lam[idx])
        c_dmet = fix_orbital_sign(c_dmet)[0]
        n_dmet = lam[idx]
        # Log as make_dmet_bath (entanglement of all partner orbitals; the others have eigenvalues 0 or 1)
        nout = len(lam) - len(idx)
        self.log.info("DMET bath:  n(Bath)= %4d  (fragment partners outside the bath: %d)", len(idx), nout)
        entropy = float(np.sum(lam * (1 - lam)))
        entropy_bath = float(np.sum(n_dmet * (1 - n_dmet)))
        self.log.info(
            "Entanglement entropy: total= %.3e  bath= %.3e (%.2f %%)",
            entropy,
            entropy_bath,
            100 * entropy_bath / entropy if entropy > 0 else 100.0,
        )
        border = np.logical_and(~mask, np.minimum(lam, 1 - lam) > 0.01 * tol)
        if np.any(border):
            self.log.info(
                "DMET bath: %d partner orbital(s) just outside dmet_threshold: occupations %r",
                int(np.count_nonzero(border)),
                lam[border],
            )
        return c_dmet, n_dmet

    def diagonalize_cluster_dm(self, *mo_coeff, tol=1e-4):
        """As Fragment.diagonalize_cluster_dm, but the cluster DM comes from the occupied MOs (no n(AO)^2 DM)."""
        c_cluster = hstack(*mo_coeff)
        x = np.dot(self.base.mo_coeff_occ.T, np.dot(self.base.get_ovlp(), c_cluster))
        dm = 2 * np.dot(x.T, x)
        e, r = np.linalg.eigh(dm)
        if tol and not np.allclose(np.fmin(abs(e), abs(e - 2)), 0, atol=2 * tol, rtol=0):
            self.log.warn("Eigenvalues of cluster-DM not all close to 0 or 2:\n%s" % e)
        e, r = e[::-1], r[:, ::-1]
        c_cluster = np.dot(c_cluster, r)
        c_cluster = fix_orbital_sign(c_cluster)[0]
        nocc = np.count_nonzero(e >= 1)
        return np.hsplit(c_cluster, [nocc])

    def get_occupied_bath(self, *args, **kwargs):
        raise NotImplementedError("lean DMET bath has no explicit environment")

    def get_virtual_bath(self, *args, **kwargs):
        raise NotImplementedError("lean DMET bath has no explicit environment")

    def get_environment(self):
        raise NotImplementedError("lean DMET bath has no explicit environment")


def r2_in_space(mol, center, c, block=1000):
    """c^T <(r - center)^2> c, exact, from AO row blocks of about `block` AOs (never the n(AO)^2 matrix)."""
    ao_loc = mol.ao_loc_nr()
    out = np.zeros((c.shape[-1], c.shape[-1]))
    with mol.with_common_origin(center):
        i0 = 0
        while i0 < mol.nbas:
            i1 = i0 + 1
            while i1 < mol.nbas and ao_loc[i1 + 1] - ao_loc[i0] <= block:
                i1 += 1
            r2_blk = mol.intor("int1e_r2", shls_slice=(i0, i1, 0, mol.nbas))  # (block AOs) x n(AO)
            out += np.dot(c[ao_loc[i0] : ao_loc[i1]].T, np.dot(r2_blk, c))
            i0 = i1
    return (out + out.T) / 2


def lean_env_split(fragment, dmet_bath, occtype):
    """Near (rcut) environment orbitals of one side, without the explicit environment.

    Returns c_near (n(AO) x n(near)) in ascending <(r-R)^2>, as R2_Bath_RHF.get_bath; the far part is implicit.
    Cached on the fragment per DMET bath, side and options, so the one-sided and coupled baths share it.
    """
    base = fragment.base
    opts = base.opts.bath_options
    unit = opts.get("unit", "Ang")
    rcut = _to_bohr(opts["rcut"], unit)
    buffer = _to_bohr(opts.get("lean_buffer", 3.0), unit)
    sv_tol = float(opts.get("lean_sv_tol", 1e-6))
    r2_block = int(opts.get("lean_r2_block", 1000))
    key = (id(dmet_bath), occtype, rcut, buffer, sv_tol)
    cache = getattr(fragment, "_lean_split_cache", None)
    if cache is None or cache.get("dmet_bath") is not dmet_bath:
        cache = fragment._lean_split_cache = dict(dmet_bath=dmet_bath)
    if key in cache:
        return cache[key]

    log = fragment.log
    t0 = timer()
    mol = fragment.mol
    ovlp = base.get_ovlp()
    # R2_Bath_RHF centres <(r-R)^2> on the first fragment atom; the local region covers all fragment atoms
    center = mol.atom_coord(fragment.atoms[0])
    coords = mol.atom_coords()
    frag_coords = coords[list(fragment.atoms)]
    dist = np.min(np.linalg.norm(coords[:, None, :] - frag_coords[None, :, :], axis=2), axis=1)
    aoslice = mol.aoslice_by_atom()
    loc_atoms = np.flatnonzero(dist <= rcut + buffer)
    loc = np.concatenate([np.arange(aoslice[a, 2], aoslice[a, 3]) for a in loc_atoms])

    if occtype == "occupied":
        c_mo, c_cl = base.mo_coeff_occ, dmet_bath.c_cluster_occ
    elif occtype == "virtual":
        c_mo, c_cl = base.mo_coeff_vir, dmet_bath.c_cluster_vir
    else:
        raise ValueError("Invalid occtype: %s" % occtype)
    # Environment of this side in MO coordinates: Q = 1 - r r^T removes the cluster orbitals
    m = np.dot(c_mo.T, ovlp[:, loc])  # n(MO) x n(local AO)
    rc = np.dot(c_mo.T, np.dot(ovlp, c_cl))  # n(MO) x n(cluster)
    m -= np.dot(rc, np.dot(rc.T, m))
    u, sv, _ = np.linalg.svd(m, full_matrices=False)
    keep = sv > sv_tol
    cand = np.dot(c_mo, u[:, keep])  # orthonormal environment directions
    t_cand = timer() - t0

    # <(r-R)^2> inside the candidate space (exact, blocked)
    t1 = timer()
    r2 = r2_in_space(mol, center, cand, block=r2_block)
    eig, rot = np.linalg.eigh(r2)
    eig = np.sqrt(np.clip(eig, 0, None))
    near = eig <= rcut
    c_near = np.dot(cand, rot[:, near])
    t_r2 = timer() - t1

    nenv = c_mo.shape[-1] - c_cl.shape[-1]
    log.info(
        "Lean rcut split (%s): rcut= %.2f A  buffer= %.2f A  local AOs= %d  candidates= %d (sv > %.0e)"
        "  near= %d  far (implicit)= %d",
        occtype, rcut * BOHR, buffer * BOHR, len(loc), cand.shape[-1], sv_tol,
        int(near.sum()), nenv - int(near.sum()),
    )
    # Orbitals within 0.1 A of rcut may flip near <-> far with a small change of the candidate space
    edge = np.flatnonzero(abs(eig - rcut) * BOHR < 0.1)
    if len(edge):
        log.info("Lean rcut split (%s): %d orbital(s) within 0.1 A of rcut: %r A", occtype, len(edge), eig[edge] * BOHR)
    if keep.all():
        log.warning(
            "Lean rcut split (%s): all %d local directions kept; lean_sv_tol may be too low to resolve the space",
            occtype, len(sv),
        )
    log.timing("Time lean rcut split (%s):  candidates= %s  r2= %s", occtype, time_string(t_cand), time_string(t_r2))
    cache[key] = c_near
    return c_near


def frozen_rdm1_lean(base, cluster):
    """Frozen (explicit + implicit far) 1-DM of a lean cluster: mean-field DM minus the active occupied part."""
    c_occ = base.mo_coeff_occ
    c_act = cluster.c_active_occ
    return 2 * (np.dot(c_occ, c_occ.T) - np.dot(c_act, c_act.T))
