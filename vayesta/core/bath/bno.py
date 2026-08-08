import numbers
import numpy as np
from vayesta.core.util import AbstractMethodError, brange, dot, einsum, fix_orbital_sign, hstack, time_string, timer
from vayesta.core import spinalg
from vayesta.core.types import Cluster
from vayesta.core.bath import helper
from vayesta.core.bath.bath import Bath

# AS: local-aux embedding helper
import pyscf
from pyscf import gto, scf, df as pyscf_df


class BNO_Threshold:
    def __init__(self, type, threshold):
        """
        number:             Fixed number of BNOs
        occupation:         Occupation threshold for BNOs ("eta")
        truncation:         Maximum number of electrons to be ignored
        electron-percent:   Add BNOs until 100-x% of the total number of all electrons is captured
        excited-percent:    Add BNOs until 100-x% of the total number of excited electrons is captured
        """
        if type not in ("number", "occupation", "truncation", "electron-percent", "excited-percent"):
            raise ValueError()
        self.type = type
        self.threshold = threshold

    def __repr__(self):
        return "%s(type=%s, threshold=%g)" % (self.__class__.__name__, self.type, self.threshold)

    def get_number(self, bno_occup, electron_total=None):
        """Get number of BNOs."""
        nbno = len(bno_occup)
        if nbno == 0:
            return 0
        if self.type == "number":
            return self.threshold
        if self.type in ("truncation", "electron-percent", "excited-percent"):
            npos = np.clip(bno_occup, 0.0, None)
            nexcited = np.sum(npos)
            nelec0 = 0
            if self.type == "truncation":
                ntarget = nexcited - self.threshold
            elif self.type == "electron-percent":
                assert electron_total is not None
                ntarget = (1.0 - self.threshold) * electron_total
                nelec0 = electron_total - nexcited
            elif self.type == "excited-percent":
                ntarget = (1.0 - self.threshold) * nexcited
            for bno_number in range(nbno + 1):
                nelec = nelec0 + np.sum(npos[:bno_number])
                if nelec >= ntarget:
                    return bno_number
            raise RuntimeError()
        if self.type == "occupation":
            return np.count_nonzero(bno_occup >= self.threshold)
        raise RuntimeError()


class BNO_Bath(Bath):
    """Bath natural orbital (BNO) bath, requires DMET bath."""

    def __init__(self, fragment, dmet_bath, occtype, *args, c_buffer=None, canonicalize=True, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.dmet_bath = dmet_bath
        if occtype not in ("occupied", "virtual"):
            raise ValueError("Invalid occtype: %s" % occtype)
        self.occtype = occtype
        self.c_buffer = c_buffer
        # Canonicalization can be set separately for occupied and virtual:
        if np.ndim(canonicalize) == 0:
            canonicalize = (canonicalize, canonicalize)
        self.canonicalize = canonicalize
        # Coefficients, occupations, and correlation energy:
        self.coeff, self.occup, self.ecorr = self.kernel()

    @property
    def c_cluster_occ(self):
        """Occupied DMET cluster orbitals."""
        return self.dmet_bath.c_cluster_occ

    @property
    def c_cluster_vir(self):
        """Virtual DMET cluster orbitals."""
        return self.dmet_bath.c_cluster_vir

    def make_bno_coeff(self, *args, **kwargs):
        raise AbstractMethodError()

    @property
    def c_env(self):

        # AS:
        c = getattr(self, "_c_env_active", None)
        if c is not None:
            return c

        # Orginal (kept)
        if self.occtype == "occupied":
            return self.dmet_bath.c_env_occ
        if self.occtype == "virtual":
            return self.dmet_bath.c_env_vir

    @property
    def ncluster(self):
        if self.occtype == "occupied":
            return self.dmet_bath.c_cluster_occ.shape[-1]
        if self.occtype == "virtual":
            return self.dmet_bath.c_cluster_vir.shape[-1]

    # =====================================================================
    # AS: ligand orbitals - always included in the MP2 bath active space
    #     Scope: RHF, molecular (non-periodic). ligand_atoms is 0-based.
    # =====================================================================
    def _get_bath_opts(self):
        try:
            return self.base.opts.bath_options
        except Exception:
            return {}

    def _get_ligand_atoms(self):
        """Explicit 0-based ligand atom indices from bath_options['ligand_atoms']."""
        lig = self._get_bath_opts().get("ligand_atoms", None)
        if lig is None:
            return []
        if self.spin_unrestricted:
            raise NotImplementedError(
                "ligand_atoms is not supported for UHF (RHF only). "
                "Remove ligand_atoms from bath_options or use an RHF reference."
            )
        def _parse(entry):
            if isinstance(entry, str):
                parts = entry.replace(" ", "").split("-")
                if len(parts) == 2:
                    return list(range(int(parts[0]), int(parts[1]) + 1))   # inclusive
                return [int(parts[0])]
            try:
                return list(entry)          # range, list, tuple, ndarray
            except TypeError:
                return [int(entry)]         # bare int

        if isinstance(lig, (str, range, int)):
            lig = [lig]
        atoms = sorted({int(a) for e in lig for a in _parse(e)})
        for a in atoms:
            if not (0 <= a < self.mol.natm):
                raise ValueError(
                    "ligand_atoms: index %d out of range (natm= %d; indices are 0-based)"
                    % (a, self.mol.natm)
                )
        return atoms

    def _get_ligand_aos(self):
        atoms = self._get_ligand_atoms()
        if not atoms:
            return np.zeros(0, dtype=int)
        aoslice = self.mol.aoslice_by_atom()
        return np.hstack([np.arange(aoslice[a][2], aoslice[a][3]) for a in atoms])

    def _ligand_bmat(self, c):
        """b with <psi|P_L|psi> = b b^T, shape (norb, n_lig_ao).
        P_L = |X_L>(S_LL)^-1<X_L|. Never forms a norb x norb matrix."""
        from scipy.linalg import cholesky, solve_triangular
        aos = self._get_ligand_aos()
        if len(aos) == 0 or c.shape[-1] == 0:
            return np.zeros((c.shape[-1], 0))
        ovlp = self.fragment.base.get_ovlp()
        a = np.dot(c.T, ovlp[:, aos])                    # (norb, n_lig_ao)
        s_ll = ovlp[np.ix_(aos, aos)]
        try:
            lmat = cholesky(s_ll, lower=True)
            return solve_triangular(lmat, a.T, lower=True).T
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(s_ll)
            keep = w > 1e-10
            return np.dot(a, v[:, keep] / np.sqrt(w[keep]))

    def _ligand_population(self, c):
        """Ligand population carried by an orthonormal set c."""
        if c.shape[-1] == 0 or len(self._get_ligand_aos()) == 0:
            return 0.0
        return float(np.sum(self._ligand_bmat(c) ** 2))

    def _split_ligand(self, c_space, tol):
        """Split orthonormal c_space into (ligand-character, remainder).
        Selection is O(norb * n_lig_ao^2), not O(norb^3)."""
        from scipy.linalg import qr
        nao, norb = c_space.shape
        if len(self._get_ligand_aos()) == 0 or norb == 0:
            return np.zeros((nao, 0)), c_space, np.zeros(0), np.zeros(0, dtype=bool)
        b = self._ligand_bmat(c_space)                   # (norb, n_lig_ao)
        u, s, _ = np.linalg.svd(b, full_matrices=False)  # thin
        w = s ** 2                                       # descending, in [0, 1]
        mask = w >= tol
        k = int(mask.sum())
        if k == 0:
            return np.zeros((nao, 0)), c_space, w, mask
        q, _ = qr(u[:, mask], mode="full")
        return np.dot(c_space, q[:, :k]), np.dot(c_space, q[:, k:]), w, mask

    def _add_ligand_env(self, c_near, c_far):
        """Move remaining ligand character from the rcut-far env into the active
        (near) env, so that MP2 correlates it. BNO truncation then filters the
        enlarged space as usual."""
        atoms = self._get_ligand_atoms()
        if not atoms or c_far.shape[-1] == 0:
            return c_near, c_far
        tol = float(self._get_bath_opts().get("ligand_tol", 0.1))
        naos = len(self._get_ligand_aos())

        c_clu = self.c_cluster_occ if self.occtype == "occupied" else self.c_cluster_vir
        n_clu = self._ligand_population(c_clu)
        n_near = self._ligand_population(c_near)
        n_far = self._ligand_population(c_far)

        c_lig, c_far_new, w, mask = self._split_ligand(c_far, tol)
        n_add = self._ligand_population(c_lig)
        n_left = self._ligand_population(c_far_new)

        self.log.info("Ligand env (%s): atoms= %r (%d AOs), tol= %.3g",
                      self.occtype, atoms, naos, tol)
        self.log.info("  sector ligand population = %.4f "
                      "[DMET cluster %.4f | rcut-near %.4f | rcut-far %.4f]",
                      n_clu + n_near + n_far, n_clu, n_near, n_far)
        self.log.info("  moved far -> active: %d orbitals, population %.4f",
                      c_lig.shape[-1], n_add)
        self.log.info("  left frozen:        population %.4f", n_left)
        if len(w):
            self.log.info("  far ligand weights: kept min= %.4f  discarded max= %.4f",
                          (w[mask][-1] if np.any(mask) else 0.0),
                          (w[~mask][0] if np.any(~mask) else 0.0))
        if n_left > 0.05:
            self.log.warning("  %.4f ligand population still frozen - lower ligand_tol", n_left)
        return np.hstack((c_near, c_lig)), c_far_new

    def kernel(self):
        c_env = self.c_env
        if self.spin_restricted and (c_env.shape[-1] == 0):
            return c_env, np.zeros(0), None
        if self.spin_unrestricted and (c_env[0].shape[-1] + c_env[1].shape[-1] == 0):
            return c_env, tuple(2 * [np.zeros(0)]), None
        self.log.info("Making %s BNOs", self.occtype.capitalize())
        self.log.info("-------%s-----", len(self.occtype) * "-")
        self.log.changeIndentLevel(1)
        coeff, occup, ecorr = self.make_bno_coeff()
        self.log_histogram(occup)
        self.log.changeIndentLevel(-1)
        self.coeff = coeff
        self.occup = occup
        return coeff, occup, ecorr

    def log_histogram(self, n_bno):
        if len(n_bno) == 0:
            return
        self.log.info("%s BNO histogram:", self.occtype.capitalize())
        bins = np.hstack([-np.inf, np.logspace(-3, -10, 8)[::-1], np.inf])
        labels = "    " + "".join("{:{w}}".format("E-%d" % d, w=5) for d in range(3, 11))
        self.log.info(helper.make_histogram(n_bno, bins=bins, labels=labels))

    def get_bath(self, bno_threshold=None, **kwargs):
        return self.truncate_bno(self.coeff, self.occup, bno_threshold=bno_threshold, **kwargs)

    @staticmethod
    def _has_frozen(c_frozen):
        return c_frozen.shape[-1] > 0

    def get_finite_bath_correction(self, c_active, c_frozen):
        if not self._has_frozen(c_frozen):
            return 0
        e1 = self.ecorr
        actspace = self.get_active_space(c_active=c_active)
        # --- Canonicalization
        fock = self.base.get_fock_for_bath()
        if self.canonicalize[0]:
            self.log.debugv("Canonicalizing occupied orbitals")
            c_active_occ = self.fragment.canonicalize_mo(actspace.c_active_occ, fock=fock)[0]
        else:
            c_active_occ = actspace.c_active_occ
        if self.canonicalize[1]:
            self.log.debugv("Canonicalizing virtual orbitals")
            c_active_vir = self.fragment.canonicalize_mo(actspace.c_active_vir, fock=fock)[0]
        else:
            c_active_vir = actspace.c_active_vir
        actspace = Cluster.from_coeffs(c_active_occ, c_active_vir, actspace.c_frozen_occ, actspace.c_frozen_vir)
        e0 = self._make_t2(actspace, fock, energy_only=True)[1]
        e_fbc = e1 - e0
        return e_fbc

    def truncate_bno(self, coeff, occup, bno_threshold=None, verbose=True):
        """Split natural orbitals (NO) into bath and rest."""

        header = "%s BNOs:" % self.occtype

        if isinstance(bno_threshold, numbers.Number):
            bno_threshold = BNO_Threshold("occupation", bno_threshold)
        nelec_cluster = self.dmet_bath.get_cluster_electrons()
        bno_number = bno_threshold.get_number(occup, electron_total=nelec_cluster)

        # Logging
        if verbose:
            if header:
                self.log.info(header.capitalize())
            fmt = "  %4s: N= %4d  max= % 9.3g  min= % 9.3g  sum= % 9.3g ( %7.3f %%)"

            def log_space(name, n_part):
                if len(n_part) == 0:
                    self.log.info(fmt[: fmt.index("max")].rstrip(), name, 0)
                    return
                with np.errstate(invalid="ignore"):  # supress 0/0 warning
                    self.log.info(
                        fmt,
                        name,
                        len(n_part),
                        max(n_part),
                        min(n_part),
                        np.sum(n_part),
                        100 * np.sum(n_part) / np.sum(occup),
                    )

            log_space("Bath", occup[:bno_number])
            log_space("Rest", occup[bno_number:])

        c_bath, c_rest = np.hsplit(coeff, [bno_number])
        # AS: restore the rcut-frozen far env into the cluster frozen space.
        # Without this the far orbitals are dropped from the Cluster entirely
        # and make_frozen_rdm1() is missing their density.
        c_far = getattr(self, "_c_env_frozen", None)
        if c_far is not None and c_far.shape[-1] > 0:
            c_rest = np.hstack((c_rest, c_far))
        return c_bath, c_rest

    def get_active_space(self, c_active=None):
        dmet_bath = self.dmet_bath
        nao = self.mol.nao
        empty = np.zeros((nao, 0)) if self.spin_restricted else np.zeros((2, nao, 0))

        # ============================
        # AS: rcut-based env screening (R2 bath)
        # ============================
        # Goal: if rcut is set, split the *environment* into
        # "near" and "far" subspaces via R2_Bath_RHF, then:
        #   - append "near" to the active sector (occ or vir, depending on occtype)
        #   - add "far" explicitly to the frozen sector (occ or vir, depending on occtype)
        #
        # This preserves a clean active/frozen partition and avoids meaningless
        # distance screening on canonical env orbitals by rotating into an <r^2> eigenbasis.
        try:
            bath_opts = self.base.opts.bath_options
        except Exception:
            bath_opts = {}

        rcut = bath_opts.get("rcut", None)
        rcut_unit = bath_opts.get("unit", "Ang")
        env_complement_thresh = getattr(self.opts, "env_complement_thresh", 1e-8) if hasattr(self, "opts") else 1e-8

        # AS: local helper to obtain (env_near, env_far) for a requested occtype
        def _split_env_by_r2(occtype):
            # Default (no screening): keep original behavior
            if rcut is None:
                if occtype == "occupied":
                    return dmet_bath.c_env_occ, empty
                if occtype == "virtual":
                    return dmet_bath.c_env_vir, empty
                raise ValueError(f"Invalid occtype for R2 split: {occtype}")

            # AS: use Vayesta R2 bath machinery
            from vayesta.core.bath.r2bath import R2_Bath_RHF

            # NOTE: R2_Bath_RHF currently requires len(fragment.atoms) == 1 and RHF;
            # this matches r2bath.py as provided.
            r2bath = R2_Bath_RHF(self.fragment, dmet_bath, occtype)
            c_near, c_far = r2bath.get_bath(rcut, unit=rcut_unit)
            return c_near, c_far

        if self.occtype == "occupied":
            if c_active is None:
            # ============================
            # ORIGINAL: c_active_occ = spinalg.hstack_matrices(dmet_bath.c_cluster_occ, self.c_env) 
            # ============================
            # AS: replace self.c_env with rcut-screened env-occ "near"; freeze the "far" env-occ
                c_env_occ_near, c_env_occ_far = _split_env_by_r2("occupied")  # AS
                c_active_occ = spinalg.hstack_matrices(dmet_bath.c_cluster_occ, c_env_occ_near)  # AS
                if rcut is not None:
                    self._c_env_active = c_env_occ_near
                    self._c_env_frozen = c_env_occ_far
            else:
                c_active_occ = c_active

            # ============================
            # ORIGINAL: c_frozen_occ = empty
            # AS: freeze remaining env-occ if rcut is used; otherwise identical (env_far is empty)
            # ============================
            if c_active is None:
                c_frozen_occ = c_env_occ_far  # AS
            else:
                # If user explicitly supplied c_active, keep original default frozen occ
                c_frozen_occ = empty  # AS (conservative)
            
            # ORIGINAL: buffer not implemented in occupied mode
            if self.c_buffer is not None:
                raise NotImplementedError

            # ORIGINAL (kept)
            c_active_vir = dmet_bath.c_cluster_vir
            c_frozen_vir = dmet_bath.c_env_vir

        elif self.occtype == "virtual":
            if c_active is None:
            # ============================
            # ORIGINAL: c_active_vir = spinalg.hstack_matrices(dmet_bath.c_cluster_vir, self.c_env) 
            # ============================
            # AS: replace self.c_env with rcut-screened env-vir "near"; freeze the "far" env-vir
                c_env_vir_near, c_env_vir_far = _split_env_by_r2("virtual")  # AS
                c_active_vir = spinalg.hstack_matrices(dmet_bath.c_cluster_vir, c_env_vir_near)  # AS
                if rcut is not None:
                    self._c_env_active = c_env_vir_near
                    self._c_env_frozen = c_env_vir_far
            else:
                c_active_vir = c_active

            # ============================
            # ORIGINAL: c_frozen_vir = empty
            # AS: freeze remaining env-vir if rcut is used; otherwise identical (env_far is empty)
            # ============================
            if c_active is None:
                c_frozen_vir = c_env_vir_far  # AS
            else:
                # If user explicitly supplied c_active, keep original default frozen vir
                c_frozen_vir = empty  # AS (conservative)

            # ORIGINAL occupied-handling logic (kept)
            if self.c_buffer is None:
                c_active_occ = dmet_bath.c_cluster_occ
                c_frozen_occ = dmet_bath.c_env_occ
            else:
                c_active_occ = spinalg.hstack_matrices(dmet_bath.c_cluster_occ, self.c_buffer)
                ovlp = self.fragment.base.get_ovlp()
                r = dot(self.c_buffer.T, ovlp, dmet_bath.c_env_occ)
                dm_frozen = np.eye(dmet_bath.c_env_occ.shape[-1]) - np.dot(r.T, r)
                e, r = np.linalg.eigh(dm_frozen)

                # ORIGINAL: c_frozen_occ = np.dot(dmet_bath.c_env_occ, r[:, e > 0.5])
                # AS: use a smaller, configurable threshold for the orthogonal complement
                c_frozen_occ = np.dot(dmet_bath.c_env_occ, r[:, e > env_complement_thresh])

        actspace = Cluster.from_coeffs(c_active_occ, c_active_vir, c_frozen_occ, c_frozen_vir)
        return actspace

    def _rotate_dm(self, dm, rot):
        return dot(rot, dm, rot.T)

    def _dm_take_env(self, dm):
        ncluster = self.ncluster
        self.log.debugv("n(cluster)= %d", ncluster)
        self.log.debugv("tr(D)= %g", np.trace(dm))
        dm = dm[ncluster:, ncluster:]
        self.log.debugv("tr(D[env,env])= %g", np.trace(dm))
        return dm

    def _diagonalize_dm(self, dm):
        n_bno, r_bno = np.linalg.eigh(dm)
        sort = np.s_[::-1]
        n_bno = n_bno[sort]
        r_bno = r_bno[:, sort]
        return r_bno, n_bno


class BNO_Bath_UHF(BNO_Bath):
    def _rotate_dm(self, dm, rot):
        return (super()._rotate_dm(dm[0], rot[0]), super()._rotate_dm(dm[1], rot[1]))

    @property
    def ncluster(self):
        if self.occtype == "occupied":
            return (self.dmet_bath.c_cluster_occ[0].shape[-1], self.dmet_bath.c_cluster_occ[1].shape[-1])
        if self.occtype == "virtual":
            return (self.dmet_bath.c_cluster_vir[0].shape[-1], self.dmet_bath.c_cluster_vir[1].shape[-1])

    def _dm_take_env(self, dm):
        ncluster = self.ncluster
        self.log.debugv("n(cluster)= (%d, %d)", ncluster[0], ncluster[1])
        self.log.debugv("tr(alpha-D)= %g", np.trace(dm[0]))
        self.log.debugv("tr( beta-D)= %g", np.trace(dm[1]))
        dm = (dm[0][ncluster[0] :, ncluster[0] :], dm[1][ncluster[1] :, ncluster[1] :])
        self.log.debugv("tr(alpha-D[env,env])= %g", np.trace(dm[0]))
        self.log.debugv("tr( beta-D[env,env])= %g", np.trace(dm[1]))
        return dm

    def _diagonalize_dm(self, dm):
        r_bno_a, n_bno_a = super()._diagonalize_dm(dm[0])
        r_bno_b, n_bno_b = super()._diagonalize_dm(dm[1])
        return (r_bno_a, r_bno_b), (n_bno_a, n_bno_b)

    def log_histogram(self, n_bno):
        if len(n_bno[0]) == len(n_bno[0]) == 0:
            return
        self.log.info("%s BNO histogram (alpha/beta):", self.occtype.capitalize())
        bins = np.hstack([-np.inf, np.logspace(-3, -10, 8)[::-1], np.inf])
        labels = "    " + "".join("{:{w}}".format("E-%d" % d, w=5) for d in range(3, 11))
        ha = helper.make_histogram(n_bno[0], bins=bins, labels=labels, rstrip=False).split("\n")
        hb = helper.make_histogram(n_bno[1], bins=bins, labels=labels).split("\n")
        for i in range(len(ha)):
            self.log.info(ha[i] + "   " + hb[i])

    def truncate_bno(self, coeff, occup, *args, **kwargs):
        c_bath_a, c_rest_a = super().truncate_bno(coeff[0], occup[0], *args, **kwargs)
        c_bath_b, c_rest_b = super().truncate_bno(coeff[1], occup[1], *args, **kwargs)
        return (c_bath_a, c_bath_b), (c_rest_a, c_rest_b)

    @staticmethod
    def _has_frozen(c_frozen):
        return (c_frozen[0].shape[-1] + c_frozen[1].shape[-1]) > 0


class MP2_BNO_Bath(BNO_Bath):
    def __init__(self, *args, project_dmet_order=0, project_dmet_mode="full", project_dmet=None, **kwargs):
        # Backwards compatibility:
        if project_dmet:
            project_dmet_order = 1
            project_dmet_mode = project_dmet
        self.project_dmet_order = project_dmet_order
        self.project_dmet_mode = project_dmet_mode
        super().__init__(*args, **kwargs)
        if project_dmet:
            # Log isn't set at the top of the function
            self.log.warning("project_dmet is deprecated; use project_dmet_order and project_dmet_mode.")

    def _make_t2(self, actspace, fock, eris=None, max_memory=None, blksize=None, energy_only=False):
        """Make T2 amplitudes and pair correlation energies."""

        if eris is None:
            eris, cderi, cderi_neg = self.get_eris_or_cderi(actspace)
        # (ov|ov)
        if eris is not None:
            self.log.debugv("Making T2 amplitudes from ERIs")
            assert eris.ndim == 4
            nocc, nvir = eris.shape[:2]
        # (L|ov)
        elif cderi is not None:
            self.log.debugv("Making T2 amplitudes from CD-ERIs")
            assert cderi.ndim == 3
            assert cderi_neg is None or cderi_neg.ndim == 3
            nocc, nvir = cderi.shape[1:]
        else:
            raise ValueError()

        # Fragment projector:
        ovlp = self.base.get_ovlp()
        rfrag = dot(actspace.c_active_occ.T, ovlp, self.c_frag)

        t2 = np.empty((nocc, nocc, nvir, nvir)) if not energy_only else None
        mo_energy = self._get_mo_energy(fock, actspace)
        eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
        max_memory = max_memory or int(1e9)
        if blksize is None:
            blksize = int(max_memory / max(nocc * nvir * nvir * 8, 1))
        nenv = nocc if self.occtype == "occupied" else nvir
        ecorr = 0
        for blk in brange(0, nocc, blksize):
            if eris is not None:
                gijab = eris[blk].transpose(0, 2, 1, 3)
            else:
                gijab = einsum("Lia,Ljb->ijab", cderi[:, blk], cderi)
                if cderi_neg is not None:
                    gijab -= einsum("Lia,Ljb->ijab", cderi_neg[:, blk], cderi_neg)
            eijab = eia[blk][:, None, :, None] + eia[None, :, None, :]
            t2blk = gijab / eijab
            if not energy_only:
                t2[blk] = t2blk
            # Projected correlation energy:
            tp = einsum("ix,i...->x...", rfrag[blk], t2blk)
            gp = einsum("ix,i...->x...", rfrag[blk], gijab)
            ecorr += 2 * einsum("ijab,ijab->", tp, gp) - einsum("ijab,ijba->", tp, gp)

        return t2, ecorr

    def _get_mo_energy(self, fock, actspace):
        c_act = actspace.c_active
        mo_energy = einsum("ai,ab,bi->i", c_act, fock, c_act)
        return mo_energy

    def _get_eris(self, actspace):
        # We only need the (ov|ov) block for MP2:
        mo_coeff = 2 * [actspace.c_active_occ, actspace.c_active_vir]
        eris = self.base.get_eris_array(mo_coeff)
        return eris

    def _get_cderi(self, actspace):

        BOHR = 0.529177210903

        def atoms_within_rcut(mol, center_atom, rcut, unit="Ang", include_center=True):
            """
            Return atom indices within distance rcut of center_atom.
            """
            center_atom = int(center_atom)
            coords_bohr = mol.atom_coords()             # (natm, 3) in Bohr
            center_bohr = mol.atom_coord(center_atom)   # (3,) in Bohr
            d_bohr = np.linalg.norm(coords_bohr - center_bohr[None, :], axis=1)

            unit_l = unit.lower()
            if unit_l.startswith("ang"):
                d = d_bohr * BOHR
            elif unit_l.startswith("b"):
                d = d_bohr
            else:
                raise ValueError(f"Invalid unit: {unit}")

            atoms = np.where(d <= float(rcut))[0].tolist()
            if not include_center and center_atom in atoms:
                atoms.remove(center_atom)
            return atoms

        def expand_atoms_by_distance(mol, atoms_seed, radius, unit="Ang"):
            """Return atoms within `radius` of any atom in atoms_seed."""
            # radius in Ang or Bohr
            coords = mol.atom_coords()  # Bohr
            if unit.lower().startswith("ang"):
                rad_bohr = radius / 0.529177210903
            else:
                rad_bohr = radius
            seed = np.asarray(atoms_seed, dtype=int)
            seed_coords = coords[seed]
            # compute min distance to seed set for every atom
            dmin = np.full(mol.natm, np.inf)
            for R in seed_coords:
                d = np.linalg.norm(coords - R[None, :], axis=1)
                dmin = np.minimum(dmin, d)
            keep = np.where(dmin <= rad_bohr)[0].tolist()
            return keep

        def ao_indices_for_atoms(mol, atoms):
            """Return AO row indices in the full mol corresponding to a list of atom indices."""
            atoms = list(map(int, atoms))
            aosl = mol.aoslice_by_atom()
            idx = []
            for a in atoms:
                p0, p1 = aosl[a][2], aosl[a][3]
                idx.extend(range(p0, p1))
            return np.asarray(idx, dtype=int)

        from scipy.linalg import cho_factor, cho_solve  # faster + stable for SPD overlap
        def build_local_aux_ctx(mf_full, atoms_prim, auxbasis=None, verbose=0):
            """
            Build and cache local MF (with DF) for a truncated molecule defined by atoms_prim.

            Returns a dict ctx with:
            - mf_loc
            - ao_idx_full
            - S_lf  (rows of S_full corresponding to local AOs)
            - chol  (Cholesky factorization of S_loc)
            - tag   (hashable identifier)
            """
            mol_full = mf_full.mol
            atoms_prim = list(map(int, atoms_prim))

            # --- Build local primary mol from subset of atoms (keep order)
            atom_spec = [mol_full.atom[i] for i in atoms_prim]

            mol_loc = gto.Mole()
            mol_loc.atom = atom_spec
            mol_loc.unit = mol_full.unit
            mol_loc.basis = mol_full.basis
            mol_loc.charge = mol_full.charge  # keep consistent by default
            mol_loc.spin = mol_full.spin
            # --- FIX electron parity mismatch ---
            ne = mol_loc.nelectron
            if (ne - mol_loc.spin) % 2 != 0:
                # Minimal correction: flip spin by 1
                mol_loc.spin = mol_loc.spin + 1
            mol_loc.build(verbose=verbose)

            # --- Dummy RHF + DF build (no SCF needed)
            mf_loc = scf.RHF(mol_loc)
            mf_loc.verbose = verbose

            if auxbasis is None:
                auxbasis = getattr(getattr(mf_full, "with_df", None), "auxbasis", None)
            if auxbasis is None:
                raise RuntimeError("auxbasis not found. Pass auxbasis or ensure mf_full.with_df.auxbasis exists.")

            mf_loc.with_df = pyscf_df.DF(mol_loc)
            mf_loc.with_df.auxbasis = auxbasis
            mf_loc.with_df.build()

            # --- AO index map full -> local
            ao_idx_full = ao_indices_for_atoms(mol_full, atoms_prim)

            # --- Overlaps needed for projection
            S_full = mf_full.get_ovlp()
            S_loc = mf_loc.get_ovlp()  # (nao_loc, nao_loc)

            # Cross overlap rows (since mol_loc built from subset in same atom order):
            S_lf = S_full[np.ix_(ao_idx_full, np.arange(S_full.shape[0]))]  # (nao_loc, nao_full)

            # Factorize S_loc once; reuse for occupied+virtual projections
            chol = cho_factor(S_loc, lower=True, check_finite=False)

            tag = (tuple(atoms_prim), str(auxbasis), int(mol_loc.nao_nr()))
            return dict(mf_loc=mf_loc, ao_idx_full=ao_idx_full, S_lf=S_lf, chol=chol, tag=tag)


        def project_mos_to_local(ctx, C_occ_full, C_vir_full):
            """
            Solve S_loc * C_proj = (S_lf @ C_full) for both occ and vir.
            Uses cached Cholesky of S_loc.
            """
            B_occ = ctx["S_lf"] @ C_occ_full
            B_vir = ctx["S_lf"] @ C_vir_full

            Cocc_loc = cho_solve(ctx["chol"], B_occ, check_finite=False)
            Cvir_loc = cho_solve(ctx["chol"], B_vir, check_finite=False)
            return Cocc_loc, Cvir_loc

        def _tail_weight(C_full, ao_keep, S_full=None):
            """
            Tail weights outside ao_keep for each MO column.
            Returns (w_tail_per_orb, w_tail_max).
            If S_full is provided: uses overlap-metric tail.
            """
            ao_keep = np.asarray(ao_keep, dtype=int)
            nao, nmo = C_full.shape
            keep_mask = np.zeros(nao, dtype=bool)
            keep_mask[ao_keep] = True

            C_out = C_full[~keep_mask, :]  # (nao_out, nmo)

            # Simple coefficient norm tail
            if S_full is None:
                w = np.sum(C_out * C_out, axis=0)
                return w, float(np.max(w)) if w.size else 0.0

            # Build S_kk and B^T S C (where B selects kept AOs)
            S_kk = S_full[np.ix_(ao_keep, ao_keep)]
            B = S_full[np.ix_(ao_keep, np.arange(nao))] @ C_full  # shape (nkeep, nmo)

            chol = cho_factor(S_kk, lower=True, check_finite=False)
            X = cho_solve(chol, B, check_finite=False)            # X = S_kk^{-1} (S_k,: C)

            # w_in = (B^T) S_kk^{-1} B  per column
            w_in = np.einsum("ki,ki->i", B, X)

            # orbital norms in S metric
            w_norm = np.einsum("pi,pq,qi->i", C_full, S_full, C_full)
            # if these deviate from 1, tail interpretation gets fuzzy
            print("S-norm stats:", w_norm.min(), w_norm.mean(), w_norm.max())

            w_tail = 1.0 - w_in
            # Clip tiny numerical excursions
            w = np.clip(w_tail, -1e-12, 1.0)

            return w, float(np.max(w)) if w.size else 0.0

        # We only need the (L|ov) block for MP2:
        mo_coeff_full = (actspace.c_active_occ, actspace.c_active_vir)

        # Options
        bath_opts = getattr(self.base.opts, "bath_options", {}) if hasattr(self.base, "opts") else {}
        local_aux_enable = bath_opts.get("local_aux_enable", False)
        if not (local_aux_enable and self.spin_restricted):
            return self.base.get_cderi(mo_coeff_full)

        rcut = bath_opts.get("rcut", 5.0)
        rcut_unit = bath_opts.get("unit", "Ang")
        local_aux_radius = bath_opts.get("local_aux_radius", 5.0)
        local_aux_unit = bath_opts.get("local_aux_unit", "Ang")
        local_aux_print = bath_opts.get("local_aux_print", False)

        # Define local atoms (deterministic)
        center_atom = int(self.fragment.atoms[0])
        atoms_support = atoms_within_rcut(self.mol, center_atom, rcut=float(rcut), unit=rcut_unit)
        atoms_aux = expand_atoms_by_distance(self.mol, atoms_support, float(local_aux_radius), unit=local_aux_unit)
        atoms_aux = sorted(set(map(int, atoms_aux)))

        # Cache container on embedding object (shared across occ/vir instances)
        if not hasattr(self.base, "_local_aux_ctx_cache"):
            self.base._local_aux_ctx_cache = {}

        # Build/reuse ctx
        auxbasis = getattr(getattr(self.base.mf, "with_df", None), "auxbasis", None)
        cache_key = (tuple(atoms_aux), str(auxbasis))

        ctx = self.base._local_aux_ctx_cache.get(cache_key, None)
        if ctx is None:
            ctx = build_local_aux_ctx(self.base.mf, atoms_aux, auxbasis=auxbasis, verbose=0)

            # Store required items explicitly as requested
            self.base.mf_local = ctx["mf_loc"]
            self.base.ao_idx_full = ctx["ao_idx_full"]
            self.base._local_ctx_tag = ctx["tag"]

            self.base._local_aux_ctx_cache[cache_key] = ctx

            if local_aux_print:
                self.log.info(
                    "Local-aux ctx built: support atoms=%d aux atoms=%d  tag=%r",
                    len(atoms_support), len(atoms_aux), self.base._local_ctx_tag
                )
        else:
            # Ensure these are visible even when reused
            self.base.mf_local = ctx["mf_loc"]
            self.base.ao_idx_full = ctx["ao_idx_full"]
            self.base._local_ctx_tag = ctx["tag"]

            if local_aux_print:
                self.log.info(
                    "Local-aux ctx reused: support atoms=%d aux atoms=%d  tag=%r",
                    len(atoms_support), len(atoms_aux), self.base._local_ctx_tag
                )
        
        # ---------------------------------------------------------------------
        # AS: diagnostics only — tail weight of FULL-space orbitals outside local AO set
        # ---------------------------------------------------------------------
        if local_aux_print:
            # Full AO overlap (for overlap-metric tail)
            S_full = self.base.mf.get_ovlp()

            ao_keep = ctx["ao_idx_full"]  # full AO indices kept by local atoms_aux

            # Tail for occupied-active and virtual-active (full-space coeffs)
            Cocc_full = mo_coeff_full[0]
            Cvir_full = mo_coeff_full[1]

            w_occ_nom, wocc_nom_max = _tail_weight(Cocc_full, ao_keep, S_full=None)
            w_vir_nom, wvir_nom_max = _tail_weight(Cvir_full, ao_keep, S_full=None)

            w_occ_S, wocc_S_max = _tail_weight(Cocc_full, ao_keep, S_full=S_full)
            w_vir_S, wvir_S_max = _tail_weight(Cvir_full, ao_keep, S_full=S_full)

            self.log.info("Tail weight outside local AOs (keep=%d AOs):", len(ao_keep))
            self.log.info("  occ  no-metric: max=%.3e  mean=%.3e", wocc_nom_max, float(np.mean(w_occ_nom)) if w_occ_nom.size else 0.0)
            self.log.info("  vir  no-metric: max=%.3e  mean=%.3e", wvir_nom_max, float(np.mean(w_vir_nom)) if w_vir_nom.size else 0.0)
            self.log.info("  occ  overlap-S: max=%.3e  mean=%.3e", wocc_S_max,   float(np.mean(w_occ_S))   if w_occ_S.size else 0.0)
            self.log.info("  vir  overlap-S: max=%.3e  mean=%.3e", wvir_S_max,   float(np.mean(w_vir_S))   if w_vir_S.size else 0.0)


        # Project current actspace orbitals into local AO basis using cached overlaps
        Cocc_loc, Cvir_loc = project_mos_to_local(ctx, mo_coeff_full[0], mo_coeff_full[1])

        # Do DF AO->MO on local mf
        from vayesta.core.eris import get_cderi_df
        cderi, cderi_neg = get_cderi_df(ctx["mf_loc"], (Cocc_loc, Cvir_loc))
        return cderi, cderi_neg

    def get_eris_or_cderi(self, actspace):
        eris = cderi = cderi_neg = None
        t0 = timer()
        if self.fragment.base.has_df:
            cderi, cderi_neg = self._get_cderi(actspace)
        else:
            eris = self._get_eris(actspace)
        self.log.timingv("Time for AO->MO transformation: %s", time_string(timer() - t0))
        # TODO: Reuse previously obtained integral transformation into N^2 sized quantity (rather than N^4)
        # else:
        #    self.log.debug("Transforming previous eris.")
        #    eris = transform_mp2_eris(eris, actspace.c_active_occ, actspace.c_active_vir, ovlp=self.base.get_ovlp())
        return eris, cderi, cderi_neg

    def _get_dmet_projector_weights(self, eig):
        assert np.all(eig > -1e-10)
        assert np.all(eig - 1 < 1e-10)
        eig = np.clip(eig, 0, 1)
        mode = self.project_dmet_mode
        if mode == "full":
            weights = np.zeros(len(eig))
        elif mode == "half":
            weights = np.full(len(eig), 0.5)
        elif mode == "linear":
            weights = 2 * abs(np.fmin(eig, 1 - eig))
        elif mode == "cosine":
            weights = (1 - np.cos(2 * eig * np.pi)) / 2
        elif mode == "cosine-half":
            weights = (1 - np.cos(2 * eig * np.pi)) / 4
        elif mode == "entropy":
            weights = 4 * eig * (1 - eig)
        elif mode == "sqrt-entropy":
            weights = 2 * np.sqrt(eig * (1 - eig))
        elif mode == "squared-entropy":
            weights = (4 * eig * (1 - eig)) ** 2
        else:
            raise ValueError("Invalid value for project_dmet_mode: %s" % mode)
        assert np.all(weights > -1e-14)
        assert np.all(weights - 1 < 1e-14)
        weights = np.clip(weights, 0, 1)
        return weights

    def _project_t2(self, t2, actspace):
        """Project and symmetrize T2 amplitudes"""
        self.log.info(
            "Projecting DMET space for MP2 bath (mode= %s, order= %d).", self.project_dmet_mode, self.project_dmet_order
        )
        weights = self._get_dmet_projector_weights(self.dmet_bath.n_dmet)
        weights = hstack(self.fragment.n_frag * [1], weights)
        ovlp = self.fragment.base.get_ovlp()
        c_fragdmet = hstack(self.fragment.c_frag, self.dmet_bath.c_dmet)
        if self.occtype == "occupied":
            rot = dot(actspace.c_active_vir.T, ovlp, c_fragdmet)
            proj = einsum("ix,x,jx->ij", rot, weights, rot)
            if self.project_dmet_order == 1:
                t2 = einsum("xa,ijab->ijxb", proj, t2)
            elif self.project_dmet_order == 2:
                t2 = einsum("xa,yb,ijab->ijxy", proj, proj, t2)
            else:
                raise ValueError
        elif self.occtype == "virtual":
            rot = dot(actspace.c_active_occ.T, ovlp, c_fragdmet)
            proj = einsum("ix,x,jx->ij", rot, weights, rot)
            if self.project_dmet_order == 1:
                t2 = einsum("xi,i...->x...", proj, t2)
            elif self.project_dmet_order == 2:
                t2 = einsum("xi,yj,ij...->xy...", proj, proj, t2)
            else:
                raise ValueError
        t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
        return t2

    def make_delta_dm1(self, t2, actspace):
        """Delta MP2 density matrix"""

        if self.project_dmet_order > 0:
            t2 = self._project_t2(t2, actspace)

        # This is equivalent to:
        # do, dv = pyscf.mp.mp2._gamma1_intermediates(mp2, eris=eris)
        # do, dv = -2*do, 2*dv
        if self.occtype == "occupied":
            dm = 2 * einsum("ikab,jkab->ij", t2, t2) - einsum("ikab,jkba->ij", t2, t2)
        elif self.occtype == "virtual":
            dm = 2 * einsum("ijac,ijbc->ab", t2, t2) - einsum("ijac,ijcb->ab", t2, t2)
        assert np.allclose(dm, dm.T)
        return dm

    def make_bno_coeff(self, eris=None):
        """Construct MP2 bath natural orbital coefficients and occupation numbers.

        This routine works for both for spin-restricted and unrestricted.

        Parameters
        ----------
        eris: mp2._ChemistERIs

        Returns
        -------
        c_bno: (n(AO), n(BNO)) array
            Bath natural orbital coefficients.
        n_bno: (n(BNO)) array
            Bath natural orbital occupation numbers.
        """
        t_init = timer()

        actspace_orig = self.get_active_space()
        fock = self.base.get_fock_for_bath()

        # --- Canonicalization [optional]
        if self.canonicalize[0]:
            self.log.debugv("Canonicalizing occupied orbitals")
            c_active_occ, r_occ = self.fragment.canonicalize_mo(actspace_orig.c_active_occ, fock=fock)
        else:
            c_active_occ = actspace_orig.c_active_occ
            r_occ = None
        if self.canonicalize[1]:
            self.log.debugv("Canonicalizing virtual orbitals")
            c_active_vir, r_vir = self.fragment.canonicalize_mo(actspace_orig.c_active_vir, fock=fock)
        else:
            c_active_vir = actspace_orig.c_active_vir
            r_vir = None
        actspace = Cluster.from_coeffs(
            c_active_occ, c_active_vir, actspace_orig.c_frozen_occ, actspace_orig.c_frozen_vir
        )

        nocc_a = actspace.c_active_occ.shape[-1]
        nvir_a = actspace.c_active_vir.shape[-1]
        nocc_f = actspace.c_frozen_occ.shape[-1]
        nvir_f = actspace.c_frozen_vir.shape[-1]
        self.log.info(
                "ClusterRHF(norb_active=%d, norb_frozen=%d)  [occA=%d virA=%d occF=%d virF=%d]",
                nocc_a + nvir_a, nocc_f + nvir_f, nocc_a, nvir_a, nocc_f, nvir_f
        )

        #import pdb; pdb.set_trace()

        t0 = timer()
        t2, ecorr = self._make_t2(actspace, fock, eris=eris)
        t_amps = timer() - t0

        dm = self.make_delta_dm1(t2, actspace)

        # --- Undo canonicalization
        if self.occtype == "occupied" and r_occ is not None:
            dm = self._rotate_dm(dm, r_occ)
        elif self.occtype == "virtual" and r_vir is not None:
            dm = self._rotate_dm(dm, r_vir)
        # --- Diagonalize environment-environment block
        dm = self._dm_take_env(dm)
        t0 = timer()
        r_bno, n_bno = self._diagonalize_dm(dm)
        t_diag = timer() - t0
        c_bno = spinalg.dot(self.c_env, r_bno)
        c_bno = fix_orbital_sign(c_bno)[0]

        self.log.timing(
            "Time MP2 bath:  amplitudes= %s  diagonal.= %s  total= %s",
            *map(time_string, (t_amps, t_diag, (timer() - t_init))),
        )

        return c_bno, n_bno, ecorr


class UMP2_BNO_Bath(MP2_BNO_Bath, BNO_Bath_UHF):
    def _get_mo_energy(self, fock, actspace):
        c_act_a, c_act_b = actspace.c_active
        mo_energy_a = einsum("ai,ab,bi->i", c_act_a, fock[0], c_act_a)
        mo_energy_b = einsum("ai,ab,bi->i", c_act_b, fock[1], c_act_b)
        return (mo_energy_a, mo_energy_b)

    def _get_eris(self, actspace):
        # We only need the (ov|ov) block for MP2:
        return self.base.get_eris_array_uhf(actspace.c_active_occ, mo_coeff2=actspace.c_active_vir)

    def _get_cderi(self, actspace):
        # We only need the (ov|ov) block for MP2:
        mo_a = [actspace.c_active_occ[0], actspace.c_active_vir[0]]
        mo_b = [actspace.c_active_occ[1], actspace.c_active_vir[1]]
        cderi_a, cderi_neg_a = self.base.get_cderi(mo_a)
        cderi_b, cderi_neg_b = self.base.get_cderi(mo_b)
        return (cderi_a, cderi_b), (cderi_neg_a, cderi_neg_b)

    def _make_t2(self, actspace, fock, eris=None, max_memory=None, blksize=None, energy_only=False):
        """Make T2 amplitudes"""

        if eris is None:
            eris, cderi, cderi_neg = self.get_eris_or_cderi(actspace)
        # (ov|ov)
        if eris is not None:
            assert len(eris) == 3
            assert eris[0].ndim == 4
            assert eris[1].ndim == 4
            assert eris[2].ndim == 4
            nocca, nvira = eris[0].shape[:2]
            noccb, nvirb = eris[2].shape[:2]
        # (L|ov)
        elif cderi is not None:
            assert len(cderi) == 2
            assert cderi[0].ndim == 3
            assert cderi[1].ndim == 3
            nocca, nvira = cderi[0].shape[1:]
            noccb, nvirb = cderi[1].shape[1:]
        else:
            raise ValueError()

        # Fragment projector:
        ovlp = self.base.get_ovlp()
        rfrag = spinalg.dot(spinalg.T(actspace.c_active_occ), ovlp, self.c_frag)

        if not energy_only:
            t2aa = np.empty((nocca, nocca, nvira, nvira))
            t2ab = np.empty((nocca, noccb, nvira, nvirb))
            t2bb = np.empty((noccb, noccb, nvirb, nvirb))
        else:
            t2aa = t2ab = t2bb = None
        mo_energy = self._get_mo_energy(fock, actspace)
        eia_a = mo_energy[0][:nocca, None] - mo_energy[0][None, nocca:]
        eia_b = mo_energy[1][:noccb, None] - mo_energy[1][None, noccb:]

        # Alpha-alpha and Alpha-beta:
        max_memory = max_memory or int(1e9)
        if blksize is None:
            blksize_a = int(max_memory / max(nocca * nvira * nvira * 8, 1))
        else:
            blksize_a = blksize
        ecorr = 0
        for blk in brange(0, nocca, blksize_a):
            # Alpha-alpha
            if eris is not None:
                gijab = eris[0][blk].transpose(0, 2, 1, 3)
            else:
                gijab = einsum("Lia,Ljb->ijab", cderi[0][:, blk], cderi[0])
                if cderi_neg[0] is not None:
                    gijab -= einsum("Lia,Ljb->ijab", cderi_neg[0][:, blk], cderi_neg[0])
            eijab = eia_a[blk][:, None, :, None] + eia_a[None, :, None, :]
            t2blk = gijab / eijab
            t2blk -= t2blk.transpose(0, 1, 3, 2)
            if not energy_only:
                t2aa[blk] = t2blk
            # Projected correlation energy:
            tp = einsum("ix,i...->x...", rfrag[0][blk], t2blk)
            gp = einsum("ix,i...->x...", rfrag[0][blk], gijab)
            ecorr += (einsum("ijab,ijab->", tp, gp) - einsum("ijab,ijba->", tp, gp)) / 4
            # Alpha-beta
            if eris is not None:
                gijab = eris[1][blk].transpose(0, 2, 1, 3)
            else:
                gijab = einsum("Lia,Ljb->ijab", cderi[0][:, blk], cderi[1])
                if cderi_neg[0] is not None:
                    gijab -= einsum("Lia,Ljb->ijab", cderi_neg[0][:, blk], cderi_neg[1])
            eijab = eia_a[blk][:, None, :, None] + eia_b[None, :, None, :]
            t2blk = gijab / eijab
            if not energy_only:
                t2ab[blk] = t2blk
            # Projected correlation energy:
            # Alpha projected:
            tp = einsum("ix,i...->x...", rfrag[0][blk], t2blk)
            gp = einsum("ix,i...->x...", rfrag[0][blk], gijab)
            ecorr += einsum("ijab,ijab->", tp, gp) / 2
            # Beta projected:
            tp = einsum("jx,ij...->ix...", rfrag[1], t2blk)
            gp = einsum("jx,ij...->ix...", rfrag[1], gijab)
            ecorr += einsum("ijab,ijab->", tp, gp) / 2

        # Beta-beta:
        if blksize is None:
            blksize_b = int(max_memory / max(noccb * nvirb * nvirb * 8, 1))
        else:
            blksize_b = blksize
        for blk in brange(0, noccb, blksize_b):
            if eris is not None:
                gijab = eris[2][blk].transpose(0, 2, 1, 3)
            else:
                gijab = einsum("Lia,Ljb->ijab", cderi[1][:, blk], cderi[1])
                if cderi_neg[0] is not None:
                    gijab -= einsum("Lia,Ljb->ijab", cderi_neg[1][:, blk], cderi_neg[1])
            eijab = eia_b[blk][:, None, :, None] + eia_b[None, :, None, :]
            t2blk = gijab / eijab
            t2blk -= t2blk.transpose(0, 1, 3, 2)
            if not energy_only:
                t2bb[blk] = t2blk
            # Projected correlation energy:
            tp = einsum("ix,i...->x...", rfrag[1][blk], t2blk)
            gp = einsum("ix,i...->x...", rfrag[1][blk], gijab)
            ecorr += (einsum("ijab,ijab->", tp, gp) - einsum("ijab,ijba->", tp, gp)) / 4

        return (t2aa, t2ab, t2bb), ecorr

    def _project_t2(self, t2, actspace):
        """Project and symmetrize T2 amplitudes"""
        self.log.info(
            "Projecting DMET space for MP2 bath (mode= %s, order= %d).", self.project_dmet_mode, self.project_dmet_order
        )
        weightsa = self._get_dmet_projector_weights(self.dmet_bath.n_dmet[0])
        weightsb = self._get_dmet_projector_weights(self.dmet_bath.n_dmet[1])
        weightsa = hstack(self.fragment.n_frag[0] * [1], weightsa)
        weightsb = hstack(self.fragment.n_frag[1] * [1], weightsb)

        # Project and symmetrize:
        t2aa, t2ab, t2bb = t2
        ovlp = self.fragment.base.get_ovlp()
        c_fragdmet_a = hstack(self.fragment.c_frag[0], self.dmet_bath.c_dmet[0])
        c_fragdmet_b = hstack(self.fragment.c_frag[1], self.dmet_bath.c_dmet[1])
        if self.occtype == "occupied":
            rota = dot(actspace.c_active_vir[0].T, ovlp, c_fragdmet_a)
            rotb = dot(actspace.c_active_vir[1].T, ovlp, c_fragdmet_b)
            proja = einsum("ix,x,jx->ij", rota, weightsa, rota)
            projb = einsum("ix,x,jx->ij", rotb, weightsb, rotb)
            if self.project_dmet_order == 1:
                t2aa = einsum("xa,ijab->ijxb", proja, t2aa)
                t2bb = einsum("xa,ijab->ijxb", projb, t2bb)
                t2ab = (einsum("xa,ijab->ijxb", proja, t2ab) + einsum("xb,ijab->ijax", projb, t2ab)) / 2
            # Not tested:
            elif self.project_dmet_order == 2:
                t2aa = einsum("xa,yb,ijab->ijxy", proja, proja, t2aa)
                t2bb = einsum("xa,yb,ijab->ijxy", projb, projb, t2bb)
                t2ab = (
                    einsum("xa,yb,ijab->ijxy", proja, projb, t2ab) + einsum("xb,ya,ijab->ijyx", projb, proja, t2ab)
                ) / 2
            else:
                raise ValueError
        elif self.occtype == "virtual":
            rota = dot(actspace.c_active_occ[0].T, ovlp, c_fragdmet_a)
            rotb = dot(actspace.c_active_occ[1].T, ovlp, c_fragdmet_b)
            proja = einsum("ix,x,jx->ij", rota, weightsa, rota)
            projb = einsum("ix,x,jx->ij", rotb, weightsb, rotb)
            if self.project_dmet_order == 1:
                t2aa = einsum("xi,i...->x...", proja, t2aa)
                t2bb = einsum("xi,i...->x...", projb, t2bb)
                t2ab = (einsum("xi,i...->x...", proja, t2ab) + einsum("xj,ij...->ix...", projb, t2ab)) / 2
            # Not tested:
            elif self.project_dmet_order == 2:
                t2aa = einsum("xi,yj,ij...->xy...", proja, proja, t2aa)
                t2bb = einsum("xi,yj,ij...->xy...", projb, projb, t2bb)
                t2ab = (
                    einsum("xi,yj,ij...->xy...", proja, projb, t2ab) + einsum("xj,yi,ij...->yx...", projb, proja, t2ab)
                ) / 2
            else:
                raise ValueError
        t2aa = (t2aa + t2aa.transpose(1, 0, 3, 2)) / 2
        t2bb = (t2bb + t2bb.transpose(1, 0, 3, 2)) / 2
        return (t2aa, t2ab, t2bb)

    def make_delta_dm1(self, t2, actspace):
        """Delta MP2 density matrix"""

        if self.project_dmet_order > 0:
            t2 = self._project_t2(t2, actspace)

        t2aa, t2ab, t2bb = t2
        # Construct occupied-occupied DM
        if self.occtype == "occupied":
            dma = einsum("imef,jmef->ij", t2aa.conj(), t2aa) / 2 + einsum("imef,jmef->ij", t2ab.conj(), t2ab)
            dmb = einsum("imef,jmef->ij", t2bb.conj(), t2bb) / 2 + einsum("mief,mjef->ij", t2ab.conj(), t2ab)
        # Construct virtual-virtual DM
        elif self.occtype == "virtual":
            dma = einsum("mnae,mnbe->ba", t2aa.conj(), t2aa) / 2 + einsum("mnae,mnbe->ba", t2ab.conj(), t2ab)
            dmb = einsum("mnae,mnbe->ba", t2bb.conj(), t2bb) / 2 + einsum("mnea,mneb->ba", t2ab.conj(), t2ab)
        assert np.allclose(dma, dma.T)
        assert np.allclose(dmb, dmb.T)
        return (dma, dmb)


# ================================================================================================ #

#    if self.opts.plot_orbitals:
#        #bins = np.hstack((-np.inf, np.self.logspace(-9, -3, 9-3+1), np.inf))
#        bins = np.hstack((1, np.self.logspace(-3, -9, 9-3+1), -1))
#        for idx, upper in enumerate(bins[:-1]):
#            lower = bins[idx+1]
#            mask = np.self.logical_and((dm_occ > lower), (dm_occ <= upper))
#            if np.any(mask):
#                coeff = c_rot[:,mask]
#                self.log.info("Plotting MP2 bath density between %.0e and %.0e containing %d orbitals." % (upper, lower, coeff.shape[-1]))
#                dm = np.dot(coeff, coeff.T)
#                dset_idx = (4001 if kind == "occ" else 5001) + idx
#                self.cubefile.add_density(dm, dset_idx=dset_idx)
