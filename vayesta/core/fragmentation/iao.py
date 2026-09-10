import os.path
import numpy as np
#########AS#########
import os
import hashlib
import h5py
#########AS#########

import pyscf
import pyscf.lo

from vayesta.core.util import dot, einsum, fix_orbital_sign
from vayesta.core.fragmentation.fragmentation import Fragmentation
from vayesta.core.fragmentation.ufragmentation import Fragmentation_UHF

# Load default minimal basis set on module initialization
default_minao = {}
path = os.path.dirname(__file__)
with open(os.path.join(path, "minao.dat"), "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        (basis, minao) = line.split()
        if minao == "none":
            minao = None
        default_minao[basis] = minao


def get_default_minao(basis):
    # TODO: Add more to data file
    if not isinstance(basis, str):
        return "minao"
    bas = basis.replace("-", "").lower()
    minao = default_minao.get(bas, "minao")
    if minao is None:
        raise ValueError("Could not chose minimal basis for basis %s automatically!", basis)
    return minao

#########AS#########
def _mol_fingerprint(mol):
    """Fingerprint to ensure cache matches the exact geometry+basis."""
    coords = mol.atom_coords(unit="Ang")
    h = hashlib.sha1()
    h.update(str(mol.natm).encode())
    h.update(str(mol.basis).encode())
    h.update(np.asarray(coords, dtype=np.float64).tobytes())
    return h.hexdigest()


def _mo_fingerprint(mo_coeff, mo_occ):
    """Fingerprint of the MO coefficients/occupations used to build IAOs.

    Needed in addition to _mol_fingerprint: two calls can share the same molecule
    (geometry+basis) but use different orbitals - e.g. the alpha and beta channels of an
    unrestricted calculation, or two different mean-field references - and must not be
    conflated in the cache.
    """
    h = hashlib.sha1()
    h.update(np.asarray(mo_coeff, dtype=np.float64).tobytes())
    h.update(np.asarray(mo_occ, dtype=np.float64).tobytes())
    return h.hexdigest()
#########AS#########


class IAO_Fragmentation(Fragmentation):
    name = "IAO"

    def __init__(self, *args, minao="auto", cache_file="iao_cache.h5", **kwargs):
        super().__init__(*args, **kwargs)
        if minao.lower() == "auto":
            minao = get_default_minao(self.mol.basis)
            self.log.info(
                "IAO:  computational basis= %s  minimal reference basis= %s (automatically chosen)",
                self.mol.basis,
                minao,
            )
        else:
            self.log.debug("IAO:  computational basis= %s  minimal reference basis= %s", self.mol.basis, minao)
        self.minao = minao
#########AS#########
        self.cache_file = cache_file
        self._mol_fp = _mol_fingerprint(self.mol)
#########AS#########
        try:
            self.refmol = pyscf.lo.iao.reference_mol(self.mol, minao=self.minao)
        except IndexError as e:
            if hasattr(self.mol, "space_group_symmetry"):
                if self.mol.space_group_symmetry:
                    self.log.error("Could not find IAOs when using space group symmetry.")
                    self.log.error("This is a known issue with some PySCF versions.")
                    self.log.error(
                        "Please set `emb.mf.mol.space_group_symmetry=False` when initialising the fragmentation (it can be turned back on afterwards)."
                    )
                    raise ValueError(
                        "Could not find IAOs when using space group symmetry. Please set `emb.mf.mol.space_group_symmetry=False`."
                    )
            raise e

    @property
    def n_iao(self):
        return self.refmol.nao

#########AS#########
    def _try_load_cache(self, add_virtuals, mo_fp):
        if not self.cache_file:
            self.log.info("IAO cache disabled (cache_file is None/empty).")
            return None
        if not os.path.isfile(self.cache_file):
            self.log.info("IAO CACHE MISS: %s not found", self.cache_file)
            return None

        try:
            with h5py.File(self.cache_file, "r") as f:
                if f.attrs.get("name", "") != "vayesta-iao-cache":
                    self.log.info("IAO CACHE MISS: wrong cache signature")
                    return None
                if f.attrs.get("mol_fp", "") != self._mol_fp:
                    self.log.info("IAO CACHE MISS: molecule fingerprint mismatch")
                    return None
                if f.attrs.get("mo_fp", "") != mo_fp:
                    self.log.info("IAO CACHE MISS: MO coefficient/occupation fingerprint mismatch")
                    return None
                if f.attrs.get("basis", "") != str(self.mol.basis):
                    self.log.info("IAO CACHE MISS: basis mismatch")
                    return None
                if f.attrs.get("minao", "") != str(self.minao):
                    self.log.info("IAO CACHE MISS: minao mismatch")
                    return None
                if bool(f.attrs.get("add_virtuals", True)) != bool(add_virtuals):
                    self.log.info("IAO CACHE MISS: add_virtuals mismatch")
                    return None

                c = f["C_iao"][...]
                self.log.info("IAO CACHE HIT: loaded C_iao from %s (shape=%s)", self.cache_file, c.shape)
                return c
        except Exception as e:
            self.log.warning("IAO CACHE MISS: failed reading %s (%s)", self.cache_file, e)
            return None

    def _write_cache(self, c_iao, add_virtuals, mo_fp):
        if not self.cache_file:
            return
        tmp = self.cache_file + ".tmp"
        try:
            with h5py.File(tmp, "w") as f:
                f.attrs["name"] = "vayesta-iao-cache"
                f.attrs["mol_fp"] = self._mol_fp
                f.attrs["mo_fp"] = mo_fp
                f.attrs["basis"] = str(self.mol.basis)
                f.attrs["minao"] = str(self.minao)
                f.attrs["add_virtuals"] = bool(add_virtuals)
                f.create_dataset("C_iao", data=c_iao, compression="gzip", compression_opts=1, shuffle=True)

            os.replace(tmp, self.cache_file)
            self.log.info("IAO CACHE WRITE: saved C_iao to %s (shape=%s)", self.cache_file, c_iao.shape)
        except Exception as e:
            self.log.warning("IAO cache write failed (%s): %s", self.cache_file, e)
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
            except Exception:
                pass


#########AS#########

    def get_coeff(self, mo_coeff=None, mo_occ=None, add_virtuals=True):
        """Make intrinsic atomic orbitals (IAOs).

        Returns
        -------
        c_iao : (n(AO), n(IAO)) array
            Orthonormalized IAO coefficients.
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
#########AS#########
        mo_fp = _mo_fingerprint(mo_coeff, mo_occ)
        cached = self._try_load_cache(add_virtuals=add_virtuals, mo_fp=mo_fp)
        if cached is not None:
            return cached

        self.log.info("IAO: building IAOs from scratch (cache miss).")
#########AS#########
        ovlp = self.get_ovlp()

        c_occ = mo_coeff[:, mo_occ > 0]
        c_iao = pyscf.lo.iao.iao(self.mol, c_occ, minao=self.minao)
        n_iao = c_iao.shape[-1]
        self.log.info(
            "n(AO)= %4d  n(MO)= %4d  n(occ-MO)= %4d  n(IAO)= %4d",
            mo_coeff.shape[0],
            mo_coeff.shape[-1],
            c_occ.shape[-1],
            n_iao,
        )

        # Orthogonalize IAO using symmetric (Lowdin) orthogonalization
        x, e_min = self.symmetric_orth(c_iao, ovlp)
        self.log.debugv(
            "Lowdin orthogonalization of IAOs: n(in)= %3d -> n(out)= %3d , min(eig)= %.3e",
            x.shape[0],
            x.shape[1],
            e_min,
        )
        if e_min < 1e-10:
            self.log.warning("Small eigenvalue in Lowdin orthogonalization: %.3e !", e_min)
        c_iao = np.dot(c_iao, x)
        # Check that all electrons are in IAO space
        self.check_nelectron(c_iao, mo_coeff, mo_occ)
        if add_virtuals:
            c_vir = self.get_virtual_coeff(c_iao, mo_coeff=mo_coeff)
            c_iao = np.hstack((c_iao, c_vir))
        # Test orthogonality of IAO
        self.check_orthonormal(c_iao)
#########AS#########
        self._write_cache(c_iao, add_virtuals=add_virtuals, mo_fp=mo_fp)
#########AS#########
        return c_iao

    def check_nelectron(self, c_iao, mo_coeff, mo_occ):
        dm = np.einsum("ai,i,bi->ab", mo_coeff, mo_occ, mo_coeff)
        ovlp = self.get_ovlp()
        ne_iao = einsum("ai,ab,bc,cd,di->", c_iao, ovlp, dm, ovlp, c_iao)
        ne_tot = einsum("ab,ab->", dm, ovlp)
        if abs(ne_iao - ne_tot) > 1e-8:
            self.log.error(
                "IAOs do not contain the correct number of electrons: IAO= %.8f  total= %.8f", ne_iao, ne_tot
            )
        else:
            self.log.debugv("Number of electrons: IAO= %.8f  total= %.8f", ne_iao, ne_tot)
        return ne_iao

    def get_labels(self):
        """Get labels of IAOs.

        Returns
        -------
        iao_labels : list of length nIAO
            Orbital label (atom-id, atom symbol, nl string, m string) for each IAO.
        """
        iao_labels_refmol = self.refmol.ao_labels(None)
        self.log.debugv("iao_labels_refmol: %r", iao_labels_refmol)
        if self.refmol.natm == self.mol.natm:
            iao_labels = iao_labels_refmol
        # If there are ghost atoms in the system, they will be removed in refmol.
        # For this reason, the atom IDs of mol and refmol will not agree anymore.
        # Here we will correct the atom IDs of refmol to agree with mol
        # (they will no longer be contiguous integers).
        else:
            ref2mol = []
            for refatm in range(self.refmol.natm):
                ref_coords = self.refmol.atom_coord(refatm)
                for atm in range(self.mol.natm):
                    coords = self.mol.atom_coord(atm)
                    if np.allclose(coords, ref_coords):
                        self.log.debugv("reference cell atom %r maps to atom %r", refatm, atm)
                        ref2mol.append(atm)
                        break
                else:
                    raise RuntimeError("No atom found with coordinates %r" % ref_coords)
            iao_labels = []
            for iao in iao_labels_refmol:
                iao_labels.append((ref2mol[iao[0]], iao[1], iao[2], iao[3]))
        self.log.debugv("iao_labels: %r", iao_labels)
        assert len(iao_labels_refmol) == len(iao_labels)
        return iao_labels

    def search_labels(self, labels):
        return self.refmol.search_ao_label(labels)

    def get_virtual_coeff(self, c_iao, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        ovlp = self.get_ovlp()
        # Add remaining virtual space, work in MO space, so that we automatically get the
        # correct linear dependency treatment, if n(MO) < n(AO)
        c_iao_mo = dot(mo_coeff.T, ovlp, c_iao)
        # Get eigenvectors of projector into complement
        p_iao = np.dot(c_iao_mo, c_iao_mo.T)
        p_rest = np.eye(p_iao.shape[-1]) - p_iao
        e, c = np.linalg.eigh(p_rest)

        # Corresponding expression in AO basis (but no linear-dependency treatment):
        # p_rest = ovlp - ovlp.dot(c_iao).dot(c_iao.T).dot(ovlp)
        # e, c = scipy.linalg.eigh(p_rest, ovlp)
        # c_rest = c[:,e>0.5]

        # Ideally, all eigenvalues of P_env should be 0 (IAOs) or 1 (non-IAO)
        # Error if > 1e-3
        mask_iao, mask_rest = (e <= 0.5), (e > 0.5)
        e_iao, e_rest = e[mask_iao], e[mask_rest]
        if np.any(abs(e_iao) > 1e-3):
            self.log.error("CRITICAL: Some IAO eigenvalues of 1-P_IAO are not close to 0:\n%r", e_iao)
        elif np.any(abs(e_iao) > 1e-6):
            self.log.warning(
                "Some IAO eigenvalues e of 1-P_IAO are not close to 0: n= %d max|e|= %.2e",
                np.count_nonzero(abs(e_iao) > 1e-6),
                abs(e_iao).max(),
            )
        if np.any(abs(1 - e_rest) > 1e-3):
            self.log.error("CRITICAL: Some non-IAO eigenvalues of 1-P_IAO are not close to 1:\n%r", e_rest)
        elif np.any(abs(1 - e_rest) > 1e-6):
            self.log.warning(
                "Some non-IAO eigenvalues e of 1-P_IAO are not close to 1: n= %d max|1-e|= %.2e",
                np.count_nonzero(abs(1 - e_rest) > 1e-6),
                abs(1 - e_rest).max(),
            )

        if not (np.sum(mask_rest) + c_iao.shape[-1] == mo_coeff.shape[-1]):
            self.log.critical(
                "Error in construction of remaining virtual orbitals! Eigenvalues of projector 1-P_IAO:\n%r", e
            )
            self.log.critical("Number of eigenvalues above 0.5 = %d", np.sum(mask_rest))
            self.log.critical("Total number of orbitals = %d", mo_coeff.shape[-1])
            raise RuntimeError("Incorrect number of remaining virtual orbitals")
        c_rest = np.dot(mo_coeff, c[:, mask_rest])  # Transform back to AO basis
        c_rest = fix_orbital_sign(c_rest)[0]

        self.check_orthonormal(np.hstack((c_iao, c_rest)), "IAO+virtual orbital")
        return c_rest


class IAO_Fragmentation_UHF(Fragmentation_UHF, IAO_Fragmentation):
    def get_coeff(self, mo_coeff=None, mo_occ=None, add_virtuals=True):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ

        self.log.info("Alpha-IAOs:")
        c_iao_a = IAO_Fragmentation.get_coeff(self, mo_coeff=mo_coeff[0], mo_occ=mo_occ[0], add_virtuals=add_virtuals)
        self.log.info(" Beta-IAOs:")
        c_iao_b = IAO_Fragmentation.get_coeff(self, mo_coeff=mo_coeff[1], mo_occ=mo_occ[1], add_virtuals=add_virtuals)
        return (c_iao_a, c_iao_b)
