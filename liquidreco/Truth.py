import numpy as np
from math import comb
from scipy.optimize import linear_sum_assignment

class HitTruth:
    """Truth information of one list of hits, same order as the hits.

    Per-hit scalars are numpy arrays. The segment fields are object arrays with one
    entry per hit, each entry holding one value per G4 segment of that hit (or None
    if the input file had no segment truth).
    """

    NO_PARTICLE = -1   # label for hits without any segment

    def __init__(self, n_segs, n_trajs, edep,
                 seg_starts=None, seg_stops=None, seg_edeps=None,
                 seg_lengths=None, seg_primary_ids=None, seg_contrib_ids=None):
        self.n_segs = np.asarray(n_segs, dtype=int)
        self.n_trajs = np.asarray(n_trajs, dtype=int)
        self.edep = np.asarray(edep, dtype=float)

        self.seg_starts = seg_starts            # per hit: [nSegs, 4]
        self.seg_stops = seg_stops              # per hit: [nSegs, 4]
        self.seg_edeps = seg_edeps              # per hit: [nSegs]
        self.seg_lengths = seg_lengths          # per hit: [nSegs]
        self.seg_primary_ids = seg_primary_ids  # per hit: [nSegs]
        self.seg_contrib_ids = seg_contrib_ids  # per hit: list of nSegs arrays

    def __len__(self):
        return len(self.edep)

    @property
    def has_segments(self):
        return self.seg_edeps is not None

    def select(self, mask):
        def sel(a):
            return None if a is None else a[mask]
        return HitTruth(
            self.n_segs[mask], self.n_trajs[mask], self.edep[mask],
            sel(self.seg_starts), sel(self.seg_stops), sel(self.seg_edeps),
            sel(self.seg_lengths), sel(self.seg_primary_ids), sel(self.seg_contrib_ids),
        )

    # ---------- derived quantities (replace the old "main track" fields) ----------
    def edep_by_primary(self):
        """Per hit: (primary_ids, edeps), energy summed over all segments of each primary."""
        assert self.has_segments, "no segment truth in this file"
        out = []
        for prim, edep in zip(self.seg_primary_ids, self.seg_edeps):
            if len(prim) == 0:
                out.append((np.empty(0, dtype=int), np.empty(0)))
                continue
            ids, inv = np.unique(prim, return_inverse=True)
            sums = np.zeros(len(ids))
            np.add.at(sums, inv, edep)
            out.append((ids, sums))
        return out

    @property
    def main_primary_id(self):
        """Primary with the most deposited energy in each hit (-1 if the hit has no segments).
        Equivalent to the old primary_id."""
        return np.array([
            ids[np.argmax(e)] if len(ids) else self.NO_PARTICLE
            for ids, e in self.edep_by_primary()
        ], dtype=int)

    @property
    def main_primary_fraction(self):
        """Fraction of the hit's energy that comes from its main primary."""
        return np.array([
            e.max() / e.sum() if len(e) and e.sum() > 0 else 0.0
            for _, e in self.edep_by_primary()
        ])


class ContingencyMatrix:
    """Rows = reco tracks (-1 = unassigned hits), columns = true particles."""
    UNASSIGNED = -1

    def __init__(self, r_ids, t_ids, M):
        self.r_ids = np.asarray(r_ids)
        self.t_ids = np.asarray(t_ids)
        self.M = np.asarray(M, dtype=float)

    @classmethod
    def from_labels(cls, reco_label, true_label, weight):
        r_ids, r = np.unique(reco_label, return_inverse=True)
        t_ids, t = np.unique(true_label, return_inverse=True)
        M = np.zeros((len(r_ids), len(t_ids)))
        np.add.at(M, (r, t), weight)
        return cls(r_ids, t_ids, M)

    def __add__(self, other):
        """Sum two matrices of the SAME event (e.g. zy + xz), aligned on the union of ids."""
        r_all = np.union1d(self.r_ids, other.r_ids)
        t_all = np.union1d(self.t_ids, other.t_ids)
        M = np.zeros((len(r_all), len(t_all)))
        for c in (self, other):
            rows = np.searchsorted(r_all, c.r_ids)
            cols = np.searchsorted(t_all, c.t_ids)
            M[np.ix_(rows, cols)] += c.M
        return ContingencyMatrix(r_all, t_all, M)

    # ---------- views ----------
    def total(self):
        return self.M.sum()

    def _real(self):
        """Matrix without the unassigned row, and the ids of the remaining rows."""
        keep = self.r_ids != self.UNASSIGNED
        return self.M[keep], self.r_ids[keep]

    def normalized(self, how="total"):
        if how == "total":
            return self.M / max(self.total(), 1e-12)
        if how == "column":    # per true particle: where did its weight go
            return self.M / np.maximum(self.M.sum(axis=0, keepdims=True), 1e-12)
        if how == "row":       # per reco track: what is it made of
            return self.M / np.maximum(self.M.sum(axis=1, keepdims=True), 1e-12)
        raise ValueError(how)

    # ---------- metrics ----------
    def purity(self):
        """Per reco track: largest share of a single particle. Returns {reco_id: purity}."""
        Mr, ids = self._real()
        return {int(i): row.max() / row.sum() for i, row in zip(ids, Mr) if row.sum() > 0}

    def completeness(self):
        """Per true particle: share of its weight in its best reco track (unassigned counts as lost)."""
        Mr, _ = self._real()
        col_tot = self.M.sum(axis=0)
        best = Mr.max(axis=0) if len(Mr) else np.zeros(len(self.t_ids))
        return {int(t): b / c for t, b, c in zip(self.t_ids, best, col_tot) if c > 0}

    def match(self, min_purity=0.5, min_completeness=0.5):
        """One-to-one matching by shared weight. Returns [(reco_id, true_id), ...]."""
        Mr, ids = self._real()
        if Mr.size == 0:
            return []
        rows, cols = linear_sum_assignment(-Mr)
        col_tot = self.M.sum(axis=0)
        matches = []
        for i, j in zip(rows, cols):
            shared = Mr[i, j]
            if shared <= 0:
                continue
            if shared / Mr[i].sum() >= min_purity and shared / col_tot[j] >= min_completeness:
                matches.append((int(ids[i]), int(self.t_ids[j])))
        return matches

    def pair_precision_recall(self):
        """Needs a matrix of hit COUNTS (build it with weight = ones)."""
        both = sum(comb(int(x), 2) for x in self.M.ravel())
        reco = sum(comb(int(x), 2) for x in self.M.sum(axis=1))
        true = sum(comb(int(x), 2) for x in self.M.sum(axis=0))
        return both / max(reco, 1), both / max(true, 1)

    def counts(self, **match_kwargs):
        """Raw numbers that can be summed over events. Ratios are computed at the very end."""
        Mr, ids = self._real()
        matches = self.match(**match_kwargs)
        rows = {int(r): k for k, r in enumerate(ids)}
        cols = {int(t): k for k, t in enumerate(self.t_ids)}
        correct = sum(Mr[rows[r], cols[t]] for r, t in matches)
        return {
            "n_true": len(self.t_ids),
            "n_reco": len(ids),
            "n_matched": len(matches),
            "weight_correct": correct,
            "weight_total": self.total(),
            "weight_unassigned": self.M[self.r_ids == self.UNASSIGNED].sum(),
        }

    def counts_by_type(self, pdg_of, group=lambda pdg: abs(int(pdg)), **match_kwargs):
        """Like counts(), but split by the type of the true particle.
        pdg_of: dict {true_id: pdg}. group: maps a pdg to the category key (here |pdg|, so e+ and e- are together)."""
        Mr, ids = self._real()
        rows = {int(r): k for k, r in enumerate(ids)}
        cols = {int(t): k for k, t in enumerate(self.t_ids)}
        matched_true = {t: r for r, t in self.match(**match_kwargs)}   # true_id -> reco_id
        col_tot = self.M.sum(axis=0)

        out = {}
        for t, k in cols.items():
            key = group(pdg_of[t])
            d = out.setdefault(key, {"n_true": 0, "n_matched": 0,
                                     "weight_total": 0.0, "weight_correct": 0.0})
            d["n_true"] += 1
            d["weight_total"] += col_tot[k]
            if t in matched_true:
                d["n_matched"] += 1
                d["weight_correct"] += Mr[rows[matched_true[t]], k]
        return out
