import numpy as np
from math import comb
from scipy.optimize import linear_sum_assignment

class HitTruth:
    """Truth information of one list of hits, same order as the hits. All fields are numpy arrays."""

    def __init__(self, track_id, primary_id, parent_id, pdg, edep, n_tracks, all_track_ids, all_edeps):
        self.track_id = np.asarray(track_id, dtype=int)
        self.primary_id = np.asarray(primary_id, dtype=int)
        self.parent_id = np.asarray(parent_id, dtype=int)
        self.pdg = np.asarray(pdg, dtype=int)
        self.edep = np.asarray(edep, dtype=float)
        self.n_tracks = np.asarray(n_tracks, dtype=int)
        self.all_track_ids = all_track_ids   # jagged, kept as it comes from the raw array
        self.all_edeps = all_edeps

    def __len__(self):
        return len(self.track_id)

    def select(self, mask):
        return HitTruth(
            self.track_id[mask], self.primary_id[mask], self.parent_id[mask],
            self.pdg[mask], self.edep[mask], self.n_tracks[mask],
            self.all_track_ids[mask], self.all_edeps[mask],
        )



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
