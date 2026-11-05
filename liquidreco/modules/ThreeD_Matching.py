import numpy as np
#from matplotlib import pyplot as plt
from liquidreco.event import Event
#import matplotlib
from liquidreco.modules.module_base import ModuleBase
from  liquidreco.Tracks import SimpleTrack, Line, VertexCandidate
from scipy.optimize import linear_sum_assignment
from liquidreco import parameters

from liquidreco.save_used_modules import log_module_name
"""
def find_best_candidate(vertex1, vertex_list2, plane1):
    #Returns the best candidate for vertex1 of vertex_list2
    if not vertex_list2:
        return None, None
    if plane1 == "zy":
        z_coordinate1 = 0
        z_coordinate2 = 1
    elif plane1 == "xz":
        z_coordinate1 = 1
        z_coordinate2 = 0
    else:
        raise ValueError(f"Not an acceptable plane: {plane1}")

    z_position1 = vertex1.position[z_coordinate1]
    z_positions2 = np.array([vertex2.position[z_coordinate2] for vertex2 in vertex_list2])

    distances = abs(z_position1-z_positions2)
    i = np.argmin(distances)
    return vertex_list2[i], distances[i]




class ThreeDMatching(ModuleBase):
    def __init__(self):
        super().__init__()
        self.requirements = ["vertices_z_y", "vertices_x_z"]

    def _setup_cli_options(self, parser):
        pass

    def _initialise(self) -> None:
        pass

    def _process(self, event:Event):
        vertices_z_y = event["vertices_z_y"]
        vertices_x_z = event["vertices_x_z"]
        x_z_candidates = []
        for vertex_z_y in vertices_z_y:
            x_z_candidates.append(find_best_candidate(vertex_z_y, vertices_x_z, "zy")) #is a list of elements of vertices_x_z
        #x_z_candiate_indices = [x_z_candidate.track_id for x_z_candidate in x_z_candidates[:,0]]
        valid = [i for i, c in enumerate(x_z_candidates) if c[0] is not None]
        cand_ids = np.array([x_z_candidates[i][0].vertex_id for i in valid])
        dists    = np.array([x_z_candidates[i][1] for i in valid])

        pairs = []
        for x_z_index in np.unique(cand_ids):
            mask = cand_ids == x_z_index                 # True where this xz vertex was chosen
            k_local = np.argmin(dists[mask])             # best one among the chosen
            k = np.where(mask)[0][k_local]               # back to the index in the full list
            best = x_z_candidates[k]                     # (xz_vertex, distance) of the winner
            pairs.append([vertices_z_y[k], best[0], best[1]])
        event.add_data("zy-xz_pairs", pairs)



    def _help(self)->str:
        return "Matches vertices from the zy and xz planes"
    
    def _finalise(self):
        pass
"""

def vertexMatching(vertices_z_y, vertices_x_z):
        if not vertices_z_y or not vertices_x_z:
            return []
        distances = np.zeros((len(vertices_z_y), len(vertices_x_z)))
        positions_zy = np.array([vertex_z_y.position for vertex_z_y in vertices_z_y])
        positions_xz = np.array([vertex_x_z.position for vertex_x_z in vertices_x_z])
        for i, p_zy in enumerate(positions_zy):
            for j, p_xz in enumerate(positions_xz):
                distances[i,j] = abs(p_zy[0]-p_xz[1])
        rows, cols = linear_sum_assignment(distances)
        pairs = [[vertices_z_y[i], vertices_x_z[j], distances[i,j]] for i, j in zip(rows, cols) if distances[i,j] < parameters.ThreeDMatching_z_vertex_threshold]
        return pairs

def z_overlap(z_min_zy, z_max_zy, z_min_xz, z_max_xz):
    lo = max(z_min_zy, z_min_xz)
    hi = min(z_max_zy, z_max_xz)
    overlap = max(0.0, hi-lo)
    return overlap

def z_overlap_term(track_zy, track_xz):
    z_min_zy, z_max_zy = track_zy.z_range("zy")
    z_min_xz, z_max_xz = track_xz.z_range("xz")
    overlap = z_overlap(z_min_zy, z_max_zy, z_min_xz, z_max_xz)
    if overlap <= 0: 
        return None
    union = max(z_max_zy, z_max_xz)- min(z_min_zy, z_min_xz)
    return 1.0-overlap/union

def trackMatching(tracks_z_y, tracks_x_z):
    if not tracks_z_y or not tracks_x_z:
        return []
    overlaps = np.zeros((len(tracks_z_y), len(tracks_x_z)))
    for i, track_zy in enumerate(tracks_z_y):
        for j, track_xz in enumerate(tracks_x_z):
            term = z_overlap_term(track_zy, track_xz)
            overlaps[i,j] = 1e6 if term is None else term
    rows, cols = linear_sum_assignment(overlaps)
    pairs = [[tracks_z_y[i], tracks_x_z[j], overlaps[i,j]] for i, j in zip(rows, cols) if overlaps[i,j] < parameters.ThreeDMatching_track_overlap_threshold]
    return pairs
    



class ThreeDMatching(ModuleBase):
    def __init__(self):
        super().__init__()
        self.requirements = ["vertices_z_y", "vertices_x_z"]

    def _setup_cli_options(self, parser):
        pass

    def _initialise(self) -> None:
        pass

    def _process(self, event:Event):
        vertices_z_y = event["vertices_z_y"]
        vertices_x_z = event["vertices_x_z"]
        vertex_pairs = vertexMatching(vertices_z_y, vertices_x_z)
        event.add_data("vertex_pairs", vertex_pairs)

        tracks_z_y = event["tracks_zy"]
        tracks_x_z = event["tracks_xz"]
        track_pairs = trackMatching(tracks_z_y, tracks_x_z)
        event.add_data("track_pairs", track_pairs)
        
        



    def _help(self)->str:
        return "Matches vertices from the zy and xz planes"
    
    def _finalise(self):
        log_module_name("ThreeDMatching", file_path="modules.txt")








