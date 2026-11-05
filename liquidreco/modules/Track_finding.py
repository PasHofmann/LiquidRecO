import numpy as np
#from matplotlib import pyplot as plt
from liquidreco.event import Event
#import matplotlib
from liquidreco.modules.module_base import ModuleBase
import ROOT
from ROOT import std
from array import array
from liquidreco.modules.Hit_assigning import data_2D
from hough3d.basic_hough import Hough3D
from RANSAC_fit.RANSACRegressor import fit_lines_ransac, best_of_n, Line2D
from  liquidreco.Tracks import SimpleTrack, Line, VertexCandidate
from sklearn.cluster import DBSCAN
from liquidreco import parameters
from liquidreco.save_used_modules import log_module_name


def points_along_track(p_in, p_f):
    length = np.linalg.norm(p_f-p_in)
    step = 0.5
    n_points = int(length/step)
    t = np.linspace(0, 1, n_points)
    return p_in + t[:, None]*(p_f-p_in)

def distances(hits, point_along_track):
    return np.linalg.norm(hits - point_along_track, axis=1)

"""
def merge_lines(p1, p2):
    if not np.array_equal(p1, p2):
        if np.linalg.norm(p1 - p2) < 4.:
            return p1
        else:
            return p2
    else:
        return p2
"""
"""
def merge_closest_endpoints(line1, line2, threshold=4.):
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    dists = [np.linalg.norm(line1[i, :] - line2[j, :]) for i, j in pairs]
    min_idx = np.argmin(dists)

    if dists[min_idx] < threshold:
        i, j = pairs[min_idx]
        line2[j, :] = line1[i, :]

    return line2
"""
def merge_lines_global(in_and_f, threshold=4.):
    n = len(in_and_f)
    candidates = []

    # collect all endpoint-pair distances across all line pairs
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            for i in range(2):
                for j in range(2):
                    d = np.linalg.norm(in_and_f[a][i, :] - in_and_f[b][j, :])
                    if d < threshold:
                        candidates.append((d, a, i, b, j))

    # process smallest distances first
    candidates.sort(key=lambda c: c[0])

    merged = set()  # track (line_idx, endpoint_idx) already merged
    for d, a, i, b, j in candidates:
        if (b, j) in merged or (a, i) in merged:
            continue  # this endpoint already got its closest match
        in_and_f[b][j, :] = in_and_f[a][i, :]
        merged.add((b, j))
        merged.add((a, i))

    return in_and_f

def merge_lines_global_with_angle(in_and_f, threshold=4., aligned_threshold=15., angle_thresh_deg=15.):
    n = len(in_and_f)
    candidates = []

    directions = []
    for a in range(n):
        d = in_and_f[a][1, :] - in_and_f[a][0, :]
        norm = np.linalg.norm(d)
        directions.append(d / norm if norm > 0 else d)

    cos_thresh = np.cos(np.radians(angle_thresh_deg))

    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            for i in range(2):
                for j in range(2):
                    p_a = in_and_f[a][i, :]
                    p_b = in_and_f[b][j, :]
                    dist = np.linalg.norm(p_a - p_b)

                    # direction-to-direction parallelism (as before)
                    cos_dir = abs(np.dot(directions[a], directions[b]))
                    lines_parallel = cos_dir > cos_thresh

                    # NEW: colinearity check - does the connecting vector
                    # itself point along the lines' shared direction?
                    connector = p_b - p_a
                    connector_norm = np.linalg.norm(connector)
                    if connector_norm > 1e-9:
                        connector_unit = connector / connector_norm
                        cos_connector = abs(np.dot(connector_unit, directions[a]))
                        is_colinear = cos_connector > cos_thresh
                    else:
                        is_colinear = True  # points already coincide

                    is_aligned = lines_parallel and is_colinear
                    eff_threshold = aligned_threshold if is_aligned else threshold

                    if dist < eff_threshold:
                        candidates.append((dist, a, i, b, j))

    candidates.sort(key=lambda c: c[0])

    merged = set()
    for d, a, i, b, j in candidates:
        if (b, j) in merged or (a, i) in merged:
            continue
        mid = (in_and_f[a][i, :] + in_and_f[b][j, :]) / 2.0
        in_and_f[a][i, :] = mid
        in_and_f[b][j, :] = mid
        merged.add((a, i))
        merged.add((b, j))

    return in_and_f

def Track_assigning(in_and_f, total_charge_per_index):
    tracks = []
    lines = []
    line_number = len(in_and_f)
    print("\n line_number = ", line_number)
    print("\n len(total_charge_per_index) = ", len(total_charge_per_index))
    for i in range(line_number):
        if i != total_charge_per_index[i,0]:
            print("Something is wrong with indexing")
        lines.append(Line(in_and_f, i, total_charge = total_charge_per_index[i, 1]))
    for i in range(line_number):
        for j in range(i):
            if np.array_equal(lines[i].p_in, lines[j].p_in):
                lines[i].neighbour_id_in = j
                lines[j].neighbour_id_in = i
            elif np.array_equal(lines[i].p_in, lines[j].p_f):
                lines[i].neighbour_id_in = j
                lines[j].neighbour_id_f = i 
            elif np.array_equal(lines[i].p_f, lines[j].p_in):
                lines[i].neighbour_id_f = j
                lines[j].neighbour_id_in = i 
            elif np.array_equal(lines[i].p_f, lines[j].p_f):
                lines[i].neighbour_id_f = j
                lines[j].neighbour_id_f = i 

    j = 0
    for i in range(line_number):
        if lines[i].track_id == -1:
            tracks.append(SimpleTrack(j))
            tracks[j].Track_generator(lines, i)
            j += 1
    return tracks


def insert_vertices(tracks, in_and_f):
    tol = parameters.insert_vertices_threshold
    vertices = []
    vertex_positions = []
    for track1 in tracks:
        #for line1 in track1.lines:
        i = 0
        while i<len(track1.lines):
            line1 = track1.lines[i]
            for track2 in tracks:
                #for line2 in track2.lines:
                j = 0
                while j<len(track2.lines):
                    line2 = track2.lines[j]
                    j+=1
                    if line1.line_id == line2.line_id or line1.line_id == line2.neighbour_id_in or line1.line_id == line2.neighbour_id_f or np.linalg.norm(line1.p_in - line2.p_in)<tol or np.linalg.norm(line1.p_f - line2.p_in)<tol or np.linalg.norm(line1.p_in - line2.p_f)<tol or np.linalg.norm(line1.p_f - line2.p_f)<tol:
                        continue
                    for point in points_along_track(line2.p_in, line2.p_f):
                        if np.linalg.norm(line1.p_in-point)<parameters.insert_vertices_threshold:
                            line1.p_in = point.copy()
                            in_and_f = track2.line_split(line2.line_id, point, in_and_f)
                            in_and_f[line1.line_id, 0, :] = point
                            break
                        elif np.linalg.norm(line1.p_f-point)<parameters.insert_vertices_threshold:
                            line1.p_f = point.copy()
                            in_and_f = track2.line_split(line2.line_id, point, in_and_f)
                            in_and_f[line1.line_id, 1, :] = point
                            break
            i+=1
    for track in tracks:
        for line in track.lines:
            if not vertex_positions:
                vertices.append(VertexCandidate(line.p_in, len(vertices), line))
                vertex_positions.append(line.p_in)
                vertices.append(VertexCandidate(line.p_f, len(vertices), line))
                vertex_positions.append(line.p_f)

            distances_in = np.linalg.norm(np.array(vertex_positions)-line.p_in, axis=1)
            i = np.argmin(distances_in)
            if distances_in[i]<1e-1:
                vertices[i].add_neighbour_line(line)
                #continue #vertex is already in the list
            else:
                vertices.append(VertexCandidate(line.p_in, len(vertices), line))
                vertex_positions.append(line.p_in)

            distances_f = np.linalg.norm(np.array(vertex_positions)-line.p_f, axis=1)
            i = np.argmin(distances_f)
            if distances_f[i]<1e-1:
                vertices[i].add_neighbour_line(line)
                #continue #vertex is already in the list
            else:
                vertices.append(VertexCandidate(line.p_f, len(vertices), line))
                vertex_positions.append(line.p_f)

    for vertex1 in vertices:
        for vertex2 in vertices:
            if vertex1.vertex_id == vertex2.vertex_id:
                continue
            if vertex1.vertex_id in vertex2.neighbour_vertices:
                continue
            if set(vertex1.connected_line_ids) & set(vertex2.connected_line_ids):
                vertex1.neighbour_vertices.append(vertex2.vertex_id)
                vertex2.neighbour_vertices.append(vertex1.vertex_id)

            
    return in_and_f, vertices
                        




class CutLines(ModuleBase):
    #Divides the lines s.t. there must always be a hit close to the fitted line (removes the parts that are too far from the hits)

    
    def __init__(self):
        super().__init__()
        self.requirements = ["x_fiber_hits", "y_fiber_hits","in_and_f_z_y", "in_and_f_x_z", "line_index_zy", "line_index_xz"]

    def _setup_cli_options(self, parser):
        #parser.add_argument("--file_name", help="Chooses which ROOT File it uses", required = False, default = "Hough2D-results.root", type = str)
        pass

    def _initialise(self) -> None:
        pass
        #self.file_name = self.args.file_name
        #self.file = ROOT.TFile.Open(self.file_name, "UPDATE")
        #self.tree = self.file.Get("RecoTree")

        

        
    def _process(self, event:Event):
        self.min_charge_thresh=event["min_charge_thresh"]

        in_and_f_z_y=event["in_and_f_z_y"] #[track, in/f, z/y]->(k, 2, 2)
        in_and_f_x_z=event["in_and_f_x_z"]
        in_and_f_x_y=event["in_and_f_x_y"]

        x_fiber_hits = event["x_fiber_hits"]
        y_fiber_hits = event["y_fiber_hits"]
        z_fiber_hits = event["z_fiber_hits"]

        line_hit_threshold = parameters.CutLines_line_hit_threshold
        if x_fiber_hits:
            new_in_and_f_z_y=[]
            line_index_zy=np.array(event["line_index_zy"])
            for track, i in zip(in_and_f_z_y, range(len(in_and_f_z_y))):
                points = points_along_track(np.array(track)[1,:], np.array(track)[0,:])
                mask = line_index_zy == i
                zy_data = data_2D(x_fiber_hits, "z", "y", self.min_charge_thresh)
                hits=np.column_stack((zy_data[:, 0], zy_data[:, 1]))
                track_hits = hits[mask]
                track_start = track[0, :]
                track_end = np.array([0, 0])
                line=False

                for point in points:
                    d = distances(track_hits, point)
                    if d.size > 0 and np.min(d) > line_hit_threshold and line == True:
                        new_in_and_f_z_y.append([track_start, track_end])
                        line = False
                    track_end = point
                    if d.size > 0 and np.min(d) < line_hit_threshold and line == False:
                        track_start = point
                        line = True
                if line == True:
                    new_in_and_f_z_y.append([track_start, track_end])
            in_and_f_z_y = np.array(new_in_and_f_z_y)
            event.add_data("in_and_f_z_y", in_and_f_z_y)


        if y_fiber_hits:
            new_in_and_f_x_z=[]
            line_index_xz=np.array(event["line_index_xz"])
            for track, i in zip(in_and_f_x_z, range(len(in_and_f_x_z))):
                points = points_along_track(np.array(track)[1,:], np.array(track)[0,:])
                mask = line_index_xz == i
                xz_data = data_2D(y_fiber_hits, "x", "z", self.min_charge_thresh)
                hits=np.column_stack((xz_data[:, 0], xz_data[:, 1]))
                track_hits = hits[mask]
                track_start = track[0, :]
                track_end = np.array([0, 0])
                line=False

                for point in points:
                    d = distances(track_hits, point)
                    if d.size == 0:
                        continue
                    if np.min(d) > line_hit_threshold and line == True:
                        new_in_and_f_x_z.append([track_start, track_end])
                        line = False
                    track_end = point
                    if np.min(d) < line_hit_threshold and line == False:
                        track_start = point
                        line = True
                if line == True:
                    new_in_and_f_x_z.append([track_start, track_end])
            in_and_f_x_z = np.array(new_in_and_f_x_z)
            event.add_data("in_and_f_x_z", in_and_f_x_z)


        if z_fiber_hits:
            line_index_xy=event["line_index_xy"]

    def _help(self)->str:
        return "Divides the lines s.t. there must always be a hit close to the fitted line (removes the parts that are too far from the hits)\n This function has not been implemented for Z-direction fibers"

    def _finalise(self):
        log_module_name("CutLines", file_path="modules.txt")




class Refit_lines(ModuleBase):
    #does the fit again using the hits paring to the previous lines as clusters.
    def __init__(self):
        super().__init__()
        self.requirements = ["x_fiber_hits", "y_fiber_hits","in_and_f_z_y", "in_and_f_x_z", "line_index_zy", "line_index_xz"]

    def _setup_cli_options(self, parser):
        pass

    def _initialise(self) -> None:
        self.hough_finder = Hough3D(
            neighbour_dist=parameters.Refit_lines_neighbour_dist, min_points_per_line=parameters.Refit_lines_min_points_per_line, lattice_step_size=parameters.Refit_lines_lattice_step_size
        )
    
    def _process(self, event:Event):
        self.min_charge_thresh=event["min_charge_thresh"]
        x_fiber_hits = event["x_fiber_hits"]
        y_fiber_hits = event["y_fiber_hits"]
        z_fiber_hits = event["z_fiber_hits"]

        for fiber_hits, u_name, v_name in zip([x_fiber_hits, y_fiber_hits, z_fiber_hits], ["z", "x", "x"], ["y", "z", "y"]):
            if len(fiber_hits)==0:
                continue
            all_linePoints=np.empty((0,2,2))
            in_and_f_name = "in_and_f_" + u_name + "_" + v_name
            number_of_tracks = len(event[in_and_f_name])
            data = data_2D(fiber_hits, u_name, v_name, self.min_charge_thresh)
            line_index_name = "line_index_" + u_name + v_name
            line_index=np.array(event[line_index_name])
            hits=np.column_stack((data[:, 0], data[:, 1], np.zeros(len(data[:, 0]))))
            for i in range(number_of_tracks + 1):
                mask = line_index == i -1
                selected = hits[mask]
                if len(selected) < parameters.Refit_lines_min_points_per_line+1:
                    #linePoints = None
                    continue
                else:
                    linePoints = self.hough_finder(selected)
                all_linePoints=np.concatenate([all_linePoints, linePoints[:,:,:2]], axis=0)
            event.add_data(in_and_f_name, all_linePoints)

    def _finalise(self):
        log_module_name("Refit_lines", file_path="modules.txt")

            

class Ransac_refit_lines(ModuleBase):
    #does the fit again using the hits paring to the previous lines as clusters.
    def __init__(self):
        super().__init__()
        self.requirements = ["x_fiber_hits", "y_fiber_hits","in_and_f_z_y", "in_and_f_x_z", "line_index_zy", "line_index_xz"]

    def _setup_cli_options(self, parser):
        pass

    def _initialise(self) -> None:
        #self.hough_finder = Hough3D(neighbour_dist=1.5, min_points_per_line=4, lattice_step_size=1.)
        pass
    
    def _process(self, event:Event):
        self.min_charge_thresh=event["min_charge_thresh"]
        x_fiber_hits = event["x_fiber_hits"]
        y_fiber_hits = event["y_fiber_hits"]
        z_fiber_hits = event["z_fiber_hits"]

        for fiber_hits, u_name, v_name in zip([x_fiber_hits, y_fiber_hits, z_fiber_hits], ["z", "x", "x"], ["y", "z", "y"]):
            if len(fiber_hits)==0:
                continue
            all_linePoints=np.empty((0,2,2))
            in_and_f_name = "in_and_f_" + u_name + "_" + v_name
            number_of_tracks = len(event[in_and_f_name])
            data = data_2D(fiber_hits, u_name, v_name, self.min_charge_thresh)
            line_index_name = "line_index_" + u_name + v_name
            line_index=np.array(event[line_index_name])
            hits=np.column_stack((data[:, 0], data[:, 1]))
            for i in range(number_of_tracks):
                mask = line_index == i
                selected = hits[mask]
                if len(selected) < 5:
                    #linePoints = None
                    continue
                else:
                    linePoints = best_of_n(selected, Line2D, n_repeats=parameters.refit_n_repeats, line_penalty=parameters.refit_line_penalty, min_inliers=parameters.refit_min_inliers, max_lines=parameters.refit_max_lines, min_samples=parameters.refit_min_samples, residual_threshold=parameters.refit_residual_threshold, max_trials=parameters.refit_max_trials) #testing RANSAC
                    for start_end in linePoints:
                        all_linePoints=np.concatenate([all_linePoints, start_end[np.newaxis, :, :]], axis=0)
            event.add_data(in_and_f_name, all_linePoints)
            


    def _help(self)->str:
        return "does the fit again using the hits paring to the previous lines as clusters."

    def _finalise(self):
        log_module_name("Ransac_refit_lines", file_path="modules.txt")

class Find_simple_tracks(ModuleBase):
    def __init__(self):
        super().__init__()
        self.requirements = ["in_and_f_z_y", "in_and_f_x_z", "line_index_zy", "line_index_xz"]

    def _setup_cli_options(self, parser):
        pass

    def _initialise(self) -> None:
        pass
    
    def _process(self, event:Event):
        for u_name, v_name in zip(["z", "x", "x"], ["y", "z", "y"]):
            in_and_f_name = "in_and_f_" + u_name + "_" + v_name
            vertices_name = "vertices_" + u_name + "_" + v_name
            total_charge_per_index_name = "total_charge_per_index_" + u_name + v_name
            keys = event.get_keys()
            if not in_and_f_name in keys or not total_charge_per_index_name in keys:
                continue
            in_and_f = event[in_and_f_name]
            total_charge_per_index = event[total_charge_per_index_name]
            """
            for line1 in in_and_f:
                for line2 in in_and_f:
                    if np.array_equal(line1, line2):
                        continue
                    merge_closest_endpoints(line1, line2)
            event.add_data(in_and_f_name, in_and_f)
            """
            #in_and_f = merge_lines_global(in_and_f,  threshold=5)
            in_and_f = merge_lines_global_with_angle(in_and_f, threshold=parameters.Find_simple_tracks_threshold, aligned_threshold=parameters.Find_simple_tracks_aligned_threshold, angle_thresh_deg=parameters.Find_simple_tracks_angle_thresh_deg)
            tracks = Track_assigning(in_and_f, total_charge_per_index)
            tracks_name = "tracks_"+ u_name + v_name

            in_and_f, vertices = insert_vertices(tracks, in_and_f)
            event.add_data(tracks_name, tracks)
            event.add_data(in_and_f_name, in_and_f)
            event.add_data(vertices_name, vertices)

    def _help(self)->str:
        return "Connects lines that have close initial or final points"

    def _finalise(self):
        log_module_name("Find_simple_tracks", file_path="modules.txt")







def fit_line_tls(u, v, w, sigma=None):
    
    """Weighted total least squares. Works for any orientation.
    Returns endpoints [[u0, v0], [u1, v1]], or None if degenerate."""
    pts = np.column_stack((u, v))
    #w = np.ones(len(pts)) if sigma is None else 1.0 / np.asarray(sigma) ** 2
    w = w / w.sum()

    centre = (w[:, None] * pts).sum(axis=0)
    d = pts - centre
    cov = (w[:, None] * d).T @ d

    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[-1] < 1e-12:
        return None
    direction = eigvecs[:, -1]

    t = d @ direction
    return np.array([centre + t.min() * direction,
                      centre + t.max() * direction])


class Refit_lines_TLS(ModuleBase):
    def __init__(self):
        super().__init__()
        self.requirements = ["x_fiber_hits", "y_fiber_hits", "in_and_f_z_y",
                              "in_and_f_x_z", "line_index_zy", "line_index_xz"]

    def _setup_cli_options(self, parser):
        pass

    def _initialise(self) -> None:
        pass  # no longer need self.hough_finder

    def _process(self, event: Event):
        self.min_charge_thresh = event["min_charge_thresh"]
        x_fiber_hits = event["x_fiber_hits"]
        y_fiber_hits = event["y_fiber_hits"]
        z_fiber_hits = event["z_fiber_hits"]

        for fiber_hits, u_name, v_name in zip(
            [x_fiber_hits, y_fiber_hits, z_fiber_hits], ["z", "x", "x"], ["y", "z", "y"]
        ):
            if len(fiber_hits) == 0:
                continue

            in_and_f_name = "in_and_f_" + u_name + "_" + v_name
            old_lines = event[in_and_f_name]
            number_of_lines = len(old_lines)

            data = data_2D(fiber_hits, u_name, v_name, self.min_charge_thresh)
            line_index_name = "line_index_" + u_name + v_name
            line_index = np.array(event[line_index_name])

            refit_lines = []
            for i in range(number_of_lines):
                mask = line_index == i
                selected = data[mask]
                if len(selected) < 5:
                    refit_lines.append(old_lines[i])
                    continue
                line = fit_line_tls(selected[:, 0], selected[:, 1], selected[:, 2])
                refit_lines.append(old_lines[i] if line is None else line)
                
            event.add_data(in_and_f_name, np.array(refit_lines))
    def _help(self)->str:
        return "does a weighted total least squarre fit using the hits paring to the previous lines as clusters."

    def _finalise(self):
        log_module_name("Refit_lines_TLS", file_path="modules.txt")





class Unasigned_refit_lines(ModuleBase):
    #does the hough fit on the unassigned hits
    def __init__(self):
        super().__init__()
        self.requirements = ["x_fiber_hits", "y_fiber_hits","in_and_f_z_y", "in_and_f_x_z", "line_index_zy", "line_index_xz"]

    def _setup_cli_options(self, parser):
        pass

    def _initialise(self) -> None:
        self.hough_finder = Hough3D(
            neighbour_dist=parameters.Unasigned_refit_lines_neighbour_dist, min_points_per_line=parameters.Unasigned_refit_lines_min_points_per_line, lattice_step_size=parameters.Unasigned_refit_lines_lattice_step_size
        )
        self.clusterer = DBSCAN(parameters.Unasigned_refit_lines_DBSCAN_epsilon, min_samples=2)
        self.time_clusterer = DBSCAN(eps= parameters.Unasigned_refit_lines_TimeSlice_epsilon, min_samples=2)
    
    def _process(self, event:Event):
        self.min_charge_thresh=event["min_charge_thresh"]
        x_fiber_hits = event["x_fiber_hits"]
        y_fiber_hits = event["y_fiber_hits"]
        z_fiber_hits = event["z_fiber_hits"]

        for fiber_hits, u_name, v_name in zip([x_fiber_hits, y_fiber_hits, z_fiber_hits], ["z", "x", "x"], ["y", "z", "y"]):
            if len(fiber_hits)==0:
                continue
            #all_linePoints=np.empty((0,2,2))
            in_and_f_name = "in_and_f_" + u_name + "_" + v_name
            all_linePoints = event[in_and_f_name]
            number_of_tracks = len(event[in_and_f_name])
            data = data_2D(fiber_hits, u_name, v_name, self.min_charge_thresh)
            time = np.array([hit.time for hit in fiber_hits if hit.weight > self.min_charge_thresh])
            
            line_index_name = "line_index_" + u_name + v_name
            line_index=np.array(event[line_index_name])
            mask = line_index == -1
            hits=np.column_stack((data[:, 0], data[:, 1], np.zeros(len(data[:, 0]))))
            selected = hits[mask]
            selected_time = time[mask]
            if len(selected) == 0:
                continue
            time_cluster_ids = self.time_clusterer.fit_predict(selected_time.reshape(-1, 1))
            for time_cluster_id in np.unique(time_cluster_ids):
                if time_cluster_id == -1:
                    continue
                cluster_ids = self.clusterer.fit_predict(selected[time_cluster_ids==time_cluster_id])
                for cluster_id in np.unique(cluster_ids):
                    if cluster_id == -1:
                        # -1 is id for "noise"
                        continue
                    cluster_points = hits[mask][time_cluster_ids==time_cluster_id][cluster_ids == cluster_id]
                    if len(cluster_points) < parameters.Unasigned_refit_lines_min_points_per_cluster:
                        #linePoints = None
                        continue
                    else:
                        linePoints = self.hough_finder(cluster_points)
                        all_linePoints=np.concatenate([all_linePoints, linePoints[:,:,:2]], axis=0)
            event.add_data(in_and_f_name, all_linePoints)

    def _finalise(self):
        log_module_name("Unasigned_refit_lines", file_path="modules.txt")


class Unasigned_RANSAC_refit_lines(ModuleBase):
    #does the hough fit on the unassigned hits
    def __init__(self):
        super().__init__()
        self.requirements = ["x_fiber_hits", "y_fiber_hits","in_and_f_z_y", "in_and_f_x_z", "line_index_zy", "line_index_xz"]

    def _setup_cli_options(self, parser):
        pass

    def _initialise(self) -> None:
        self.hough_finder = Hough3D(
            neighbour_dist=parameters.Unasigned_refit_lines_neighbour_dist, min_points_per_line=parameters.Unasigned_refit_lines_min_points_per_line, lattice_step_size=parameters.Unasigned_refit_lines_lattice_step_size
        )
        self.clusterer = DBSCAN(parameters.Unasigned_RANSAC_refit_lines_DBSCAN_epsilon)
    
    def _process(self, event:Event):
        self.min_charge_thresh=event["min_charge_thresh"]
        x_fiber_hits = event["x_fiber_hits"]
        y_fiber_hits = event["y_fiber_hits"]
        z_fiber_hits = event["z_fiber_hits"]

        for fiber_hits, u_name, v_name in zip([x_fiber_hits, y_fiber_hits, z_fiber_hits], ["z", "x", "x"], ["y", "z", "y"]):
            if len(fiber_hits)==0:
                continue
            #all_linePoints=np.empty((0,2,2))
            in_and_f_name = "in_and_f_" + u_name + "_" + v_name
            all_linePoints = event[in_and_f_name]
            number_of_tracks = len(event[in_and_f_name])
            data = data_2D(fiber_hits, u_name, v_name, self.min_charge_thresh)
            
            line_index_name = "line_index_" + u_name + v_name
            line_index=np.array(event[line_index_name])
            mask = line_index == -1
            hits=np.column_stack((data[:, 0], data[:, 1])) #, np.zeros(len(data[:, 0]))
            selected = hits[mask]
            if len(selected) == 0:
                continue
            cluster_ids = self.clusterer.fit_predict(selected)
            for cluster_id in np.unique(cluster_ids):
                if cluster_id == -1:
                    # -1 is id for "noise"
                    continue
                cluster_points = hits[mask][cluster_ids == cluster_id]
                if len(cluster_points) < parameters.Unasigned_RANSAC_refit_lines_min_points_per_cluster:
                    #linePoints = None
                    continue
                else:
                    linePoints = best_of_n(selected, Line2D, n_repeats=parameters.Unasigned_RANSAC_refit_lines_n_repeats, line_penalty=parameters.Unasigned_RANSAC_refit_lines_line_penalty, min_inliers=parameters.Unasigned_RANSAC_refit_lines_min_inliers, max_lines=parameters.Unasigned_RANSAC_refit_lines_max_lines, min_samples=parameters.Unasigned_RANSAC_refit_lines_min_samples, residual_threshold=parameters.Unasigned_RANSAC_refit_lines_residual_threshold, max_trials=parameters.Unasigned_RANSAC_refit_lines_max_trials) #testing RANSAC
                    for start_end in linePoints:
                        all_linePoints=np.concatenate([all_linePoints, start_end[np.newaxis, :, :]], axis=0)
            event.add_data(in_and_f_name, all_linePoints)


    def _finalise(self):
        log_module_name("Unasigned_RANSAC_refit_lines", file_path="modules.txt")





