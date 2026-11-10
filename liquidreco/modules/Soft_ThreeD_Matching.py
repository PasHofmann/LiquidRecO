import numpy as np
import math
from itertools import permutations
#from matplotlib import pyplot as plt
from liquidreco.event import Event
#import matplotlib
from liquidreco.modules.module_base import ModuleBase
from collections import defaultdict
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from itertools import zip_longest, chain

from  liquidreco.Tracks import Line, SoftVertexCandidate, SoftTrack, ThreeDTrack, turning_point_class, turning_point_matches, Node

from liquidreco import parameters
from liquidreco.save_used_modules import log_module_name
from liquidreco.modules.Hit_assigning import data_2D

def points_along_track(p_in, p_f): #Creates an np.array of points along the line. The points are at half a milimeter from eachother
    length = np.linalg.norm(p_f-p_in)
    step = 0.5
    n_points = int(length/step)
    t = np.linspace(0, 1, n_points)
    return p_in + t[:, None]*(p_f-p_in)

def make_list_of_lines(in_and_f, total_charge_per_index): #Reads in_and_f and returns a list of line objects
    lines = []
    line_number = len(in_and_f)
    for i in range(line_number):
        if i != total_charge_per_index[i,0]:
            print("Something is wrong with indexing")
        lines.append(Line(in_and_f, i, total_charge = total_charge_per_index[i, 1]))
    return lines

def find_vertex_net(vertex_candidates):
    #Parse over all the pairs of vertices
    for vertex1 in vertex_candidates:
        for vertex2 in vertex_candidates:
            #Exclude vertex pairs that are 2 times the same vertex as well as already paired vertices
            if vertex1.vertex_id == vertex2.vertex_id or vertex1.vertex_id in vertex2.neighbour_vertices or vertex1.vertex_id in vertex2.mut_exc_vertices:
                continue
            #Check if the vertices in the pair share a lines
            attachments1 = np.array(vertex1.attachments, dtype = object)
            attachments2 = np.array(vertex2.attachments, dtype = object)
            connected_line_ids1 = attachments1[:, 0]
            connected_line_ids2 = attachments2[:, 0]
            shared = set(connected_line_ids1) & set(connected_line_ids2)
            if shared:
                
                for line_id in shared:
                    #Check if the 2 vertices already know eachother: either already neighbours or known to exclude eachother
                    if vertex1.vertex_id in vertex2.neighbour_vertices or vertex1.vertex_id in vertex2.mut_exc_vertices:
                        continue
                    i = np.flatnonzero(connected_line_ids1 == line_id)[0]
                    j = np.flatnonzero(connected_line_ids2 == line_id)[0]
                    #If the 2 vertices are attacked to the same endpoint of the line they exclude eachother
                    if attachments1[i,1] == attachments2[j,1]:
                        vertex1.mut_exc_vertices.append(vertex2.vertex_id)
                        vertex2.mut_exc_vertices.append(vertex1.vertex_id)
                    #The vertices are neighbours
                    else:
                        vertex1.neighbour_vertices.append(vertex2.vertex_id)
                        vertex2.neighbour_vertices.append(vertex1.vertex_id)
    return vertex_candidates


def distance(data, in_x, in_y, f_x, f_y):
    x = data[:, 0]
    y = data[:, 1]
    dx = f_x - in_x
    dy = f_y - in_y
    seg_len_sq = dx**2 + dy**2
    if seg_len_sq == 0:
        return np.hypot(x - in_x, y - in_y)
    t = ((x - in_x) * dx + (y - in_y) * dy) / seg_len_sq #projection of the hit onto the line
    #t = max(0, min(1, t)) 
    t = np.clip(t, 0, 1) #chooses the closest end point of the line for the projection point of hits that project outside of the line endpoints
    closest_x= in_x + t*dx
    closest_y= in_y + t*dy
    return np.hypot(x - closest_x, y - closest_y)

def check_line_overlap(p1_in, p1_f, p2_in, p2_f, direction1, direction2, overlaped_angle_threshold = parameters.check_line_overlap_angle_threshold, overlaped_width_threshold = parameters.check_line_overlap_width_threshold):
    cos_thresh = np.cos(np.radians(overlaped_angle_threshold))
    distances_line1_endpoints2 = distance(np.array([p2_in, p2_f]), p1_in[0], p1_in[1], p1_f[0], p1_f[1])
    distances_line2_endpoints1 = distance(np.array([p1_in, p1_f]), p2_in[0], p2_in[1], p2_f[0], p2_f[1])
    cos_dir = abs(np.dot(direction1, direction2))
    if cos_dir > cos_thresh and (min(distances_line1_endpoints2) <= overlaped_width_threshold and min(distances_line2_endpoints1) <= overlaped_width_threshold):
        return True
    else:
        return False




def make_list_of_vertex_candidates(lines, threshold = parameters.make_list_of_vertex_candidates_threshold, aligned_threshold = parameters.make_list_of_vertex_candidates_aligned_threshold, angle_thresh_deg =parameters.make_list_of_vertex_candidates_angle_thresh_deg):
    #Takes the list of lines, and using the same conditions as for the greedy track assignment creates vertex candidates. 
    #Vertex candidates know to which line (and endpoint) and vertices they are connected.
    n = len(lines)
    vertex_candidates = []

    #We compute all the normalized directions of the lines
    directions = []
    for line in lines:
        d=line.p_f - line.p_in
        norm = np.linalg.norm(d)
        directions.append(d/norm if norm > 0 else d)


    cos_thresh = np.cos(np.radians(angle_thresh_deg))
    for i in range(n):
        for j in range(i+1,n):
            line1 = lines[i]
            line2 = lines[j]
            p1_in = line1.p_in
            p2_in = line2.p_in
            p1_f = line1.p_f
            p2_f = line2.p_f
            overlapping = check_line_overlap(p1_in, p1_f, p2_in, p2_f, directions[i], directions[j])

            #We want to test each combination of end points of the lines to see which ones are to be considered possibly connected
            for p1, end1 in zip([p1_in, p1_f], ["in", "f"]):
                for p2, end2 in zip([p2_in, p2_f], ["in", "f"]):
                    out1 = directions[i] if end1 == "in" else -directions[i]   # points away from the vertex
                    out2 = directions[j] if end2 == "in" else -directions[j]
                    if np.dot(out1, out2) > 0:
                        continue

                    if overlapping:
                        point_index = np.argmin(np.array([np.linalg.norm(p1_in-p2_in), np.linalg.norm(p1_in-p2_f), np.linalg.norm(p1_f-p2_in), np.linalg.norm(p1_f-p2_f)]))
                        points = np.array([[p1_in, p2_in], [p1_in, p2_f], [p1_f, p2_in], [p1_f, p2_f]])
                        ends = np.array([["in", "in"], ["in", "f"], ["f", "in"], ["f", "f"]])
                        if (end1, end2) != tuple(ends[point_index]):
                            continue
                        p1 = points[point_index, 0]
                        p2 = points[point_index, 1]
                        end1 = ends[point_index, 0]
                        end2 = ends[point_index, 1]
                        vertex_id = len(vertex_candidates)
                        vertex_candidates.append(SoftVertexCandidate(p1 + (p2 - p1) / 2, vertex_id))
                        vertex_candidates[vertex_id].add_neighbour_line(line1,end1, 0.)
                        vertex_candidates[vertex_id].add_neighbour_line(line2, end2, 0.)
                        continue
                            #We check whether the 2 lines have similar direction
                    cos_dir = abs(np.dot(directions[i], directions[j]))
                    lines_parallel = cos_dir > cos_thresh
                    if not lines_parallel:
                        continue

                    dist = np.linalg.norm(p1-p2)
                    # We check if the connection between the 2 endpoints has a similar direction as the first line (probably suboptimal, should test both)
                    connector = p1 - p2
                    connector_norm = np.linalg.norm(connector)
                    if connector_norm > 1e-9: #Check if the 2 points are different (necessary for colinearity test)
                        connector_unit = connector / connector_norm
                        cos_connector_i = abs(np.dot(connector_unit, directions[i]))
                        cos_connector_j = abs(np.dot(connector_unit, directions[j]))
                        is_colinear = (cos_connector_i > cos_thresh) and (cos_connector_j > cos_thresh)
                    else:
                        is_colinear = True  #The 2 points are the same point (colinearity test is usless and will not work)
                    is_aligned = lines_parallel and is_colinear
                    eff_threshold = aligned_threshold if is_aligned else threshold #Choose the threshold depending on whether the 2 lines are colinear or not
                    #If the 2 points are close enough a vertex is created in between the 2 lines and the 2 lines are saved as possibly connected to the vertex
                    if dist < eff_threshold:
                        vertex_id = len(vertex_candidates)
                        vertex_candidates.append(SoftVertexCandidate(p1 + (p2 - p1) / 2, vertex_id))
                        vertex_candidates[vertex_id].add_neighbour_line(line1,end1, dist/2)
                        vertex_candidates[vertex_id].add_neighbour_line(line2, end2, dist/2)

    #The connection between the vertices is added to the vertices
    vertex_candidates = find_vertex_net(vertex_candidates)

    return vertex_candidates


def lines_as_tracks(lines, plane): 
    #Returns a list of tracks made of just 1 line each
    tracks = []
    for i, line in enumerate(lines):
        tracks.append(SoftTrack(i, line, plane))
    return tracks

def vertex_line_dic(vertex_candidates):
    #Creates a dictionary that connects the tuple (line_id, end) to all vertices that are connected to that line end
    vertex_by_line_end = defaultdict(list)
    for vertex in vertex_candidates:
        for line_id, end, gap in vertex.attachments:
            vertex_by_line_end[(line_id, end)].append(vertex)
    return vertex_by_line_end

def new_line_layer_for_tracks(tracks, vertex_candidates, lines):
    #Takes a set of tracks made of n lines and returns all the possible tracks made of n+1 lines
    new_tracks = []
    #seen is a list that will contain [set of lines in a track, end point to which we connect lines] and is used to avoid saving the same track multiple times
    seen = set()
    #Get the dictionary that connects end points to vertices
    vertex_by_line_end = vertex_line_dic(vertex_candidates)
    for track in tracks:
        #We need to consider that when we create the first tracks (those with just one line) we don't have a preferred point of connection for the next lines
        if track.attachment_point[1] == "both":
            #The list of connected vertices will contain both sides of the line -> we can connect lines to both endpoints of the line
            connected_vertices = vertex_by_line_end.get((track.attachment_point[0], "in"),[]) + vertex_by_line_end.get((track.attachment_point[0], "f"),[])
        else:
            #Normal case: only on endpoint allows the track to be connected to a new line
            connected_vertices = vertex_by_line_end.get(track.attachment_point, [])
        #Iterate over all the vertices and over all the lines that are connected to those vertices    
        for connected_vertex in connected_vertices:
            for attachment in connected_vertex.attachments:
                #We want to exclude the line in the track itself (no use of connecting the line to itself)
                if attachment[0] in track.line_ids:
                    continue    
                #Create the new track
                new_track = track.copy(len(tracks)+len(new_tracks))
                new_end = "f" if attachment[1] == "in" else "in"
                new_track.add_line(lines[attachment[0]], attachment[2]*2, new_end)
                new_track.add_vertex(connected_vertex)
                #Don't save the track if a track with the same lines and endpoint is already in seen
                key = (frozenset(new_track.line_ids), tuple(new_track.attachment_point))
                #if [set(new_track.line_ids), new_track.attachment_point] in seen:
                if key in seen:
                    continue
                #Save the track in new_tracks and in seen
                new_tracks.append(new_track)
                #seen.append([set(new_track.line_ids), new_track.attachment_point])
                seen.add(key)
                if len(new_tracks) > parameters.track_candidates_track_number_threshold:
                    print(f"WARNING: track candidate overflow at {len(new_tracks)} tracks, stopping layer")
                    return []
    return new_tracks

def remove_duplicate_tracks(tracks):
    best = {}
    for track in tracks:
        key = frozenset(track.line_ids.tolist())
        if key not in best or sum(track.gaps) < sum(best[key].gaps):
            best[key] = track
    unique = list(best.values())
    for k, track in enumerate(unique):
        track.track_id = k
    return unique

def track_candidates(lines, vertex_candidates, plane):
    #Returns a list of tracks that contains all the possible tracks given the lines in lines and connection options in vertex_candidates
    #First create the list of tracks that are just made of 1 line
    tracks = lines_as_tracks(lines, plane)
    previous_layer_of_tracks = tracks
    #Increase the number of lines per track by repeatedly applying new_line_layer_for_tracks on the subset of tracks with the highest amount of lines per track.
    #The loop stops if all the lines are in a single track or if no longer track has been created
    for i in range(len(lines)):
        i+=1
        new_tracks = new_line_layer_for_tracks(previous_layer_of_tracks, vertex_candidates, lines)
        if new_tracks == []:
            break
        for new_track in new_tracks:
            tracks.append(new_track)
        previous_layer_of_tracks = new_tracks
    tracks = remove_duplicate_tracks(tracks)
    
    return tracks





class Find_candidate_tracks(ModuleBase):
    def __init__(self):
        super().__init__()
        self.requirements = ["in_and_f_z_y", "in_and_f_x_z", "line_index_zy", "line_index_xz", "total_charge_per_index_zy", "total_charge_per_index_xz"]

    def _setup_cli_options(self, parser):
        pass

    def _initialise(self) -> None:
        pass
    
    def _process(self, event:Event):
        for u_name, v_name in zip(["z", "x", "x"], ["y", "z", "y"]):
            in_and_f_name = "in_and_f_" + u_name + "_" + v_name
            vertices_name = "vertices_" + u_name + "_" + v_name
            tracks_name = "tracks_" + u_name + "_" + v_name
            total_charge_per_index_name = "total_charge_per_index_" + u_name + v_name
            keys = event.get_keys()
            if not in_and_f_name in keys or not total_charge_per_index_name in keys:
                continue
            in_and_f = event[in_and_f_name]
            total_charge_per_index = event[total_charge_per_index_name]
            lines = make_list_of_lines(in_and_f, total_charge_per_index)
            vertex_candidates = make_list_of_vertex_candidates(lines)
            plane = u_name + v_name
            tracks = track_candidates(lines, vertex_candidates, plane)
            event.add_data(vertices_name, vertex_candidates)
            event.add_data(tracks_name, tracks)
            

    def _help(self)->str:
        return "Creates a list of candidate tracks: all lines on their own, all pair of lines that might be connected, ..."

    def _finalise(self):
        log_module_name("Find_candidate_tracks", file_path="modules.txt")






def z_overlap(z_min_zy, z_max_zy, z_min_xz, z_max_xz):
    lo = max(z_min_zy, z_min_xz)
    hi = min(z_max_zy, z_max_xz)
    overlap = max(0.0, hi-lo)
    return overlap

def z_overlap_term(track_zy, track_xz):
    z_min_zy, z_max_zy = track_zy.z_range()
    z_min_xz, z_max_xz = track_xz.z_range()
    overlap = z_overlap(z_min_zy, z_max_zy, z_min_xz, z_max_xz)
    if overlap <= 0: 
        return None
    union = max(z_max_zy, z_max_xz)- min(z_min_zy, z_min_xz)
    return 1.0-overlap/union

def z_charge_density_term(track_zy, track_xz):
    density_zy = track_zy.charge_density()
    density_xz = track_xz.charge_density()    
    return abs(density_zy - density_xz)/max(density_zy,density_xz, 1e-12)

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
    

def makePairs(track_candidates_zy, track_candidates_xz):
    pairs = []
    for i, track_candidate_zy in enumerate(track_candidates_zy):
        for j, track_candidate_xz in enumerate(track_candidates_xz):
            overlap_cost = z_overlap_term(track_candidate_zy, track_candidate_xz)        # None if no overlap
            if overlap_cost is None or overlap_cost >= parameters.makePairs_threshold:
                continue
            pair_id = (i, j)
            charge_density_cost = z_charge_density_term(track_candidate_zy, track_candidate_xz)
            cost = parameters.makePairs_overlap_coefficient *overlap_cost + parameters.makePairs_density_coefficient * charge_density_cost
            quality = 1. - cost
            charge = (track_candidate_zy.total_charge + track_candidate_xz.total_charge)/2
            weight = charge * quality - parameters.makePairs_gap_coefficient * (sum(track_candidate_zy.gaps) + sum(track_candidate_xz.gaps)) - parameters.makePairs_pair_penalty
            pairs.append((i, j, weight, cost))
    return pairs

def selectPairs(track_candidates_zy, track_candidates_xz, pairs):
    if not pairs:
        return []
    rows, cols, row_of = [], [], {}
    for pair_id, (i, j, w, cost) in enumerate(pairs):
        for line_id in track_candidates_zy[i].line_ids:
            rows.append(row_of.setdefault(("zy", int(line_id)), len(row_of)))
            cols.append(pair_id)
        for line_id in track_candidates_xz[j].line_ids:
            rows.append(row_of.setdefault(("xz", int(line_id)), len(row_of)))
            cols.append(pair_id)
    pair_content_matrix = coo_matrix((np.ones(len(rows)), (rows,cols)), shape = (len(row_of), len(pairs)))

    res = milp(c=-np.array([w for _, _, w, _ in pairs]),          # milp minimises
               constraints=LinearConstraint(pair_content_matrix, -np.inf, 1),       # each line at most once
               integrality=np.ones(len(pairs)),
               bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(res.message)    
    return [pairs[p] for p in np.flatnonzero(res.x > 0.5)]


            



#TODO
class SoftThreeDMatching(ModuleBase):
    def __init__(self):
        super().__init__()
        self.requirements = ["tracks_z_y", "tracks_x_z"]

    def _setup_cli_options(self, parser):
        pass

    def _initialise(self) -> None:
        pass

    def _process(self, event:Event):
        print("##################### \n event_id = ", event["id"])

        tracks_z_y = event["tracks_z_y"]
        tracks_x_z = event["tracks_x_z"]
        pairs = makePairs(tracks_z_y, tracks_x_z)

        pairs = selectPairs(tracks_z_y, tracks_x_z, pairs)
        track_pairs = [[tracks_z_y[i], tracks_x_z[j], cost] for i, j, w, cost in pairs]

        for i, track_pair in enumerate(track_pairs):
            track_pair[0].track_id = i
            track_pair[0].connect_lines_in_track()
            track_pair[1].track_id = i
            track_pair[1].connect_lines_in_track()
  
        event.add_data("track_pairs", track_pairs)

        tracks = [track_pair[0] for track_pair in track_pairs] 
        tracks_name = "tracks_z_y"
        event.add_data(tracks_name, tracks)
        
        tracks = [track_pair[1] for track_pair in track_pairs]
        tracks_name = "tracks_x_z"
        event.add_data(tracks_name, tracks)

        in_and_f_z_y = event["in_and_f_z_y"]
        in_and_f_x_z = event["in_and_f_x_z"]
        for track_pair in track_pairs:
            for line in track_pair[0].lines:
                in_and_f_z_y[line.line_id, 0, :] = line.p_in
                in_and_f_z_y[line.line_id, 1, :] = line.p_f
            for line in track_pair[1].lines:
                in_and_f_x_z[line.line_id, 0, :] = line.p_in
                in_and_f_x_z[line.line_id, 1, :] = line.p_f
        event.add_data("in_and_f_z_y", in_and_f_z_y)
        event.add_data("in_and_f_x_z", in_and_f_x_z)
        
        



    def _help(self)->str:
        return "Matches vertices from the zy and xz planes"
    
    def _finalise(self):
        log_module_name("ThreeDMatching", file_path="modules.txt")


def number_of_hits_per_line(data, line_id, line_index):
    mask = line_index == line_id
    return len(data[mask])

def number_of_hits_per_track(data, track, line_index):
    number_of_hits = 0
    for line_id in track.line_ids:
        number_of_hits += number_of_hits_per_line(data, line_id, line_index)
    return number_of_hits


class Muon_Main_track_statistics(ModuleBase):
    def __init__(self):
        super().__init__()
        self.requirements = ["track_pairs", "x_fiber_hits", "y_fiber_hits", "min_charge_thresh", "line_index_zy", "line_index_xz"]
        self.min_charge_thresh = 0.

    def _setup_cli_options(self, parser):
        pass

    def _process(self, event:Event):
        self.min_charge_thresh = event["min_charge_thresh"]

        x_fiber_hits = event["x_fiber_hits"]
        y_fiber_hits = event["y_fiber_hits"]
        track_pairs = event["track_pairs"]
        track_lengths = [0,0]
        data = [0,0]
        total_number_of_hits = 0

        number_of_hits_for_longest_track = 0.

        for fiber_hits, u_name, v_name, axis_index in zip([x_fiber_hits, y_fiber_hits], ["z", "x"], ["y", "z"], [0, 1]):
            if not fiber_hits:
                continue
            data[axis_index] = data_2D(fiber_hits, u_name, v_name, self.min_charge_thresh)
            total_number_of_hits += len(data[axis_index])
            track_lengths[axis_index] = np.array([pair[axis_index].track_length() for pair in track_pairs])
        track_lengths = track_lengths[0] + track_lengths[1]
        if len(track_pairs) == 0:
            event.add_data("assigned_hit_ratio", 0)
            event.add_data("number_of_tracks", len(track_pairs))         
            return  
        longest_pair_index = np.argmax(track_lengths)
        for track, u_name, v_name, axis_index in zip(track_pairs[longest_pair_index], ["z", "x"], ["y", "z"], [0, 1]):
            line_index_name = "line_index_" + u_name + v_name
            line_index=np.array(event[line_index_name])
            number_of_hits_for_longest_track +=  number_of_hits_per_track(data[axis_index], track, line_index)
        if total_number_of_hits == 0:
            assigned_hit_ratio = -1
            event.add_data("assigned_hit_ratio", assigned_hit_ratio)
            return
        assigned_hit_ratio = number_of_hits_for_longest_track / total_number_of_hits

        event.add_data("assigned_hit_ratio", assigned_hit_ratio)
        event.add_data("number_of_tracks", len(track_pairs))


def get_all_turning_matches(turningpoints_zy, turningpoints_xz):
    pair_candidates = []
    for turningpoint_zy in turningpoints_zy:
        for turningpoint_xz in turningpoints_xz:
            if turningpoint_zy.extrema_type == turningpoint_xz.extrema_type:
                pair_candidates.append(turning_point_matches([(turningpoint_zy, turningpoint_xz)]))
    possible_pairings = list(pair_candidates)
    for number_of_pairs in range(1, min(len(turningpoints_zy), len(turningpoints_xz))):
        new_possible_pairings = []
        for possible_pairing in possible_pairings:
            for pair in pair_candidates:
                (pair_zy_id, ) = pair.indices_zy
                (pair_xz_id, ) = pair.indices_xz
                if (pair_zy_id > max(possible_pairing.indices_zy)) and (pair_xz_id > max(possible_pairing.indices_xz)) and (possible_pairing.get_number_of_pairs() == number_of_pairs):
                    new_possible_pairing = possible_pairing.copy()
                    new_possible_pairing = new_possible_pairing.add_match(pair.matched_list[0])
                    new_possible_pairings.append(new_possible_pairing)
        if len(new_possible_pairings) == 0:
            break
        possible_pairings.extend(new_possible_pairings)
    possible_pairings.append(turning_point_matches([]))
    return possible_pairings

    
def match_turning_points(turningpoints_zy, turningpoints_xz):
    all_turning_matches = get_all_turning_matches(turningpoints_zy, turningpoints_xz)
    best_match_index = np.argmin(np.array([turning_match.get_match_cost(len(turningpoints_zy), len(turningpoints_xz)) for turning_match in all_turning_matches]))
    return all_turning_matches[best_match_index]
"""    
def make_threeD_point(TwoD_point, subtrack):
    if subtrack.plane == "zy":
        point_plane_z_index = 1
        point_plane_not_z_index = 0
        track_plane_not_z_index = 1    
        ThreeD_second_point_coordinate = 0
        ThreeD_second_track_coordinate = 1
    elif subtrack.plane == "xz":
        point_plane_z_index = 0
        point_plane_not_z_index = 1
        track_plane_not_z_index = 0 
        ThreeD_second_point_coordinate = 1
        ThreeD_second_track_coordinate = 0
    else:
        raise ValueError(f"{subtrack.plane} is not a known plane")
    ThreeD_point = [0, 0, 0]       
    ThreeD_point[2] = TwoD_point[point_plane_z_index]
    ThreeD_point[ThreeD_second_point_coordinate] = TwoD_point[point_plane_not_z_index]
    Other_plane_TwoD_points = subtrack.get_point_in_function_of_z(TwoD_point[point_plane_z_index])
    if len(Other_plane_TwoD_points) == 0:
       return None
    Other_plane_TwoD_point = Other_plane_TwoD_points[0]
    ThreeD_point[ThreeD_second_track_coordinate] = Other_plane_TwoD_point[track_plane_not_z_index]
    return np.array(ThreeD_point) #.reshape(-1, 3)
"""
def make_threeD_node(TwoD_node, subtrack):
    if subtrack.plane == "zy":
        point_plane_z_index = 1
        point_plane_not_z_index = 0
        track_plane_not_z_index = 1    
        ThreeD_second_point_coordinate = 0
        ThreeD_second_track_coordinate = 1
    elif subtrack.plane == "xz":
        point_plane_z_index = 0
        point_plane_not_z_index = 1
        track_plane_not_z_index = 0 
        ThreeD_second_point_coordinate = 1
        ThreeD_second_track_coordinate = 0
    else:
        raise ValueError(f"{subtrack.plane} is not a known plane")
    TwoD_point = TwoD_node.position
    ThreeD_point = [0, 0, 0]       
    ThreeD_point[2] = TwoD_point[point_plane_z_index]
    ThreeD_point[ThreeD_second_point_coordinate] = TwoD_point[point_plane_not_z_index]
    Other_plane_TwoD_points = subtrack.get_point_in_function_of_z(TwoD_point[point_plane_z_index])
    if len(Other_plane_TwoD_points) == 0:
       return None
    Other_plane_TwoD_point = Other_plane_TwoD_points[0]
    ThreeD_point[ThreeD_second_track_coordinate] = Other_plane_TwoD_point[track_plane_not_z_index]
    return Node(np.array(ThreeD_point), TwoD_node.charge, TwoD_node.plane)

def change_direction(current_direction):
    if current_direction == "forward":
        return "backward"
    elif current_direction == "backward":
        return "forward"
    else:
        raise ValueError(f"{current_direction} is not a direction")

def is_turning_point_func(point, turning_points, tol=1e-9):
    for i, tp in enumerate(turning_points):
        if np.linalg.norm(point - tp.position) < tol:
            return i
    return None

def change_sign(current_sign):
    if current_sign > 0:
        return -1.
    else:
        return 1.

"""
def path_length(seq):
    return sum(np.linalg.norm(seq[i+1] - seq[i]) for i in range(len(seq) - 1))


def order_horizontal_runs(points, z_tol=parameters.z_range_fiber_dist, max_brute=8):
    pts = [np.asarray(p, dtype=float) for p in points]
    n = len(pts)
    result = []
    i = 0
    while i < n:
        # find the run of consecutive points sharing (almost) the same z
        j = i
        while j + 1 < n and abs(pts[j+1][2] - pts[i][2]) <= z_tol:
            j += 1
        run = pts[i:j+1]

        if len(run) > 1:
            prev = [result[-1]] if result else []   # already placed point before the run
            nxt  = [pts[j+1]] if j + 1 < n else []  # point after the run (different z, untouched)

            if len(run) <= max_brute:
                best = min(
                    permutations(run),
                    key=lambda perm: path_length(prev + list(perm) + nxt)
                )
                run = list(best)
            else:
                # fallback: greedy nearest neighbour starting from prev (or run[0])
                ref = prev[0] if prev else run[0]
                remaining = list(run)
                chain = []
                while remaining:
                    k = int(np.argmin([np.linalg.norm(p - ref) for p in remaining]))
                    ref = remaining.pop(k)
                    chain.append(ref)
                run = chain

        result.extend(run)
        i = j + 1
    return np.array(result)
"""
    

class Build_ThreeD_tracks(ModuleBase):
    def __init__(self):
        super().__init__()
        self.requirements = ["track_pairs"]

    def _setup_cli_options(self, parser):
        pass

    def _initialise(self) -> None:
        pass

    def _process(self, event:Event):
        track_pairs = event["track_pairs"]
        ThreeD_lines = []
        ThreeD_tracks = []
        for track_pair in track_pairs:
            track_z_y, track_x_z, _ = track_pair
            if track_z_y.monotonous()[0] and track_x_z.monotonous()[0]:
                points_z_y = track_z_y.nodes
                points_x_z = track_x_z.nodes
                ThreeD_nodes = create_threeD_points("zy", points_z_y, track_x_z) + create_threeD_points("xz", points_x_z, track_z_y)
                if len(ThreeD_nodes) < 2:
                    raise ValueError(f"ThreeD_points is too short: len(ThreeD_points) = {len(ThreeD_nodes)}")
                    continue
                ThreeD_nodes.sort(key=lambda n: n.position[2])
                ThreeD_points = [n.position for n in ThreeD_nodes]
                free_line_id = len(ThreeD_lines)
                free_track_id = len(ThreeD_tracks)
                ThreeD_line = Line.from_points(ThreeD_points[0], ThreeD_points[1], free_line_id)
                ThreeD_lines.append(ThreeD_line)
                ThreeD_track = ThreeDTrack(free_track_id, ThreeD_line)
                for i in range(2, len(ThreeD_points)):
                    free_line_id = len(ThreeD_lines)
                    ThreeD_track.add_point_to_track(ThreeD_points[i], free_line_id)
                    ThreeD_lines.append(ThreeD_track.lines[-1])
                ThreeD_track.nodes = ThreeD_nodes
                ThreeD_tracks.append(ThreeD_track)
            else:
                p_in_zy, p_f_zy = track_z_y.get_endpoints_of_track()
                p_in_xz, p_f_xz = track_x_z.get_endpoints_of_track()
                a = np.argmin(np.array([abs(p_in_zy[0]-p_in_xz[1]), abs(p_in_zy[0]-p_f_xz[1]), abs(p_f_zy[0]-p_in_xz[1]), abs(p_f_zy[0]-p_f_xz[1])]))
                
                if a == 1:
                    track_x_z = track_x_z.reverse_track()
                if a == 2:
                    track_z_y = track_z_y.reverse_track()
                if a == 3:
                    track_z_y = track_z_y.reverse_track()
                    track_x_z = track_x_z.reverse_track()

                signs_zy, turning_points_zy, subtracks_zy = track_z_y.split_into_monotonous_subtracks()
                signs_xz, turning_points_xz, subtracks_xz = track_x_z.split_into_monotonous_subtracks()
                if signs_zy[0] != signs_xz[0]:
                    track_z_y = track_z_y.reverse_track()
                    track_x_z = track_x_z.reverse_track()
                    signs_zy, turning_points_zy, subtracks_zy = track_z_y.split_into_monotonous_subtracks()
                    signs_xz, turning_points_xz, subtracks_xz = track_x_z.split_into_monotonous_subtracks()
                    if signs_zy[0] != signs_xz[0]:                     
                        print(f"\n weird edge case: signs_zy[0] = {signs_zy[0]} while signs_xz[0] = {signs_xz[0]}")
                        continue
                matched_turning_points = match_turning_points(turning_points_zy, turning_points_xz)
                nodes_z_y = track_z_y.nodes
                nodes_x_z = track_x_z.nodes
                if signs_zy[0] > 0:
                    if nodes_z_y[0].position[0] < nodes_x_z[0].position[1]:
                        current_zy_node_index = 0
                        current_xz_node_index = -1
                        current_plane = "zy"
                    else:
                        current_zy_node_index = -1
                        current_xz_node_index = 0
                        current_plane = "xz"
                else:
                    if nodes_z_y[0].position[0] < nodes_x_z[0].position[1]:
                        current_zy_node_index = -1
                        current_xz_node_index = 0
                        current_plane = "xz"
                    else:
                        current_zy_node_index = 0
                        current_xz_node_index = -1
                        current_plane = "zy"
                current_sign = signs_zy[0]
                current_direction_zy = "forward"
                current_direction_xz = "forward"
                current_zy_subtrack_index = 0
                current_xz_subtrack_index = 0
                ThreeD_nodes = []

                i = 0
                while i < 2 * (len(nodes_z_y)+len(nodes_x_z)):
                    i+=1
                    if current_plane == "zy":
                        current_twoD_node = nodes_z_y[current_zy_node_index]
                        current_plane_turning_points = turning_points_zy
                        if not 0 <= current_xz_subtrack_index < len(subtracks_xz):
                            break
                        ThreeD_node = make_threeD_node(current_twoD_node, subtracks_xz[current_xz_subtrack_index])
                        if ThreeD_node is not None:
                            ThreeD_nodes.append(ThreeD_node)
                    else:
                        current_twoD_node = nodes_x_z[current_xz_node_index]
                        current_plane_turning_points = turning_points_xz
                        if not 0 <= current_zy_subtrack_index < len(subtracks_zy):
                            break
                        ThreeD_node = make_threeD_node(current_twoD_node, subtracks_zy[current_zy_subtrack_index])
                        if ThreeD_node is not None:
                            ThreeD_nodes.append(ThreeD_node)

                    is_turning_point = is_turning_point_func(current_twoD_node.position, current_plane_turning_points)
                    if is_turning_point is not None:                    
                        if current_plane == "zy":
                            if is_turning_point not in matched_turning_points.indices_zy:
                                current_sign = change_sign(current_sign)
                                current_direction_xz = change_direction(current_direction_xz)
                                if current_direction_zy == "forward":
                                    current_zy_subtrack_index += 1
                                else:
                                    current_zy_subtrack_index -= 1

                            else:
                                other_plane_matched_turningpoint = matched_turning_points.get_matched_point(current_twoD_node.position, current_plane)
                                for j, other_plane_twoD_node in enumerate(nodes_x_z):
                                    if np.linalg.norm(other_plane_twoD_node.position-other_plane_matched_turningpoint.position) < 1e-9:
                                        current_xz_node_index = j
                                        break
                                current_sign = change_sign(current_sign)
                                current_plane = "xz"
                                if current_direction_zy == "forward":
                                    current_zy_subtrack_index += 1
                                else:
                                    current_zy_subtrack_index -= 1
                                if current_direction_xz == "forward":
                                    current_xz_subtrack_index += 1 
                                else:
                                    current_xz_subtrack_index -= 1

                        else:
                            if is_turning_point not in matched_turning_points.indices_xz:
                                current_sign = change_sign(current_sign)
                                current_direction_zy = change_direction(current_direction_zy)
                                if current_direction_xz == "forward":
                                    current_xz_subtrack_index += 1
                                else:
                                    current_xz_subtrack_index -= 1

                            else:
                                other_plane_matched_turningpoint = matched_turning_points.get_matched_point(current_twoD_node.position, current_plane)
                                for j, other_plane_twoD_node in enumerate(nodes_z_y):
                                    if np.linalg.norm(other_plane_twoD_node.position-other_plane_matched_turningpoint.position) < 1e-9:
                                        current_zy_node_index = j
                                        break
                                current_sign = change_sign(current_sign)
                                current_plane = "zy"
                                if current_direction_zy == "forward":
                                    current_zy_subtrack_index += 1
                                else:
                                    current_zy_subtrack_index -= 1
                                if current_direction_xz == "forward":
                                    current_xz_subtrack_index += 1 
                                else:
                                    current_xz_subtrack_index -= 1

                    if current_direction_zy == "forward":
                        next_zy_node_index = current_zy_node_index + 1
                    else:
                        next_zy_node_index = current_zy_node_index - 1
                    if current_direction_xz == "forward":
                        next_xz_node_index = current_xz_node_index + 1
                    else:
                        next_xz_node_index = current_xz_node_index - 1

                    if (0 <= next_zy_node_index < len(nodes_z_y)) and (0 <= next_xz_node_index < len(nodes_x_z)): 
                        next_node_zy = nodes_z_y[next_zy_node_index]
                        next_node_xz = nodes_x_z[next_xz_node_index]
                        if current_sign > 0:
                            if next_node_zy.position[0] < next_node_xz.position[1]:
                                current_plane = "zy"
                                current_zy_node_index = next_zy_node_index
                            else:
                                current_plane = "xz"
                                current_xz_node_index = next_xz_node_index
                        else:
                            if next_node_zy.position[0] > next_node_xz.position[1]:
                                current_plane = "zy"
                                current_zy_node_index = next_zy_node_index
                            else:
                                current_plane = "xz"
                                current_xz_node_index = next_xz_node_index
                    elif 0 <= next_zy_node_index < len(nodes_z_y):
                        current_plane = "zy"
                        current_zy_node_index = next_zy_node_index
                    elif 0 <= next_xz_node_index < len(nodes_x_z):
                        current_plane = "xz"
                        current_xz_node_index = next_xz_node_index
                    else:
                        break

                free_line_id = len(ThreeD_lines)
                free_track_id = len(ThreeD_tracks)
                if len(ThreeD_nodes)<2:
                    continue
                ThreeD_points = [n.position for n in ThreeD_nodes]
                #ThreeD_points = order_horizontal_runs(ThreeD_points) Needs to be fixed
                ThreeD_line = Line.from_points(ThreeD_points[0], ThreeD_points[1], free_line_id)
                ThreeD_lines.append(ThreeD_line)
                ThreeD_track = ThreeDTrack(free_track_id, ThreeD_line)
                for i in range(2, len(ThreeD_points)):
                    free_line_id = len(ThreeD_lines)
                    ThreeD_track.add_point_to_track(ThreeD_points[i], free_line_id)
                    ThreeD_lines.append(ThreeD_track.lines[-1])
                ThreeD_track.nodes = ThreeD_nodes
                ThreeD_tracks.append(ThreeD_track)                        
        event.add_data("ThreeD_lines", ThreeD_lines)
        event.add_data("ThreeD_tracks", ThreeD_tracks)
    def _help(self)->str:
        return "Creates 3D tracks from 2D track pairs"
    
    def _finalise(self):
        log_module_name("Build_ThreeD_tracks", file_path="modules.txt")                


def create_threeD_points(point_plane, TwoD_points, track):
    ThreeD_points = []
    if point_plane == "zy":
        point_plane_z_index = 0
        point_plane_not_z_index = 1
        track_plane_not_z_index = 0
        ThreeD_second_plane_coordinate = 1
        ThreeD_second_track_coordinate = 0
    elif point_plane == "xz":
        point_plane_z_index = 1
        point_plane_not_z_index = 0
        track_plane_not_z_index = 1
        ThreeD_second_plane_coordinate = 0
        ThreeD_second_track_coordinate = 1
    else:
        raise ValueError(f"{point_plane} is not a valid plane")
    for TwoD_point in TwoD_points:
        Other_plane_TwoD_points = track.get_point_in_function_of_z(TwoD_point.position[point_plane_z_index])
        for Other_plane_TwoD_point in Other_plane_TwoD_points:
            ThreeD_point = [0, 0, 0]
            ThreeD_point[2] = TwoD_point.position[point_plane_z_index]
            ThreeD_point[ThreeD_second_plane_coordinate] = TwoD_point.position[point_plane_not_z_index]
            ThreeD_point[ThreeD_second_track_coordinate] = Other_plane_TwoD_point[track_plane_not_z_index]
            ThreeD_points.append(Node(np.array(ThreeD_point), TwoD_point.charge / len(Other_plane_TwoD_points), point_plane))

    return ThreeD_points

def define_node_hit_region_t(data, line, line_index, average_node_hit_number = parameters.define_node_hit_regions_average_node_hit_number ):
    line_length = line.length()
    hit_density = number_of_hits_per_line(data, line.line_id, line_index) / line_length
    if hit_density < 1e-9:
        return [0., 1.]
    node_region_center_distance = average_node_hit_number / hit_density
    number_of_gaps = max(1, math.floor(line_length / node_region_center_distance))
    number_of_node_regions = number_of_gaps + 1
    t_values = np.linspace(0, 1, number_of_node_regions)
    return t_values

def assign_hits_to_node_region(hits, t_hits, t, t_gap):
    
    t_min = t - t_gap/2
    t_max = t + t_gap/2
    if t <= 1e-6:
        t_min = -np.inf
    if t >= 1-1e-6:
        t_max = np.inf
    hit_selection = []
    for i, t_hit in enumerate(t_hits):
        if t_min <= t_hit <= t_max:
            hit_selection.append(hits[i])
    return hit_selection

def create_track_node(hit_group, plane):
    hit_group = np.array(hit_group)
    if hit_group.size == 0:
        return None
    return Node(np.average(hit_group[:, :2], axis = 0, weights = hit_group[:, 2]), np.sum(hit_group[:, 2]), plane)

def preaverage_hits(hits, plane, direction, z_tol = .3): #parameters.z_range_fiber_dist
    hits = np.asarray(hits, dtype = float)
    if hits.size == 0:
        return hits
    not_z_idx = 1 if plane == "zy" else 0
    z_idx = 0 if plane == "zy" else 1
    idx = z_idx if direction=="z" else not_z_idx
    order = np.argsort(hits[:, idx])
    hits = hits[order]

    groups = []
    current_group = [hits[0]]
    for hit in hits[1:]:
        if abs(hit[idx]-current_group[-1][idx]<= z_tol):
            current_group.append(hit)
        else:
            groups.append(current_group)
            current_group = [hit]
    groups.append(current_group)
    averaged_hits = []
    #if len(groups) == 1:
    #    return hits[:, :3]
    for group in groups:
        group = np.array(group)
        number_of_hits_in_group = len(group)
        total_charge = np.sum(group[:, 2])
        position = np.average(group[:, :2], axis = 0, weights=group[:, 2])
        for k in range(number_of_hits_in_group):
            averaged_hits.append([position[0], position[1], total_charge/number_of_hits_in_group])

    return np.array(averaged_hits)


    

def create_track_nodes(data, track, line_index):
    nodes = []
    additional_points_of_connection_point = [] #the connection point of a line needs to consider both hit lists: of the 2 connected lines
    number_of_lines_in_track = len(track)
    track_point_sequence = track.get_point_sequence()
    for i in range(number_of_lines_in_track):
        line = track.lines[i]
        hits = data[line_index == line.line_id]
        hits = preaverage_hits(hits, track.plane, "z")
        hits = preaverage_hits(hits, track.plane, "other_coordinate")
        p_in = track_point_sequence[i]
        p_f = track_point_sequence[i+1]
        direction = p_f - p_in
        direction_length = direction @ direction
        if direction_length < 1e-9:
            continue
        t_hits = (hits[:, :2] - p_in) @ direction / (direction_length)
        t_values = define_node_hit_region_t(data, line, line_index)
        t_gap = t_values[1] - t_values[0]
        hit_group_list = []
        for t in t_values:
            hit_group_list.append(assign_hits_to_node_region(hits, t_hits, t, t_gap))
        hit_group_list[0].extend(additional_points_of_connection_point)
        additional_points_of_connection_point = hit_group_list[-1]
        for hit_group in hit_group_list[:-1]:
            node = create_track_node(hit_group, track.plane)
            if node is not None:
                nodes.append(node)
        if i == number_of_lines_in_track-1:
            node = create_track_node(hit_group_list[-1], track.plane)
            if node is not None:
                nodes.append(node)
    return nodes
        
def make_track_from_nodes(Nodes, Old_track, line_id, vertex_id):
    if len(Nodes) < 2:
        return None, line_id, vertex_id
    first_line = Line.from_points(Nodes[0].position, Nodes[1].position, line_id)
    line_id += 1
    track = SoftTrack(Old_track.track_id, first_line, Old_track.plane)
    for i in range(2, len(Nodes)):
        line = Line.from_points(Nodes[i-1].position, Nodes[i].position, line_id)
        line_id += 1
        track.add_line(line, 0, "f")
        vertex = SoftVertexCandidate(np.array(Nodes[i-1].position, dtype=float), vertex_id, charge = Nodes[i-1].charge)
        vertex_id += 1
        vertex.add_neighbour_line(track.lines[i - 2], "f", 0.)
        vertex.add_neighbour_line(track.lines[i-1], "in", 0.)
        track.add_vertex(vertex)
    track.nodes = list(Nodes)
    track.total_charge = sum(node.charge for node in Nodes)
    return track, line_id, vertex_id

def store_pairs_in_event(event, track_pairs):
    in_and_f = {"zy": [], "xz": []}
    
    for k, pair in enumerate(track_pairs):
        for track, plane in ((pair[0], "zy"), (pair[1], "xz")):
            track.track_id = k
            for line in track.lines:
                in_and_f[plane].append([line.p_in, line.p_f])
            track.line_ids = np.array([line.line_id for line in track.lines])

    event.add_data("in_and_f_z_y", np.array(in_and_f["zy"]).reshape(-1, 2, 2))
    event.add_data("in_and_f_x_z", np.array(in_and_f["xz"]).reshape(-1, 2, 2))
    event.add_data("tracks_z_y", [pair[0] for pair in track_pairs])
    event.add_data("tracks_x_z", [pair[1] for pair in track_pairs])
    event.add_data("track_pairs", track_pairs)

class Add_internal_nodes(ModuleBase):
    def __init__(self):
        super().__init__()
        self.requirements = ["track_pairs", "x_fiber_hits", "y_fiber_hits", "min_charge_thresh", "line_index_zy", "line_index_xz"]

    def _setup_cli_options(self, parser):
        pass

    def _initialise(self) -> None:
        pass

    def _process(self, event:Event):
        track_pairs = event["track_pairs"]
        x_fiber_hits = event["x_fiber_hits"]
        y_fiber_hits = event["y_fiber_hits"]
        min_charge_thresh = event["min_charge_thresh"]
        line_index_zy = np.asarray(event["line_index_zy"])
        line_index_xz = np.asarray(event["line_index_xz"])
        New_track_pairs = []
        line_id_list = [0, 0]
        vertex_id_list = [0, 0]


        for track_pair in track_pairs:
            track_zy, track_xz, cost = track_pair
            New_track_pair = [None, None, cost]
            line_id_temp = list(line_id_list)
            vertex_id_temp = list(vertex_id_list)
            for fiber_hits, u_name, v_name, line_index, track, pair_index in zip([x_fiber_hits, y_fiber_hits], ["z", "x"], ["y", "z"], [line_index_zy, line_index_xz], [track_zy, track_xz], [0, 1]):
                line_id = line_id_list[pair_index]
                vertex_id = vertex_id_list[pair_index]
                data = data_2D(fiber_hits, u_name, v_name, min_charge_thresh)
                if len(data) == 0:
                    return
                Nodes = create_track_nodes(data, track, line_index)
                New_track_pair[pair_index], line_id, vertex_id = make_track_from_nodes(Nodes, track, line_id, vertex_id)
                line_id_temp[pair_index] = line_id
                vertex_id_temp[pair_index] = vertex_id
            if (New_track_pair[0] is not None) and (New_track_pair[1] is not None):
                New_track_pairs.append(New_track_pair)
                line_id_list = list(line_id_temp)
                vertex_id_list = list(vertex_id_temp)
        #event.add_data("track_pairs", New_track_pairs)

        store_pairs_in_event(event, New_track_pairs)

                


    def _help(self)->str:
        return "Adds internal nodes to each line of the chosen tracks that are defined by the mean positions of hit groups"
    
    def _finalise(self):
        log_module_name("Add_internal_nodes", file_path="modules.txt")                
