#reconstruction.py

#   HoughTransform2D
#HoughTransform2D_DBSCAN_epsilon = 2.5
HoughTransform2D_DBSCAN_epsilon = 40.0
#HoughTransform2D_TimeSlice_epsilon = 6.
HoughTransform2D_neighbour_dist = 1.5
#HoughTransform2D_min_points_per_line = 5
HoughTransform2D_min_points_per_line = 100
HoughTransform2D_lattice_step_size = 1. 


#   RANSAC2D
RANSAC2D_DBSCAN_epsilon = 10.
RANSAC2D_n_repeats = 1 #number of times the multiline ransac is run to check for the best outcome
RANSAC2D_line_penalty = 3 #how much the multiple line algorithm is penalized by the number of final lines
RANSAC2D_min_inliers = 4 #the smallest number of hits before the loop in fit_lines_ransac is left (this is the loop that keeps looking for new lines applying ransac_single_line on the remaining hits)
RANSAC2D_max_lines = 10 #the maximal number of lines the algorithm can find within 1 cluster
RANSAC2D_min_samples = 2 #in ransac from skimage: The minimum number of data points to fit a model to
RANSAC2D_residual_threshold = 1.5 #In class Line2D: distance threshold for which hit are assigned to a candidate line in the ransac
RANSAC2D_max_trials = 1000

#Track_finding.py

#   CutLines
#CutLines_line_hit_threshold=4.
CutLines_line_hit_threshold=15. #distance between a point on the line and the closest hit at which the line is cut
#CutLines_line_hit_threshold= 5. #10.0 #distance between a point on the line and the closest hit at which the line is cut

#   Refit_lines
Refit_lines_neighbour_dist = 1.5
Refit_lines_min_points_per_line = 4
Refit_lines_lattice_step_size = 1.


#   Ransac_refit_lines
refit_n_repeats = 1 #number of times the multiline ransac is run to check for the best outcome
refit_line_penalty = 3 #how much the multiple line algorithm is penalized by the number of final lines
refit_min_inliers = 4 #the smallest number of hits before the loop in fit_lines_ransac is left (this is the loop that keeps looking for new lines applying ransac_single_line on the remaining hits)
refit_max_lines = 10 #the maximal number of lines the algorithm can find within 1 cluster
refit_min_samples = 2 #in ransac from skimage: The minimum number of data points to fit a model to
refit_residual_threshold = 1.5 #In class Line2D: distance threshold for which hit are assigned to a candidate line in the ransac
refit_max_trials = 1000

#   Find_simple_tracks
Find_simple_tracks_threshold = 2. #distance at which line endpoints are merged
Find_simple_tracks_aligned_threshold = 15. #distance at whihc line endpoints are merged if the two lines are considered aligned
Find_simple_tracks_angle_thresh_deg = 15. #angle difference for which the lines can still be considered aligned

insert_vertices_threshold = 2.


#   Unasigned_refit_lines
Unasigned_refit_lines_neighbour_dist = 1.5
#Unasigned_refit_lines_TimeSlice_epsilon = 6.
Unasigned_refit_lines_min_points_per_line = 3
Unasigned_refit_lines_lattice_step_size = 1.
Unasigned_refit_lines_DBSCAN_epsilon = 3.
Unasigned_refit_lines_min_points_per_cluster = 5

#   Unasigned_RANSAC_refit_lines
Unasigned_RANSAC_refit_lines_DBSCAN_epsilon = 2.5
Unasigned_RANSAC_refit_lines_min_points_per_cluster = 5
Unasigned_RANSAC_refit_lines_n_repeats = 1
Unasigned_RANSAC_refit_lines_line_penalty = 3
Unasigned_RANSAC_refit_lines_min_inliers = 10
Unasigned_RANSAC_refit_lines_max_lines = 10
Unasigned_RANSAC_refit_lines_min_samples = 2
Unasigned_RANSAC_refit_lines_residual_threshold = 1.5
Unasigned_RANSAC_refit_lines_max_trials = 1000






#Hit_assigning

#   AssignHits
AssignHits_threshold = 1. #Max distance at which a hit is still assigned to a line

#ThreeD_Matching

#   ThreeDMatching
ThreeDMatching_z_vertex_threshold = 5.0
ThreeDMatching_track_overlap_threshold = .7

#Soft_ThreeD_Matching

#   track_candidates
track_candidates_track_number_threshold = 100

#   make_list_of_vertex_candidates
make_list_of_vertex_candidates_threshold = 4. #distance at whihc track endpoints are considered close enough for a vertex candidate
make_list_of_vertex_candidates_aligned_threshold = 15. #distance at which track endpoints are considered close enough for a vertex candidate if the lines are considered colinear
make_list_of_vertex_candidates_angle_thresh_deg = 35. #allowed difference in angle for the lines to be considered colinear
check_line_overlap_angle_threshold =15. #allowed difference in angle that allows the lines to be considered overlapped
check_line_overlap_width_threshold = 1.

#   makePairs
makePairs_threshold = .7
makePairs_overlap_coefficient = .7
makePairs_density_coefficient = 0.3
makePairs_gap_coefficient = 0.
makePairs_pair_penalty = 1.

#module_base
number_of_events = 200

#tracks
z_range_fiber_dist = .55
#   turning_point_matches
turning_point_matches_sigma_z = 5
turning_point_matches_unmatched_penalty = .5

#ModuleBase
ModuleBase_SelectedEvents = []

#Add_internal_nodes
define_node_hit_regions_average_node_hit_number = 4.
min_hits_per_node = 4



