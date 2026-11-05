import numpy as np
#from matplotlib import pyplot as plt
from liquidreco.event import Event
#import matplotlib
from liquidreco.modules.module_base import ModuleBase
import ROOT
from ROOT import std
from array import array
import json
import ROOT
from liquidreco import parameters
from  liquidreco.Tracks import SimpleTrack, Line, VertexCandidate

from liquidreco.save_used_modules import log_module_name

def save_params():
    cfg = {}
    for name, value in vars(parameters).items():
        if name.startswith("_"):
            continue
        if isinstance(value, (int, float, str, bool, list, tuple, dict)):
            cfg[name] = value
    text = json.dumps(cfg)
    return text



def fill_vector_from_array(vec, arr, cast=int):
    vec.clear()
    for val in arr:
        vec.push_back(cast(val))
    return vec

def data_2D(fiber_hits, u_name, v_name, min_charge_thresh): #takes the fiber hits and returns position np.array. e.g. x_fiber_hits -> zy positions for plots
    
    return np.array([[getattr(hit, u_name), getattr(hit, v_name), hit.weight, hit.time] for hit in fiber_hits if hit.weight> min_charge_thresh]) 



class MakeTree(ModuleBase):
    """
        -saves initial and final points of 2D lines
        -saves the assignment of hits
    """

    def __init__(self):
        super().__init__()
        self.requirements = ["x_fiber_hits", "y_fiber_hits", "z_fiber_hits","in_and_f_z_y", "in_and_f_x_z", "in_and_f_x_y","min_charge_thresh" ]

    def _setup_cli_options(self, parser):
        
        parser.add_argument(
            "--file_name", 
            help="Name of the saved root file", 
            required = False, default = "Hough2D-results", type = str,
        )

    def _initialise(self)-> None:
          
        self.file_name = self.args.file_name

        complete_file_name = self.file_name + ".root"
        self.file = ROOT.TFile(complete_file_name, "RECREATE")
        self.tree = ROOT.TTree("RecoTree", "initial and final points of the Hough transform line")

        #zy plot lines
        self.zyplot_in_zpos = std.vector('float')()
        self.zyplot_in_ypos = std.vector('float')()
        self.tree.Branch("zyplot_in_zpos", self.zyplot_in_zpos)
        self.tree.Branch("zyplot_in_ypos", self.zyplot_in_ypos)

        self.zyplot_f_zpos = std.vector('float')()
        self.zyplot_f_ypos = std.vector('float')()
        self.tree.Branch("zyplot_f_zpos", self.zyplot_f_zpos)
        self.tree.Branch("zyplot_f_ypos", self.zyplot_f_ypos)

        #xz plot lines
        self.xzplot_in_xpos = std.vector('float')()
        self.xzplot_in_zpos = std.vector('float')()
        self.tree.Branch("xzplot_in_xpos", self.xzplot_in_xpos)
        self.tree.Branch("xzplot_in_zpos", self.xzplot_in_zpos)

        self.xzplot_f_xpos = std.vector('float')()
        self.xzplot_f_zpos = std.vector('float')()
        self.tree.Branch("xzplot_f_xpos", self.xzplot_f_xpos)
        self.tree.Branch("xzplot_f_zpos", self.xzplot_f_zpos)

        #xy plot lines
        self.xyplot_in_xpos = std.vector('float')()
        self.xyplot_in_ypos = std.vector('float')()
        self.tree.Branch("xyplot_in_xpos", self.xyplot_in_xpos)
        self.tree.Branch("xyplot_in_ypos", self.xyplot_in_ypos)

        self.xyplot_f_xpos = std.vector('float')()
        self.xyplot_f_ypos = std.vector('float')()
        self.tree.Branch("xyplot_f_xpos", self.xyplot_f_xpos)
        self.tree.Branch("xyplot_f_ypos", self.xyplot_f_ypos)

        self.zy_hits_z = std.vector('float')()
        self.zy_hits_y = std.vector('float')()
        self.zy_hits_time = std.vector('float')()
        self.xz_hits_x = std.vector('float')()
        self.xz_hits_z = std.vector('float')()
        self.xz_hits_time = std.vector('float')()
        self.xy_hits_x = std.vector('float')()
        self.xy_hits_y = std.vector('float')()
        self.xy_hits_time = std.vector('float')()


        self.tree.Branch("zy_hits_z", self.zy_hits_z)
        self.tree.Branch("zy_hits_y", self.zy_hits_y)
        self.tree.Branch("zy_hits_time", self.zy_hits_time)
        self.tree.Branch("xz_hits_x", self.xz_hits_x)
        self.tree.Branch("xz_hits_z", self.xz_hits_z)
        self.tree.Branch("xz_hits_time", self.xz_hits_time)
        self.tree.Branch("xy_hits_x", self.xy_hits_x)
        self.tree.Branch("xy_hits_y", self.xy_hits_y)
        self.tree.Branch("xy_hits_time", self.xy_hits_time)

        self.line_index_zy = std.vector('int')() #will share the same index as the hits, and each entry tells which line on the zy plane the corresponding hit has been assigned to
        self.tree.Branch("line_index_zy", self.line_index_zy)
        self.line_index_xz = std.vector('int')() 
        self.tree.Branch("line_index_xz", self.line_index_xz)
        self.line_index_xy = std.vector('int')() 
        self.tree.Branch("line_index_xy", self.line_index_xy)

        self.track_number_zy = np.zeros(1, dtype=np.int32)
        self.tree.Branch("track_number_zy", self.track_number_zy, "track_number_zy/I")
        self.track_number_xz = np.zeros(1, dtype=np.int32)
        self.tree.Branch("track_number_xz", self.track_number_xz, "track_number_xz/I")
        self.track_number_xy = np.zeros(1, dtype=np.int32)
        self.tree.Branch("track_number_xy", self.track_number_xy, "track_number_xy/I")

        self.assigned_hit_ratio = np.zeros(1, dtype = np.float32)
        self.number_of_tracks = np.zeros(1, dtype = np.int32)
        self.tree.Branch("assigned_hit_ratio", self.assigned_hit_ratio, "assigned_hit_ratio/F")
        self.tree.Branch("number_of_tracks", self.number_of_tracks, "number_of_tracks/I")


        self.track_line_ids_zy = std.vector('vector<int>')()
        self.tree.Branch("track_line_ids_zy", self.track_line_ids_zy)
        self.track_line_ids_xz = std.vector('vector<int>')()
        self.tree.Branch("track_line_ids_xz", self.track_line_ids_xz)
        self.track_line_ids_xy = std.vector('vector<int>')()
        self.tree.Branch("track_line_ids_xy", self.track_line_ids_xy)

        self.zy_vertex_positions = std.vector('vector<float>')()
        self.tree.Branch("zy_vertex_positions", self.zy_vertex_positions)
        self.xz_vertex_positions = std.vector('vector<float>')()
        self.tree.Branch("xz_vertex_positions", self.xz_vertex_positions)

        self.vertex_pairs_zy_position = std.vector('vector<float>')()
        self.vertex_pairs_xz_position = std.vector('vector<float>')()
        self.tree.Branch("vertex_pairs_zy_position", self.vertex_pairs_zy_position)
        self.tree.Branch("vertex_pairs_xz_position", self.vertex_pairs_xz_position)

        self.track_pair_zy_ids = std.vector('int')()
        self.tree.Branch("track_pair_zy_ids", self.track_pair_zy_ids)
        self.track_pair_xz_ids = std.vector('int')()
        self.tree.Branch("track_pair_xz_ids", self.track_pair_xz_ids)

        self.event_id = np.zeros(1,dtype=np.int32)
        self.tree.Branch("event_id", self.event_id, "event_id/I")

        self.track_points_x = std.vector('vector<float>')() #first index track_id, second index point in the track
        self.track_points_y = std.vector('vector<float>')()
        self.track_points_z = std.vector('vector<float>')()
        self.track_ids = std.vector('int')()
        self.tree.Branch("track_points_x", self.track_points_x)
        self.tree.Branch("track_points_y", self.track_points_y)
        self.tree.Branch("track_points_z", self.track_points_z)
        self.tree.Branch("track_ids", self.track_ids)



    def _process(self, event:Event):
        self.event_id[0] = event["id"]
        print(event["id"])
        self.min_charge_thresh=event["min_charge_thresh"]

        in_and_f_z_y=event["in_and_f_z_y"]
        in_and_f_x_z=event["in_and_f_x_z"]
        in_and_f_x_y=event["in_and_f_x_y"]

        x_fiber_hits = event["x_fiber_hits"]
        y_fiber_hits = event["y_fiber_hits"]
        z_fiber_hits = event["z_fiber_hits"]

        #zy plot
        self.zyplot_in_zpos.clear()
        self.zyplot_in_ypos.clear()

        self.zyplot_f_zpos.clear()
        self.zyplot_f_ypos.clear()

        #xz plot
        self.xzplot_in_xpos.clear()
        self.xzplot_in_zpos.clear()

        self.xzplot_f_xpos.clear()
        self.xzplot_f_zpos.clear()

        #xy plot
        self.xyplot_in_xpos.clear()
        self.xyplot_in_ypos.clear()

        self.xyplot_f_xpos.clear()
        self.xyplot_f_ypos.clear()

        self.line_index_zy.clear()
        self.line_index_xz.clear()
        self.line_index_xy.clear()       

        self.track_line_ids_zy.clear()
        self.track_line_ids_xz.clear()
        self.track_line_ids_xy.clear()

        self.zy_vertex_positions.clear()
        self.xz_vertex_positions.clear()

        self.vertex_pairs_zy_position.clear()
        self.vertex_pairs_xz_position.clear()

        self.track_pair_zy_ids.clear()
        self.track_pair_xz_ids.clear()

        self.track_points_x.clear()
        self.track_points_y.clear()
        self.track_points_z.clear()
        self.track_ids.clear()


        #zy plot
        if x_fiber_hits:
            line_index_zy = event["line_index_zy"]
            for track in in_and_f_z_y:
                self.zyplot_in_zpos.push_back(float(track[0,0]))
                self.zyplot_in_ypos.push_back(float(track[0,1]))

                self.zyplot_f_zpos.push_back(float(track[1,0]))
                self.zyplot_f_ypos.push_back(float(track[1,1]))

            zy_data = data_2D(x_fiber_hits, "z", "y", self.min_charge_thresh)
            self.zy_hits_z = fill_vector_from_array(self.zy_hits_z, zy_data[:, 0], cast=float)
            self.zy_hits_y = fill_vector_from_array(self.zy_hits_y, zy_data[:, 1], cast=float)
            self.zy_hits_time = fill_vector_from_array(self.zy_hits_time, zy_data[:, 3], cast=float)
            self.line_index_zy = fill_vector_from_array(self.line_index_zy, line_index_zy)
            if "tracks_z_y" in event._data:
                tracks_zy = event["tracks_z_y"]
                self.track_number_zy[0] = len(tracks_zy)
                for track in tracks_zy:
                    inner = std.vector('int')()
                    inner = fill_vector_from_array(inner, track.line_ids)
                    self.track_line_ids_zy.push_back(inner)


        #xz plot
        if y_fiber_hits:
            line_index_xz = event["line_index_xz"]
            for track in in_and_f_x_z:
                self.xzplot_in_xpos.push_back(float(track[0,0]))
                self.xzplot_in_zpos.push_back(float(track[0,1]))

                self.xzplot_f_xpos.push_back(float(track[1,0]))
                self.xzplot_f_zpos.push_back(float(track[1,1]))

            xz_data = data_2D(y_fiber_hits, "x", "z", self.min_charge_thresh)
            self.xz_hits_x = fill_vector_from_array(self.xz_hits_x, xz_data[:, 0], cast=float)
            self.xz_hits_z = fill_vector_from_array(self.xz_hits_z, xz_data[:, 1], cast=float)
            self.xz_hits_time = fill_vector_from_array(self.xz_hits_time, xz_data[:, 3], cast=float)
            self.line_index_xz = fill_vector_from_array(self.line_index_xz, line_index_xz)
            if "tracks_x_z" in event._data:
                tracks_xz = event["tracks_x_z"]
                self.track_number_xz[0] = len(tracks_xz)
                for track in tracks_xz:
                    inner = std.vector('int')()
                    inner = fill_vector_from_array(inner, track.line_ids)
                    self.track_line_ids_xz.push_back(inner)
                    

        #xy plot
        if z_fiber_hits:
            line_index_xy = event["line_index_xy"]
            for track in in_and_f_x_y:
                self.xyplot_in_xpos.push_back(float(track[0,0]))
                self.xyplot_in_ypos.push_back(float(track[0,1]))

                self.xyplot_f_xpos.push_back(float(track[1,0]))
                self.xyplot_f_ypos.push_back(float(track[1,1]))

            xy_data = data_2D(z_fiber_hits, "x", "y", self.min_charge_thresh)
            self.xy_hits_x = fill_vector_from_array(self.xy_hits_x, xy_data[:, 0], cast=float)
            self.xy_hits_y = fill_vector_from_array(self.xy_hits_y, xy_data[:, 1], cast=float)
            self.xy_hits_time = fill_vector_from_array(self.xy_hits_time, xy_data[:, 3], cast=float)
            self.line_index_xy = fill_vector_from_array(self.line_index_xy, line_index_xy)
            if "tracks_x_y" in event._data:
                tracks_xy = event["tracks_x_y"]
                self.track_number_xy[0] = len(tracks_xy)
                for track in tracks_xy:
                    inner = std.vector('int')()
                    inner = fill_vector_from_array(inner, track.line_ids)
                    self.track_line_ids_xy.push_back(inner)

        if "vertices_z_y" in event._data:
            vertices_z_y = event["vertices_z_y"]
            zy_vertex_positions = [vertex.position for vertex in vertices_z_y]
            for zy_vertex_position in zy_vertex_positions:
                inner = std.vector('float')()
                self.zy_vertex_positions.push_back(fill_vector_from_array(inner, zy_vertex_position, cast=float))
            

        if "vertices_x_z" in event._data:
            vertices_x_z = event["vertices_x_z"]
            xz_vertex_positions = [vertex.position for vertex in vertices_x_z]
            for xz_vertex_position in xz_vertex_positions:
                inner = std.vector('float')()
                self.xz_vertex_positions.push_back(fill_vector_from_array(inner, xz_vertex_position, cast=float))

        if "vertex_pairs" in event._data:
            vertex_pairs = event["vertex_pairs"]

            zy_vertices = [pair[0] for pair in vertex_pairs]
            xz_vertices = [pair[1] for pair in vertex_pairs]

            vertex_pairs_zy_position = [vertex.position for vertex in zy_vertices]
            vertex_pairs_xz_position = [vertex.position for vertex in xz_vertices]

            for zy_position in vertex_pairs_zy_position:
                inner = std.vector('float')()
                self.vertex_pairs_zy_position.push_back(fill_vector_from_array(inner, zy_position, cast=float))    

            for xz_position in vertex_pairs_xz_position:
                inner = std.vector('float')()
                self.vertex_pairs_xz_position.push_back(fill_vector_from_array(inner, xz_position, cast=float))  

        if "track_pairs" in event._data:
            track_pairs = event["track_pairs"]  
            track_pair_zy_ids = [pair[0].track_id for pair in track_pairs]
            track_pair_xz_ids = [pair[1].track_id for pair in track_pairs]
            self.track_pair_zy_ids = fill_vector_from_array(self.track_pair_zy_ids, track_pair_zy_ids)
            self.track_pair_xz_ids = fill_vector_from_array(self.track_pair_xz_ids, track_pair_xz_ids)
        if "assigned_hit_ratio" in event._data:
            self.assigned_hit_ratio[0] = event["assigned_hit_ratio"]
            self.number_of_tracks[0] = event["number_of_tracks"]

        if "ThreeD_tracks" in event._data:
            ThreeD_tracks = event["ThreeD_tracks"]
            self.track_ids = fill_vector_from_array(self.track_ids, [ThreeD_track.track_id for ThreeD_track in ThreeD_tracks], cast=int)
            ThreeD_tracks_points = [ThreeD_track.points for ThreeD_track in ThreeD_tracks]
            for ThreeD_track_points in ThreeD_tracks_points:
                ThreeD_track_points = np.array(ThreeD_track_points)
                ThreeD_track_points_x = ThreeD_track_points[:, 0]
                ThreeD_track_points_y = ThreeD_track_points[:, 1]
                ThreeD_track_points_z = ThreeD_track_points[:, 2]
                inner = std.vector('float')()
                self.track_points_x.push_back(fill_vector_from_array(inner, ThreeD_track_points_x, cast=float))
                inner = std.vector('float')()
                self.track_points_y.push_back(fill_vector_from_array(inner, ThreeD_track_points_y, cast=float))
                inner = std.vector('float')()
                self.track_points_z.push_back(fill_vector_from_array(inner, ThreeD_track_points_z, cast=float))



        self.tree.Fill()


    def _help(self)->str:
        return "Generates a ROOT Tree where the following information is saved for the 3 combinations zy, xz, and xy: \n - initial and final points of the Hough fitted lines\n - the hits\n - the line index which is a vector that contains the index of the closest track \n\n!!! Has never been tested for xy!!!"


    def _finalise(self):
        self.file.Write()

        text = save_params()
        ROOT.TNamed("config", text).Write()
        
        self.file.Close()

        log_module_name("Maketree", file_path="modules.txt")


























