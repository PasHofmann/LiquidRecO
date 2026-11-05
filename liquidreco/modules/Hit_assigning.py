import numpy as np
#from matplotlib import pyplot as plt
from liquidreco.event import Event
#import matplotlib
from liquidreco.modules.module_base import ModuleBase
import ROOT
from ROOT import std
from array import array
from liquidreco import parameters

from liquidreco.save_used_modules import log_module_name


def fill_vector_from_array(vec, arr, cast=int):
    vec.clear()
    for val in arr:
        vec.push_back(cast(val))
    return vec

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

def data_2D(fiber_hits, u_name, v_name, min_charge_thresh): #takes the fiber hits and returns position np.array. e.g. x_fiber_hits -> zy positions for plots
    
    return np.array([[getattr(hit, u_name), getattr(hit, v_name), hit.weight] for hit in fiber_hits if hit.weight> min_charge_thresh]) 

def assign_line(data, x_in_array, y_in_array, x_f_array, y_f_array):
    #returns an array that contains the index of the closest line for each hit in data
    hit_number = len(data[:,0])
    track_number = len(x_in_array)

    if track_number == 0:
        return np.full(hit_number,-1)
    distances = np.empty((hit_number, track_number))
    for i in range(track_number):
        distances[:, i] = distance(data, x_in_array[i], y_in_array[i], x_f_array[i], y_f_array[i])
    line_index = np.argmin(distances, axis=1)
    min_distance = np.min(distances, axis=1)
    line_index[min_distance > parameters.AssignHits_threshold] = -1
    return line_index

def total_charge_per_index_func(line_index, data, n_lines):
    charges = np.zeros((n_lines,2))
    line_index = np.array(line_index)
    hit_charges = data[:, 2]
    ids = np.unique(line_index)
    ids = ids[ids>=0]
    for i in range(n_lines):
        charges[i,0] = i
    for i in ids:
        charges[i, 1] = hit_charges[line_index==i].sum()
    #return np.array([[i, hit_charges[line_index==i].sum()] for i in ids])
    return charges


class AssignHits(ModuleBase):
    """ should assign the hits to the closest line


    """
    def __init__(self):
        super().__init__()
        self.requirements = ["x_fiber_hits", "y_fiber_hits", "z_fiber_hits","in_and_f_z_y", "in_and_f_x_z", "in_and_f_x_y", "min_charge_thresh"]

    def _setup_cli_options(self, parser):
        """
        parser.add_argument(
            "--file_name", 
            help="Name of the saved root file", 
            required = False, default = "Hough2D-results", type = str,
        )
        """
        pass

    def _initialise(self) -> None:
        """
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
        self.xz_hits_x = std.vector('float')()
        self.xz_hits_z = std.vector('float')()
        self.xy_hits_x = std.vector('float')()
        self.xy_hits_y = std.vector('float')()


        self.tree.Branch("zy_hits_z", self.zy_hits_z)
        self.tree.Branch("zy_hits_y", self.zy_hits_y)
        self.tree.Branch("xz_hits_x", self.xz_hits_x)
        self.tree.Branch("xz_hits_z", self.xz_hits_z)
        self.tree.Branch("xy_hits_x", self.xy_hits_x)
        self.tree.Branch("xy_hits_y", self.xy_hits_y)

        self.line_index_zy = std.vector('int')() #will share the same index as the hits, and each entry tells which line on the zy plane the corresponding hit has been assigned to
        self.tree.Branch("line_index_zy", self.line_index_zy)
        self.line_index_xz = std.vector('int')() 
        self.tree.Branch("line_index_xz", self.line_index_xz)
        self.line_index_xy = std.vector('int')() 
        self.tree.Branch("line_index_xy", self.line_index_xy)
        """

        
    def _process(self, event:Event):
        """
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

        #zy plot
        if x_fiber_hits:
            for track in in_and_f_z_y:
                self.zyplot_in_zpos.push_back(float(track[0,0]))
                self.zyplot_in_ypos.push_back(float(track[0,1]))

                self.zyplot_f_zpos.push_back(float(track[1,0]))
                self.zyplot_f_ypos.push_back(float(track[1,1]))

            zy_data = data_2D(x_fiber_hits, "z", "y", self.min_charge_thresh)
            self.zy_hits_z = fill_vector_from_array(self.zy_hits_z, zy_data[:, 0], cast=float)
            self.zy_hits_y = fill_vector_from_array(self.zy_hits_y, zy_data[:, 1], cast=float)
            self.line_index_zy = fill_vector_from_array(self.line_index_zy, assign_line(zy_data, self.zyplot_in_zpos, self.zyplot_in_ypos, self.zyplot_f_zpos, self.zyplot_f_ypos))

            event.add_data("line_index_zy", self.line_index_zy)


        #xz plot
        if y_fiber_hits:
            for track in in_and_f_x_z:
                self.xzplot_in_xpos.push_back(float(track[0,0]))
                self.xzplot_in_zpos.push_back(float(track[0,1]))

                self.xzplot_f_xpos.push_back(float(track[1,0]))
                self.xzplot_f_zpos.push_back(float(track[1,1]))

            xz_data = data_2D(y_fiber_hits, "x", "z", self.min_charge_thresh)
            self.xz_hits_x = fill_vector_from_array(self.xz_hits_x, xz_data[:, 0], cast=float)
            self.xz_hits_z = fill_vector_from_array(self.xz_hits_z, xz_data[:, 1], cast=float)
            self.line_index_xz = fill_vector_from_array(self.line_index_xz, assign_line(xz_data, self.xzplot_in_xpos, self.xzplot_in_zpos, self.xzplot_f_xpos, self.xzplot_f_zpos))

            event.add_data("line_index_xz", self.line_index_xz)

        #xy plot
        if z_fiber_hits:
            for track in in_and_f_x_y:
                self.xyplot_in_xpos.push_back(float(track[0,0]))
                self.xyplot_in_ypos.push_back(float(track[0,1]))

                self.xyplot_f_xpos.push_back(float(track[1,0]))
                self.xyplot_f_ypos.push_back(float(track[1,1]))

            xy_data = data_2D(z_fiber_hits, "x", "y", self.min_charge_thresh)
            self.xy_hits_x = fill_vector_from_array(self.xy_hits_x, xy_data[:, 0], cast=float)
            self.xy_hits_y = fill_vector_from_array(self.xy_hits_y, xy_data[:, 1], cast=float)
            self.line_index_xy = fill_vector_from_array(self.line_index_xy, assign_line(xy_data, self.xyplot_in_xpos, self.xyplot_in_ypos, self.xyplot_f_xpos, self.xyplot_f_ypos))

            event.add_data("line_index_xy", self.line_index_xy)

        self.tree.Fill()
        """
        self.min_charge_thresh=event["min_charge_thresh"]

        x_fiber_hits = event["x_fiber_hits"]
        y_fiber_hits = event["y_fiber_hits"]
        z_fiber_hits = event["z_fiber_hits"]

        for fiber_hits, u_name, v_name, ax_ids in zip([x_fiber_hits, y_fiber_hits, z_fiber_hits], ["z", "x", "x"], ["y", "z", "y"], [[1,1], [0,0], [1,0]]):
            if not fiber_hits:
                continue
            data = data_2D(fiber_hits, u_name, v_name, self.min_charge_thresh)
            in_and_f_name = "in_and_f_" + u_name + "_" + v_name
            in_and_f=event[in_and_f_name]   
            if in_and_f.size == 0:
                in_and_f = np.empty((0, 2, 2))
                event.add_data(in_and_f_name, in_and_f)
                #continue         
            line_index = assign_line(data, in_and_f[:, 0, 0], in_and_f[:, 0, 1], in_and_f[:, 1, 0], in_and_f[:, 1, 1])
            total_charge_per_index = total_charge_per_index_func(line_index, data, len(in_and_f))
            total_charge_per_index_name = "total_charge_per_index_" + u_name + v_name
            line_index_name = "line_index_" + u_name + v_name
            #print("\n event number = ", event["id"]," graph =", u_name, v_name, " line_index = ", line_index)
            event.add_data(line_index_name, line_index)
            event.add_data(total_charge_per_index_name, total_charge_per_index)



    def _help(self)->str:
        return "Assigns hits to the closest fitted line"
        #return "Generates a ROOT Tree where the following information is saved for the 3 combinations zy, xz, and xy: \n - initial and final points of the Hough fitted lines\n - the hits\n - the line index which is a vector that contains the index of the closest track \n\n!!! Has never been tested for xy!!!"

    def _finalise(self):
        log_module_name("AssignHits", file_path="modules.txt")
        #self.file.Write()
        #self.file.Close()




