import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from liquidreco.event import Event

from linscan import LINSCAN
# 3D hough transform
from hough3d.basic_hough import Hough3D, Hough2D
# 2D hough transform
from skimage.transform import probabilistic_hough_line

from sklearn.cluster import DBSCAN

import matplotlib
import typing

from scipy.sparse import csr_array
from scipy.sparse import csgraph

from sklearn.neighbors import NearestNeighbors


from liquidreco.plotting import make_corner_plot, make_rotating_gif, make_corner_plot_fiber_hits
from liquidreco.modules.module_base import ModuleBase

from liquidreco.save_used_modules import log_module_name


def plotHoughSpace(HoughSpace, maxRho, fig, ax):
    ax.set_xlim(0, HoughSpace.shape[1])
    ax.set_ylim(0, HoughSpace.shape[0])
    ax.set_xlabel("rho[mm]")
    ax.set_ylabel("theta[rad]")

    # Replace index ticks with physical-value ticks
    n_ticks = 5
    rho_tick_positions = np.linspace(0, HoughSpace.shape[1], n_ticks)
    rho_tick_labels = np.linspace(-maxRho, maxRho, n_ticks)
    ax.set_xticks(rho_tick_positions)
    ax.set_xticklabels([f"{v:.0f}" for v in rho_tick_labels], fontsize=7)   

    theta_tick_positions = np.linspace(0, HoughSpace.shape[0], n_ticks)
    theta_tick_labels = np.linspace(0, np.pi, n_ticks)
    ax.set_yticks(theta_tick_positions)
    ax.set_yticklabels([f"{v:.2f}" for v in theta_tick_labels], fontsize=7)

    im = ax.imshow(HoughSpace, origin='lower', aspect='auto', cmap='viridis', norm=PowerNorm(gamma=0.1, vmin=HoughSpace.min(), vmax=HoughSpace.max()))
    ax.set_autoscale_on(False)
    fig.colorbar(im, ax=ax)

    peak_idx = np.unravel_index(np.argmax(HoughSpace), HoughSpace.shape)
    theta_peak_idx, rho_peak_idx = peak_idx
    ax.scatter(rho_peak_idx, theta_peak_idx, color='yellow', linewidths=0.3, marker='x', s=5)

def zoomedHoughSpace(HoughSpace, fig, ax, rho_bin_centers, thetas):
    peak_idx = np.unravel_index(np.argmax(HoughSpace), HoughSpace.shape)
    theta_peak_idx, rho_peak_idx = peak_idx

    t_lo, t_hi = max(0, theta_peak_idx-10), min(HoughSpace.shape[0]-1, theta_peak_idx+10)
    r_lo, r_hi = max(0, rho_peak_idx-25), min(HoughSpace.shape[1]-1, rho_peak_idx+25)   
    theta_lo_value = thetas[t_lo]
    theta_hi_value = thetas[t_hi]
    rho_lo_value = rho_bin_centers[r_lo]
    rho_hi_value = rho_bin_centers[r_hi]


    ax.set_xlabel("rho[mm]")
    ax.set_ylabel("theta[rad]")


    #ax.set_xticklabels([f"{v:.0f}" for v in rho_tick_labels], rotation=45, ha='right')
    n_ticks = 5
    rho_tick_positions = np.linspace(0, r_hi-r_lo-1, n_ticks)
    rho_tick_labels = np.linspace(rho_lo_value, rho_hi_value, n_ticks)
    ax.set_xticks(rho_tick_positions)
    ax.set_xticklabels([f"{v:.0f}" for v in rho_tick_labels], fontsize=7) 

    theta_tick_positions = np.linspace(0, t_hi-t_lo-1, n_ticks)
    theta_tick_labels = np.linspace(theta_lo_value, theta_hi_value, n_ticks)
    ax.set_yticks(theta_tick_positions)
    ax.set_yticklabels([f"{v:.2f}" for v in theta_tick_labels], fontsize=7) 

    im = ax.imshow(HoughSpace[t_lo:t_hi, r_lo:r_hi], origin='lower', aspect='auto', cmap='viridis') #, norm=PowerNorm(gamma=0.1, vmin=HoughSpace.min(), vmax=HoughSpace.max())
    ax.set_autoscale_on(False)
    fig.colorbar(im, ax=ax)
    ax.scatter(rho_peak_idx-r_lo, theta_peak_idx-t_lo, color='red', linewidths=1., marker='o', s=10)
    fig.suptitle("Hough Space around the highest voted line")


def plotHoughSpace3dResult(linePoints, n_theta, stepSize, maxRho, ax):
    x0 = linePoints[:, 0, 0]
    y0 = linePoints[:, 0, 1]
    x1 = linePoints[:, 1, 0]
    y1 = linePoints[:, 1, 1]

    dir_vec = linePoints[:, 1, :] - linePoints[:, 0, :]
    dir_vec = dir_vec / np.linalg.norm( dir_vec, axis = 1, keepdims=True)
    sintheta = dir_vec[:, 0]
    costheta = -dir_vec[:, 1]
    theta = np.arctan2(sintheta, costheta)
    rho = x0 * costheta + y0 * sintheta

    mask = theta<0
    theta[mask] += np.pi
    rho[mask]*= -1

    theta_idx = theta / np.pi * (n_theta - 1)          # matches thetas = np.linspace(0, np.pi, n_theta)
    rho_idx = (rho + maxRho) / stepSize 

    ax.scatter(rho_idx, theta_idx, facecolors='none', color='red', linewidths=0.3, marker='o', s=5)


def oneDHoughPlot(_PDF, u_name, v_name, HoughSpace, maxRho, linePoints, n_theta, stepSize, event_number):
     # Sum over theta, giving a 1D histogram over rho
    rho_sum_projection = HoughSpace.sum(axis=0)   # shape: (n_rho,)
    rho_max_projection = HoughSpace.max(axis=0)

    # Sum over rho, giving a 1D histogram over theta
    theta_sum_projection = HoughSpace.sum(axis=1)  # shape: (n_theta,)
    theta_max_projection = HoughSpace.max(axis=1)

    fig1d, axs1d = plt.subplots(2, 2, sharex= 'col', sharey = 'row', figsize=(5, 5))

    axs1d[0,0].plot(rho_sum_projection)
    #axs1d[0,0].set_xlabel("rho [bin index]")
    axs1d[0,0].set_ylabel("summed votes")

    axs1d[1,0].plot(rho_max_projection)
    axs1d[1,0].set_xlabel("rho [bin index]")
    axs1d[1,0].set_ylabel("max votes")

    axs1d[0,1].plot(theta_sum_projection)
    #axs1d[0,1].set_xlabel("theta [bin index]")
    #axs1d[0,1].set_ylabel("summed votes over rho")

    axs1d[1,1].plot(theta_max_projection)
    axs1d[1,1].set_xlabel("theta [bin index]")
    #axs1d[1,1].set_ylabel("max votes over rho")

    n_ticks = 5
    rho_tick_positions = np.linspace(0, HoughSpace.shape[1], n_ticks)
    rho_tick_labels = np.linspace(-maxRho, maxRho, n_ticks)
    axs1d[1,0].set_xticks(rho_tick_positions)
    axs1d[1,0].set_xticklabels([f"{v:.0f}" for v in rho_tick_labels])

    theta_tick_positions = np.linspace(0, HoughSpace.shape[0], n_ticks)
    theta_tick_labels = np.linspace(0, np.pi, n_ticks)
    axs1d[1,1].set_xticks(theta_tick_positions)
    axs1d[1,1].set_xticklabels([f"{v:.2f}" for v in theta_tick_labels])

    x0 = linePoints[:, 0, 0]
    y0 = linePoints[:, 0, 1]
    x1 = linePoints[:, 1, 0]
    y1 = linePoints[:, 1, 1]

    dir_vec = linePoints[:, 1, :] - linePoints[:, 0, :]
    dir_vec = dir_vec / np.linalg.norm( dir_vec, axis = 1, keepdims=True)
    sintheta = dir_vec[:, 0]
    costheta = -dir_vec[:, 1]
    theta = np.arctan2(sintheta, costheta)
    rho = x0 * costheta + y0 * sintheta

    mask = theta<0
    theta[mask] += np.pi
    rho[mask]*= -1

    theta_idx = theta / np.pi * (n_theta - 1)          # matches thetas = np.linspace(0, np.pi, n_theta)
    rho_idx = (rho + maxRho) / stepSize

    for peak_idx in rho_idx:
        axs1d[0,0].axvline(x=peak_idx, color='red', linestyle='--', linewidth=1)
        axs1d[1,0].axvline(x=peak_idx, color='red', linestyle='--', linewidth=1)
    for peak_idx in theta_idx:
        axs1d[0,1].axvline(x=peak_idx, color='red', linestyle='--', linewidth=1)
        axs1d[1,1].axvline(x=peak_idx, color='red', linestyle='--', linewidth=1)

    title = "1 dimensional projection of the " + u_name + v_name + " Hough space \n of event " + str(event_number)
    fig1d.suptitle(title)
    _PDF.savefig(fig1d)
    plt.close(fig1d)










class HoughTransformPlot(ModuleBase):
    """ Performs simple Hough line transform on 2D hits

    Right now only makes plots of the fitted lines, doesn't save them to the event
    """

    def __init__(self):
        super().__init__()

        self.requirements = ["x_fiber_hits", "y_fiber_hits", "z_fiber_hits"]

    def _setup_cli_options(self, parser):

        parser.add_argument(
            "--min-charge-thresh", 
            help="Minimum charge that a hit must have to be included in the track fitting", 
            required = False, default = 0.0, type = float,
        )
        parser.add_argument(
            "--max-plot-charge", 
            help="Maximum charge for plots", 
            required = False, default = 100.0, type = float,
        )
        parser.add_argument("--event_number", help = "Event you want to plot", required = True, type = int )
        
    def _initialise(self) -> None:
        
        self.min_charge_thresh = self.args.min_charge_thresh
        self.max_plot_charge = self.args.max_plot_charge
        self.event_number = self.args.event_number

        self.hough_finder = Hough3D(
            neighbour_dist=1.5, min_points_per_line=5, lattice_step_size=1.
        ) #they were neighbour_dist=1.5, min_points_per_line=5, lattice_step_size=1.

        self.hough_space2D = Hough2D(stepSize=1.)
        pdf_name = "Hough_Space_Plot_event_"+str(self.event_number)+".pdf"
        self._pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

        self.clusterer = DBSCAN(40.0)
        #self.clusterer = DBSCAN(2.5)
        



    def _process(self, event:Event):
        """ Perform Hough transform on an event and save the result to a given file

        :param event: Object describing the hits in an event
        :type event: Event
        """
        current_event_n = event["id"]
        #print("\n Current event number = ", current_event_n)
        if current_event_n == self.event_number:
            event.add_data("min_charge_thresh", self.min_charge_thresh)

            x_fiber_hits = event["x_fiber_hits"]
            y_fiber_hits = event["y_fiber_hits"]
            z_fiber_hits = event["z_fiber_hits"]

            fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(5, 5))
            figHough, axsHough = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(5, 5))
            figZoom, axZoom = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(5, 5))
            HoughTransformName = "Hough Transform 2D event: " + str(self.event_number)
            fig.suptitle(HoughTransformName)
            HoughSpaceName = "Hough Space event: " + str(self.event_number)
            figHough.suptitle(HoughSpaceName)

            make_corner_plot_fiber_hits(
                fig,
                axs, 
                x_fiber_hits,
                y_fiber_hits,
                z_fiber_hits
            )

            for fiber_hits, u_name, v_name, ax_ids in zip([
                x_fiber_hits,
                y_fiber_hits,
                z_fiber_hits
            ],
            ["z", "x", "x"],
            ["y", "z", "y"],
            [[1,1], [0,0], [1,0]]
            ):

                data = np.array(
                    [[getattr(hit, u_name), getattr(hit, v_name), 0.0] for hit in fiber_hits if hit.weight > self.min_charge_thresh]
                )

                ax = axsHough[ax_ids[0], ax_ids[1]]
                in_and_f_name="in_and_f_"+u_name+"_"+v_name #Pascal
                all_linePoints=np.empty((0,2,2))
                event.add_data(in_and_f_name, all_linePoints)

                if data.shape[0] == 0:
                    # maybe there are no hits above threshold
                    
                    plt.tight_layout()

                    self._pdf.savefig(fig)
                    self._pdf.savefig(figHough)
                    self._pdf.savefig(figZoom)
                    fig.clf()
                    figHough.clf()
                    figZoom.clf()

                    return

                cluster_ids = self.clusterer.fit_predict(data)

                
                unique_ids, counts = np.unique(cluster_ids, return_counts=True)
                biggest_idx = np.argmax(counts)
                cluster_id = unique_ids[biggest_idx]
                if cluster_id == -1:
                    # -1 is id for "noise"
                        continue
                
                #event.add_data(in_and_f_name, linePoints[:,:,:2]) #Pascal
                HoughSpace, maxRho, n_theta, stepSize, rho_bin_centers, thetas = self.hough_space2D(data[cluster_ids == cluster_id, :2])
                print("\n min hough = ", HoughSpace.min())
                print("\n max hough = ", HoughSpace.max()) 
                plotHoughSpace(HoughSpace, maxRho, figHough, axsHough[ax_ids[0], ax_ids[1]])
                print("\n xlim, ylim, HoughSpace.shape", ax.get_xlim(), ax.get_ylim(), HoughSpace.shape)
                linePoints = self.hough_finder(data[cluster_ids == cluster_id, :]) #, plotHoughSpace=True 
                plotHoughSpace3dResult(linePoints, n_theta, stepSize, maxRho, ax)
                all_linePoints=np.concatenate([all_linePoints, linePoints[:,:,:2]], axis=0)
                oneDHoughPlot(self._pdf, u_name, v_name, HoughSpace, maxRho, linePoints, n_theta, stepSize, self.event_number)
                zoomedHoughSpace(HoughSpace, figZoom, axZoom[ax_ids[0], ax_ids[1]], rho_bin_centers, thetas)
                
                for i in range(linePoints.shape[0]):
                        axs[ax_ids[0], ax_ids[1]].plot(linePoints[i, :, 0], linePoints[i, :, 1])
                event.add_data(in_and_f_name, all_linePoints) #Pascal
                #print("\n"+in_and_f_name+"=", event[in_and_f_name])



            plt.tight_layout()

            self._pdf.savefig(fig)
            self._pdf.savefig(figHough)
            self._pdf.savefig(figZoom)

            plt.close(fig)
            plt.close(figHough)
            plt.close(figZoom)
    
    def _finalise(self):
        self._pdf.close()
        log_module_name("HoughTransformPlot", file_path="modules.txt")
