import typing

import numpy as np
import uproot
from tqdm import tqdm

from liquidreco.modules.module_base import ModuleBase
from liquidreco.event import Event

class EventProcessor:
    """Main application loop for liquidreco

    This is responsible for loading up events from the input file and
    looping over them, applying fit modules specified by the user.
    """

    ## Optional per-hit truth branches (written by TTreeMaker when run with -O truth)
    ## maps name of the branch in the hit tree -> name of the data in the Event
    TRUTH_HIT_BRANCHES: typing.Dict[str, str] = {
        "true_ntracks":    "raw_true_n_tracks",   ## number of true tracks contributing to the hit
        "true_track_id":   "raw_true_track_ids",  ## id of the main contributing track (most deposited energy)
        "true_parent_id":  "raw_true_parent_ids", ## id of the parent of the main track
        "true_primary_id": "raw_true_primary_ids",## id of the primary particle the main track descends from
        "true_pdg":        "raw_true_pdgs",       ## PDG code of the main track
        "true_edep":       "raw_true_edeps",      ## energy deposited in the hit by the main track
        "true_start":      "raw_true_starts",     ## start point of the main track, shape [nHits, 4] (x, y, z, t)
        "true_end":        "raw_true_ends",       ## end point of the main track, shape [nHits, 4] (x, y, z, t)
        "true_track_ids":  "raw_true_all_track_ids", ## ids of *all* contributing tracks (variable length per hit)
        "true_edeps":      "raw_true_all_edeps",     ## energy deposited by each of those tracks (variable length per hit)
    }

    ## Optional tree with one entry per true trajectory (written by TTreeMaker when run with -O truth)
    TRAJECTORY_TREE_NAME: str = "trajectories"
    TRAJECTORY_BRANCHES: typing.List[str] = [
        "track_id", "parent_id", "primary_id", "pdg", "start", "end", "mom"
    ]

    def __init__(
            self, 
            file_name: str, 
            tree_name: str, 
            modules: typing.List[ModuleBase],
            max_n_events: int = None
        ):
        """Initialiser

        :param file_name: Name of the root file containing hit info
        :type file_name: str
        :param tree_name: The name of the tree within the input file that holds the hits
        :type tree_name: str
        :param modules: List of modules to apply to the events
        :type modules: typing.List[ModuleBase]
        :param max_n_events: Only process this many events, defaults to None
        :type max_n_events: int, optional
        """

        self._file_name = file_name
        self._tree_name = tree_name
        self._modules = modules
        self._max_n_events = max_n_events

        self._read_hit_info()

    def _read_hit_info(self) -> None:
        """load up the input root file and read event info from it
        """

        with uproot.open(self._file_name) as input_file:

            ## get info from the hit tree
            hit_tree = input_file[self._tree_name]

            assert hit_tree is not None, f"No hit tree found with the name {self._tree_name}"

            self._positions = np.array(hit_tree["pos"].array(library="np").tolist())
            self._weights   = np.array(hit_tree["charge"].array(library="np").tolist())
            self._times     = np.array(hit_tree["time"].array(library="np").tolist())
            self._event_ids = np.array(hit_tree["event"].array(library="np").tolist())

            ## get the truth info for the hits, if the tree has any
            self._truth: typing.Dict[str, np.ndarray] = {}
            available_branches = hit_tree.keys()
            for branch_name, data_name in self.TRUTH_HIT_BRANCHES.items():
                if branch_name in available_branches:
                    self._truth[data_name] = hit_tree[branch_name].array(library="np")

            ## get the true trajectories, if the file has them
            self._trajectories: typing.Dict[str, np.ndarray] = {}
            self._trajectory_event_ids: np.ndarray = None
            if self.TRAJECTORY_TREE_NAME in input_file:
                trajectory_tree = input_file[self.TRAJECTORY_TREE_NAME]
                self._trajectory_event_ids = trajectory_tree["event"].array(library="np")
                for branch_name in self.TRAJECTORY_BRANCHES:
                    self._trajectories[branch_name] = trajectory_tree[branch_name].array(library="np")

    def event_loop(self) -> None:
        """Does the loop over all events, applying each specified module to them
        """

        ## loop through each event
        unique_event_ids = np.unique(self._event_ids)

        ## check if max n events was specified, if so only do that many
        event_ids = unique_event_ids
        if self._max_n_events is not None and len(unique_event_ids) > self._max_n_events:
            event_ids = unique_event_ids[:self._max_n_events]

        ## initialise the modules
        for module in self._modules:
            module.initialise()
        
        for event_id in tqdm(event_ids):

            ## positions in the position, weight and time arrays for this event
            event_array_positions = np.where(self._event_ids == event_id)[0]

            ## values for hits belonging to this particular event
            event_positions = self._positions[event_array_positions, ...]
            event_weights   = self._weights[event_array_positions, ...]
            event_times     = self._times[event_array_positions, ...]

            ## set up the event
            event = Event()
            event.add_data("raw_positions", event_positions)
            event.add_data("raw_weights", event_weights)
            event.add_data("raw_times", event_times)
            event.add_data("id", event_id)

            ## add truth info for the hits (only present if it was in the input file)
            for data_name, truth_values in self._truth.items():
                event.add_data(data_name, truth_values[event_array_positions, ...])

            ## add the true trajectories of this event (only present if they were in the input file)
            ## dict of arrays with one entry per trajectory: track_id, parent_id, primary_id, pdg, start, end, mom
            if self._trajectory_event_ids is not None:
                event_trajectory_positions = np.where(self._trajectory_event_ids == event_id)[0]
                event.add_data(
                    "true_trajectories", 
                    {
                        name: values[event_trajectory_positions, ...] 
                        for name, values in self._trajectories.items()
                    }
                )

            ## run each of the specified modules
            for module in self._modules:
                module.process(event)

        ## tear down the modules
        for module in self._modules:
            module.finalise()