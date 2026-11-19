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

    ## Optional per-hit truth branches (written by TTreeMaker when run with -O truth_g4seg)
    ## maps name of the branch in the hit tree -> name of the data in the Event
    TRUTH_HIT_BRANCHES: typing.Dict[str, str] = {
        "true_nsegs":      "raw_true_n_segs",     ## number of true G4 segments contributing to the hit
        "true_ntrajs":     "raw_true_n_trajs",    ## number of distinct true trajectories contributing to the hit
        "true_edep":       "raw_true_edeps",      ## total true energy deposited in the hit (sum over segments)
    }

    ## Optional per-hit true G4 segment branches (written by TTreeMaker when run with -O truth_g4seg)
    ## each holds a variable length vector per hit, one entry per segment
    SEGMENT_BRANCHES: typing.List[str] = [
        "true_seg_start_x", "true_seg_start_y", "true_seg_start_z", "true_seg_start_t",
        "true_seg_stop_x", "true_seg_stop_y", "true_seg_stop_z", "true_seg_stop_t",
        "true_seg_edep", "true_seg_length", "true_seg_primary_id",
        "true_seg_ncontrib", "true_seg_contrib_ids",
    ]

    ## Optional tree with one entry per true trajectory (written by TTreeMaker when run with -O truth_traj)
    TRAJECTORY_TREE_NAME: str = "trajectories"
    TRAJECTORY_BRANCHES: typing.List[str] = [
        "traj_id", "parent_id", "primary_id", "pdg", "start", "end", "mom"
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

            ## get the true G4 segments of the hits, if the tree has them
            ## (object arrays, one variable length array per hit)
            self._segments: typing.Dict[str, np.ndarray] = {}
            if all(branch_name in available_branches for branch_name in self.SEGMENT_BRANCHES):
                for branch_name in self.SEGMENT_BRANCHES:
                    self._segments[branch_name] = hit_tree[branch_name].array(library="np")

            ## get the true trajectories, if the file has them
            self._trajectories: typing.Dict[str, np.ndarray] = {}
            self._trajectory_event_ids: np.ndarray = None
            if self.TRAJECTORY_TREE_NAME in input_file:
                trajectory_tree = input_file[self.TRAJECTORY_TREE_NAME]
                self._trajectory_event_ids = trajectory_tree["event"].array(library="np")
                for branch_name in self.TRAJECTORY_BRANCHES:
                    self._trajectories[branch_name] = trajectory_tree[branch_name].array(library="np")

    def _add_segments(self, event: Event, hit_positions: np.ndarray) -> None:
        """Add the true G4 segments of the hits of one event to the event

        Adds the following, each an object array with one entry per hit, where each
        entry holds one value per G4 segment of that hit:

        - raw_true_seg_starts:       segment start points, each of shape [nSegs, 4] (x, y, z, t)
        - raw_true_seg_stops:        segment stop points, each of shape [nSegs, 4] (x, y, z, t)
        - raw_true_seg_edeps:        energy deposited by each segment, each of shape [nSegs]
        - raw_true_seg_lengths:      length of each segment, each of shape [nSegs]
        - raw_true_seg_primary_ids:  id of the primary particle of each segment, each of shape [nSegs]
        - raw_true_seg_contrib_ids:  for each segment, the ids of all the trajectories that contributed to it
                                     (a list of nSegs arrays, since a segment can combine several trajectories)

        The contributor ids are the G4 trajectory ids, they can be traced back with the
        true_trajectories (traj_id column) if the input file has the trajectories tree.

        :param event: the event to add the segments to
        :type event: Event
        :param hit_positions: positions in the input arrays of the hits that belong to this event
        :type hit_positions: np.ndarray
        """

        segs = {name: values[hit_positions] for name, values in self._segments.items()}
        n_hits = len(hit_positions)

        def per_hit(values) -> np.ndarray:
            ## make an object array so that numpy doesn't try to merge the per-hit arrays
            ret = np.empty(n_hits, dtype=object)
            for i, v in enumerate(values):
                ret[i] = v
            return ret

        def points(prefix: str) -> np.ndarray:
            return per_hit(
                np.stack(
                    [
                        np.asarray(segs[f"{prefix}_{coord}"][i], dtype=float).reshape(-1)
                        for coord in ("x", "y", "z", "t")
                    ],
                    axis=1
                )
                for i in range(n_hits)
            )

        event.add_data("raw_true_seg_starts", points("true_seg_start"))
        event.add_data("raw_true_seg_stops", points("true_seg_stop"))
        event.add_data("raw_true_seg_edeps", per_hit(np.asarray(v) for v in segs["true_seg_edep"]))
        event.add_data("raw_true_seg_lengths", per_hit(np.asarray(v) for v in segs["true_seg_length"]))
        event.add_data("raw_true_seg_primary_ids", per_hit(np.asarray(v) for v in segs["true_seg_primary_id"]))

        ## the contributor ids are stored concatenated for all segments of a hit,
        ## split them back up using the number of contributors of each segment
        event.add_data(
            "raw_true_seg_contrib_ids",
            per_hit(
                np.split(np.asarray(ids), np.cumsum(n_contrib)[:-1]) if len(n_contrib) > 0 else []
                for ids, n_contrib in zip(segs["true_seg_contrib_ids"], segs["true_seg_ncontrib"])
            )
        )

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

            ## add the true G4 segments of the hits (only present if they were in the input file)
            if self._segments:
                self._add_segments(event, event_array_positions)

            ## add the true trajectories of this event (only present if they were in the input file)
            ## dict of arrays with one entry per trajectory: traj_id, parent_id, primary_id, pdg, start, end, mom
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