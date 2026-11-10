import typing

from liquidreco.base import Singleton

from liquidreco.modules.module_base import ModuleBase
from liquidreco.modules.peak_finding import HesseRidgeDetection2D, HesseRidgeDetection3D, PeakFinder2D
from liquidreco.modules.reconstruction import HoughTransform2D, HoughTransform3D, LocalMeanDBSCAN, MinimumSpanningTree2D, RANSAC2D
from liquidreco.modules.hit_building import HitBuilder2D, HitBuilder3D
from liquidreco.modules.utility import WeightScaling
from liquidreco.modules.plotting import HitPlotter2D, HitPlotter3D

from liquidreco.modules.Hit_assigning import AssignHits #Pascal
from liquidreco.modules.Track_finding import CutLines, Refit_lines, Find_simple_tracks, Ransac_refit_lines, Refit_lines_TLS, Unasigned_refit_lines, Unasigned_RANSAC_refit_lines
from liquidreco.modules.tree_making import MakeTree
from liquidreco.modules.plot_HoughSpace import HoughTransformPlot
from liquidreco.modules.ThreeD_Matching import ThreeDMatching
from liquidreco.modules.Soft_ThreeD_Matching import Find_candidate_tracks, SoftThreeDMatching, Muon_Main_track_statistics, Build_ThreeD_tracks, Add_internal_nodes

    
class ModuleList(metaclass=Singleton):

    def __init__(self):
        
        self._modules: typing.List[typing.Type[ModuleBase]] = []
        self._module_map: typing.List[str, typing.Type[ModuleBase]] = {}
    
    def register(self, module: typing.Type[ModuleBase]) -> None:
        """Register a module

        :param module: The module class to register
        :type module: typing.Type[ModuleBase]
        """

        if module in self._modules:
            print(f"WARNING: Trying to register module {module.__name__} but it has already been registered")
            return
        
        self._modules.append(module)
        self._module_map[module.__name__] = self._modules[-1]

    def get_modules(self) -> typing.List[typing.Type[ModuleBase]]:
        """Get a list of all the registered modules

        :return: All the registered modules
        :rtype: typing.List[typing.Type[ModuleBase]]
        """

        return self._modules
    
    def get_module_names(self) -> typing.List[str]:
        """Get a list of the names of all registered modules

        :return: list of names
        :rtype: typing.List[str]
        """

        return [c.__name__ for c in self._modules]
    
    def get_module(self, name: str) -> typing.Type[ModuleBase]:
        """Get a module by name

        :param name: the name of the module
        :type name: str
        :raises ValueError: If the module that has been asked for has not been registered
        :return: The requested module
        :rtype: typing.Type[ModuleBase]
        """

        if not name in self._module_map.keys():
            raise ValueError(f"Asked for module {name} but it has not been registered!!\nAvailable modules: {self._modules}")
        
        return self._module_map[name]

    
####### Register all the modules ###########

## utility
ModuleList().register(WeightScaling)

## hit building
ModuleList().register(HitBuilder2D)
ModuleList().register(HitBuilder3D)

## peak finding
ModuleList().register(HesseRidgeDetection2D)
ModuleList().register(HesseRidgeDetection3D)
ModuleList().register(PeakFinder2D)

## reconstruction
ModuleList().register(HoughTransform2D)
ModuleList().register(HoughTransform3D)
ModuleList().register(MinimumSpanningTree2D)

ModuleList().register(AssignHits) #Pascal
ModuleList().register(CutLines)
ModuleList().register(Refit_lines)
ModuleList().register(Find_simple_tracks)
ModuleList().register(Ransac_refit_lines)
ModuleList().register(Refit_lines_TLS)
ModuleList().register(Unasigned_refit_lines)
ModuleList().register(HoughTransformPlot)
ModuleList().register(RANSAC2D)
ModuleList().register(Unasigned_RANSAC_refit_lines)

#3D matching
ModuleList().register(ThreeDMatching)
ModuleList().register(Find_candidate_tracks)
ModuleList().register(SoftThreeDMatching)
ModuleList().register(Add_internal_nodes)
ModuleList().register(Build_ThreeD_tracks)
#ModuleList().register(LocalMeanDBSCAN) <- this thing is mad broken... maybe one day it will be fixed

## plotting
ModuleList().register(HitPlotter2D)
ModuleList().register(HitPlotter3D)

##ROOT Tree
ModuleList().register(MakeTree)
##Statistics
ModuleList().register(Muon_Main_track_statistics)