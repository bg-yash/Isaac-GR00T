from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, TypedDict

from p2_teleop.actions import Action
from p2_teleop.observations import Observation


class StateActionStep(TypedDict):
    """Class for representing a state-action pair"""

    time: float
    obs: Observation
    actions: Action


@dataclass
class Episode:
    """Class for representing an episode when loaded from saved data"""

    path: Path
    metadata: Dict
    metadata_path: Path
    image_paths_dict: Dict
    images_dict: Dict
    were_images_loaded: bool
    capture_times_dict: Dict
    state_action_data: List[StateActionStep]
