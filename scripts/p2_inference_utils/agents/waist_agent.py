from time import time

from oculus_reader import OculusReader

from p2_teleop.agents.base_agent import BaseAgent
from p2_teleop.constants import DEFAULT_LEFT_IP
from p2_teleop.event_manager import ButtonEventManager
from p2_teleop.p2 import waist_api
from p2_teleop.p2.bg_p2_utils import setup_waist
from p2_teleop.observations import ObservationWaist
from p2_teleop.actions import ActionWaist


class P2WaistAgent(BaseAgent):
    """Does not inherit from QuestAgent because it shouldn't really share and implementations, just the interface"""

    def __init__(
        self,
        event_manager: ButtonEventManager | None,
        oculus_reader: OculusReader | None,
        ip: str = DEFAULT_LEFT_IP,
        dry_run=False,
    ):
        super().__init__(event_manager)
        self.oculus_reader = oculus_reader
        self.ip = ip
        self.dry_run = dry_run

        if not dry_run:
            setup_waist(self.ip)

    def get_episode_metadata(self):
        return {
            "ip": self.ip,
        }

    def get_obs(self):
        _, waist_positions = waist_api.getWaistJointsPosition(self.ip)
        t = time()
        return ObservationWaist(
            waist_yaw=waist_positions[0],
            waist_pitch=waist_positions[1],
            t=t,
        )

    def plan(self, obs: ObservationWaist):
        return ActionWaist()

    def execute(self, action: ActionWaist):
        pass
