import time
import numpy as np
from pynput import keyboard


class KeyboardYawController:
    def __init__(self, hfg_agent):
        self.hfg_agent = hfg_agent
        self.current_angle = 0.0
        self.increment = 1.0
        self.running = True
        self.key_pressed = None
        self.MIN_ANGLE = np.rad2deg(self.hfg_agent.yawing_gripper.POSITION_LIMITS[0])
        self.MAX_ANGLE = np.rad2deg(self.hfg_agent.yawing_gripper.POSITION_LIMITS[1])

    def on_press(self, key):
        try:
            if key.char == "a":
                self.key_pressed = "left"
            elif key.char == "d":
                self.key_pressed = "right"
            elif key.char == "q":
                self.running = False
                return False
        except AttributeError:
            if key == keyboard.Key.enter:
                print("\nPress q to quit")

    def on_release(self, key):
        self.key_pressed = None

    def control_loop(self):
        print("\n=== Interactive Yaw Control Mode ===")
        print("Hold 'a' - Rotate couter-clockwise")
        print("Hold 'd' - Rotate clockwise")
        print("Press 'q' - Confirm Position\n")

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        while self.running:
            if self.key_pressed == "left":
                if self.current_angle <= self.MIN_ANGLE:
                    print(
                        f"\rAt minimum angle limit ({self.MIN_ANGLE}°)",
                        end="",
                        flush=True,
                    )
                    continue

                self.current_angle = max(
                    self.MIN_ANGLE, self.current_angle - self.increment
                )
                self.hfg_agent.yawing_gripper.rotate(
                    np.deg2rad(self.current_angle),
                    self.hfg_agent.yawing_gripper.CMD_VELOCITY,
                    self.hfg_agent.yawing_gripper.CMD_ACCELERATION,
                )
                print(
                    f"\rCurrent angle: {self.current_angle:.1f} degrees ",
                    end="",
                    flush=True,
                )

            elif self.key_pressed == "right":
                if self.current_angle >= self.MAX_ANGLE:
                    print(
                        f"\rAt maximum angle limit ({self.MAX_ANGLE}°)     ",
                        end="",
                        flush=True,
                    )
                    continue

                self.current_angle = min(
                    self.MAX_ANGLE, self.current_angle + self.increment
                )
                self.hfg_agent.yawing_gripper.rotate(
                    np.deg2rad(self.current_angle),
                    self.hfg_agent.yawing_gripper.CMD_VELOCITY,
                    self.hfg_agent.yawing_gripper.CMD_ACCELERATION,
                )
                print(
                    f"\rCurrent angle: {self.current_angle:.1f} degrees",
                    end="",
                    flush=True,
                )

            time.sleep(0.1)

        listener.stop()


def interactive_yaw_control(hfg_agent):
    """Start interactive yaw control mode.

    :param hfg_agent: Robot agent interface
    :returns: Final yaw angle or None if cancelled
    :rtype: float or None
    """

    controller = KeyboardYawController(hfg_agent)

    current_yaw_angle = np.rad2deg(
        controller.hfg_agent.yawing_gripper.get_state().position
    )
    print(
        f"\nWould you like to enter interactive yaw control mode ? (y/n) | Current yaw angle : {current_yaw_angle:.1f} deg"
    )
    if input().lower() != "y":
        controller.current_angle = current_yaw_angle
        return controller.current_angle

    controller.control_loop()

    if controller.current_angle is not None:
        return controller.current_angle
    return None
