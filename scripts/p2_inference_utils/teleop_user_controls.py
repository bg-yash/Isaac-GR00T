import time
from enum import Enum
from typing import Callable, Dict

from p2_teleop.event_manager import ButtonEvent, EventState


class ButtonType(Enum):
    PRIMARY = 1
    SECONDARY = 2
    JOYSTICK = 3


def get_primary_button(which_hand, button_data):
    if button_data is None or len(button_data) == 0:
        return False
    if which_hand == "l":
        # The quest may mis-recognize which controller is which
        return button_data.get("X", False)
    else:

        return button_data.get("A", False)


def get_secondary_button(which_hand, button_data):
    if button_data is None or len(button_data) == 0:
        return False
    return (
        button_data.get("Y", False)
        if which_hand == "l"
        else button_data.get("B", False)
    )


def get_joystick_button(which_hand, button_data):
    if button_data is None or len(button_data) == 0:
        return False
    return button_data.get(f"{which_hand.upper()}J", False)


def get_button(button_type: ButtonType, which_hand, button_data):
    match button_type:
        case ButtonType.PRIMARY:
            return get_primary_button(which_hand, button_data)
        case ButtonType.SECONDARY:
            return get_secondary_button(which_hand, button_data)
        case ButtonType.JOYSTICK:
            return get_joystick_button(which_hand, button_data)
        case _:
            raise NotImplementedError(f"Button type {button_type} not implemented")


def get_other_button_pressed(button_type, which_hand, button_data):
    other_pressed = False
    for btn_type in ButtonType:
        if btn_type == button_type:
            continue
        other_pressed = other_pressed or get_button(btn_type, which_hand, button_data)
    return other_pressed


class SingleClick(ButtonEvent):

    def __init__(
        self,
        button_type,
        which_hand,
        callback,
        priority=0,
        timeout=None,
        mutually_exclusive=True,
    ):
        """

        :param button_type: Primary or Secondary
        :param which_hand: Which hand to check, "l" or "r"
        :param callback: Callback
        :param priority: Used to determine which events take priority over others
        :param timeout: Used to prevent LongPresses from being triggered
        :param mutually_exclusive: Used to prevent failed/sloppy Multi-Button gestures from triggering SingleClicks
        """
        super().__init__(callback, priority)
        self.button_type = button_type
        self.which_hand = which_hand
        self.timeout = timeout
        self.was_pressed = False
        self.mutually_exclusive = mutually_exclusive
        self.start_t = None

    def check_event(self, button_data):
        pressed = get_button(self.button_type, self.which_hand, button_data)
        other_pressed = get_other_button_pressed(
            self.button_type, self.which_hand, button_data
        )
        if self.mutually_exclusive and other_pressed:
            self.was_pressed = False
            self._state = EventState.INACTIVE
            return False
        elif pressed and not self.was_pressed:
            self.was_pressed = True
            self.start_t = time.time()
            self._state = EventState.ACTIVE
        elif not self.was_pressed and not pressed:
            self.was_pressed = False
            self._state = EventState.INACTIVE
        elif self.was_pressed and not pressed:
            pressed_time = time.time() - self.start_t
            if self.timeout is None:
                self.was_pressed = False
                self._state = EventState.INACTIVE
                return True
            else:
                if pressed_time > self.timeout:
                    self.was_pressed = False
                    self._state = EventState.INACTIVE
                    return False
                else:
                    self.was_pressed = False
                    self._state = EventState.INACTIVE
                    return True
        return False


class LongPress(ButtonEvent):

    def __init__(self, button_type, which_hand, callback, duration=1.0, priority=1):
        super().__init__(callback, priority)
        self.button_type = button_type
        self.which_hand = which_hand
        self.duration = duration

        # State variables
        self.start_t = None

    def check_event(self, button_data):
        pressed = get_button(self.button_type, self.which_hand, button_data)
        if self.start_t is not None:
            if not pressed:
                pressed_time = time.time() - self.start_t
                if pressed_time > self.duration:
                    self.start_t = None
                    self._state = EventState.INACTIVE
                    return True
                else:
                    self._state = EventState.INACTIVE
                    self.start_t = None
            return False

        if pressed:
            self._state = EventState.ACTIVE
            self.start_t = time.time()
        else:
            self._state = EventState.INACTIVE
            self.start_t = None

        return False


class MultiButtonLongPress(ButtonEvent):
    def __init__(self, buttons, callback, priority=0, duration=0.75, grace_period=0.25):
        super().__init__(callback, priority)
        self.buttons = buttons
        self.duration = duration
        self.grace_period = grace_period
        self.press_times = {
            hand: {button: 0.0 for button in bs} for hand, bs in self.buttons.items()
        }
        self.release_times = {
            hand: {button: 0.0 for button in bs} for hand, bs in self.buttons.items()
        }

    def check_event(self, button_data):
        current_time = time.time()

        # Check press times
        for which_hand, button_types in self.buttons.items():
            for button in button_types:
                pressed = get_button(button, which_hand, button_data)
                if pressed:
                    if self.press_times[which_hand][button] == 0:
                        self.press_times[which_hand][button] = current_time

        # Check release times
        for which_hand, button_types in self.buttons.items():
            for button in button_types:
                if self.press_times[which_hand][button] == 0:
                    continue
                pressed = get_button(button, which_hand, button_data)
                if not pressed:
                    if self.release_times[which_hand][button] == 0:
                        self.release_times[which_hand][button] = current_time

        # Check if all buttons are pressed for the duration
        for which_hand, button_types in self.buttons.items():
            for button in button_types:
                is_released = self.release_times[which_hand][button] != 0
                press_duration = current_time - self.press_times[which_hand][button]
                if is_released and press_duration < self.duration:
                    self._state = EventState.INACTIVE
                    self.reset_times()
                    return False

        # Check if all buttons are released within the grace period
        for which_hand, button_types in self.buttons.items():
            for button in button_types:
                if self.press_times[which_hand][button] == 0:
                    return False
                if self.release_times[which_hand][button] == 0:
                    return False
                self._state = EventState.ACTIVE
                if (
                    current_time - self.release_times[which_hand][button]
                    > self.grace_period
                ):
                    self.press_times[which_hand][button] = 0
                    self.release_times[which_hand][button] = 0
                    self._state = EventState.INACTIVE
                    return False

        # Reset press and release times
        self.reset_times()

        self._state = EventState.INACTIVE
        return True

    def reset_times(self):
        for which_hand, button_types in self.buttons.items():
            for button in button_types:
                self.press_times[which_hand][button] = 0
                self.release_times[which_hand][button] = 0


class SingleClickCycler:
    """Cycles through an iterable using a button"""

    def __init__(
        self,
        event_manager,
        button_type,
        which_hand,
        options,
        next_option_cb: Callable | None = None,
        priority=0,
    ):
        self.event = SingleClick(
            button_type, which_hand, self.on_click, priority, timeout=0.4
        )
        event_manager.add_event(self.event)
        self.options = options
        self.idx = 0
        self.cb = next_option_cb

    def on_click(self, button_data: Dict):
        self.idx = (self.idx + 1) % len(self.options)
        if self.cb:
            self.cb(self.idx, self.get())

    def get(self):
        return self.options[self.idx]
