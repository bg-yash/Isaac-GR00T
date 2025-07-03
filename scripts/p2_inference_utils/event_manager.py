from enum import Enum, auto
from typing import Callable, Dict


class EventState(Enum):
    """A class to define the state of an event"""

    # Inactive means the event has not started
    INACTIVE = auto()

    # Active means the event has started, and we have not yet determined if it matches the event criteria
    ACTIVE = auto()


class ButtonEvent:
    """A class to define/detect button events"""

    def __init__(self, callback, priority=0):
        self.callback = callback
        self.priority = priority
        self._state = EventState.INACTIVE

    def check_event(self, input_state):
        """
        Should return True if the event has occurred, False otherwise.
        If returning True, lower priority events which conflict with this one should not be executed.

        """
        raise NotImplementedError

    @property
    def event_state(self):
        return self._state

    def execute(self, input_state: Dict):
        if self.callback:
            self.callback(input_state)

    def append_callback(self, new_cb: Callable):
        """Call the new callback after the old callback (if it exists)"""
        old_cb = self.callback

        def _combined_cb(*args, **kwargs):
            if old_cb:
                old_cb(*args, **kwargs)
            new_cb(*args, **kwargs)

        self.callback = _combined_cb

    def prepend_callback(self, new_cb: Callable):
        """Call the new callback before the old callback (if it exists)"""
        old_cb = self.callback

        def _combined_cb(*args, **kwargs):
            new_cb(*args, **kwargs)
            if old_cb:
                old_cb(*args, **kwargs)

        self.callback = _combined_cb


class ButtonEventManager:
    """
    Controls how events are processed and executed, allowing them to have priorities.
    This allows us to have multiple similar events (press A, press A&B) without them stomping all over each other.
    """

    def __init__(self):
        self.events = []
        self.highest_priority_event = None

    def add_event(self, event):
        self.events.append(event)
        self.events.sort(key=lambda e: e.priority)

    def process_events(self, input_state):
        for event in self.events:
            if event.check_event(input_state):
                self.highest_priority_event = event

        if self.highest_priority_event is not None:
            if all(event.event_state == EventState.INACTIVE for event in self.events):
                self.highest_priority_event.execute(input_state)
                self.highest_priority_event = None
