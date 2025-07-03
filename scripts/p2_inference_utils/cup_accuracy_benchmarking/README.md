# Cup Accuracy Benchmarking Suite

This folder contains a set of scripts designed to perform cup grasp accuracy experiments. This document outlines the file structure, configuration options, key functionality, and instructions on how to run the tests.

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Running the Code](#running-the-code)
4. [Command Line Arguments & User Inputs](#command-line-arguments--user-inputs)
5. [Logging and Output](#logging-and-output)
6. [File Descriptions](#file-descriptions)
   - [data_classes.py](#dataclassespy)
   - [single_test_script.py](#single_test_scriptpy)
   - [keyboard_yaw_control.py](#keyboard_yaw_controlpy) 


---

## Overview

The objective is to determine the positional and angular tolerance of various cup designs during pick attempts. In other words, the goal is to quantify how far off in XYZ and in angular orientation the grasp can be while still achieving a secure suction seal and lift. The code allows for simple configuration through YAML as well as interactive adjustments via keyboard controls. 

---
## Configuration

- **YAML File (`config.yaml`):**  
  Modify this file to adjust experiment parameters. For example:
  ```yaml
  test_config:
    sampling_strategy: "grid"
    x_range: [-0.01, 0.00, 0.001]
    y_range: [0.065, 0.00, -0.001]
    z_range: [0.03, 0.00, -0.001]
    roll_angles: [-45.0, 0.0, 5.0]
    pitch_angles: [45.0, 0, -5.0]
    yaw_angles: [-15.0, 0, 5.0]
    axis_combinations:
      - ["x"]
      - ["y"]
      - ["z"]
      - ["roll"]
      - ["pitch"]
      - ["yaw"]
      - ["x", "y"]
    approach_settings:
      use_vertical_approach: true
      approach_height: 0.1
      approach_velocity: 0.3
      approach_acceleration: 0.3

### YAML Configuration Details

- **Sampling Strategy:**  
  Currently, only the "grid" sampling strategy is implemented.

- **Positional Ranges:**  
  The `x_range`, `y_range`, and `z_range` values are specified in meters.

- **Angular Ranges:**  
  The `roll_angles`, `pitch_angles`, and `yaw_angles` values are in degrees.

- **Axis Combinations:**  
  This setting defines which axes are sampled together (e.g., `["x", "y"]` means both x and y adjustments are varied simultaneously).

- **Approach Settings:**  
  These control a vertical approach to help avoid collisions when adjusting adjustments (especially for the side cup). Key parameters include:  
  - `use_vertical_approach`: Enables vertical movement above the object  
  - `approach_height`: The height (in meters) above the object  
  - `approach_velocity` and `approach_acceleration`: Movement speed and acceleration during vertical motion

---
## Running the Code

Execute the single_test_script.py to perform an individual cup-object test:

```
# Run with default config
python3 single_test_script.py
```

```
# Run with custom config
python3 single_test_script.py --config-yaml path/to/your/config.yaml
```

---
## Command Line Arguments & User Inputs

**Command Line Arguments:**  
- The testing suite uses a YAML configuration file to define test parameters. You can either:
    - Use the default config at `cup_accuracy_benchmarking/config.yaml`
    - Specify your own config file path

**User Inputs:**  
- **Cup Configuration:**  
  At runtime, you will be prompted to enter:
  - Cup ID (Takes an integer value like for e.g., 1,2, or a part number like 110345)
  - Cup type (e.g., `line`, `side`, or `pinch`)  
  - Cup diameter (in mm)  
  - Vacuum threshold (in PSI)  

- **Test Object Configuration:**  
  You will be asked to provide:
  - Object ID  (Takes an integer value like for e.g., 1,5 etc)
  - Object type (e.g., `box`, `pencil`, `dvd_case`)  

---

## Logging and Output

The CSV file records one row per individual pick attempt with the following parameters:

- **timestamp:** The start time (in seconds since epoch) when the attempt began.
- **cup_id:** The identifier for the suction cup used.
- **object_id:** The identifier for the test object ID.
- **object_type:** The identifier for the test object type.
- **adjustment_x_mm, adjustment_y_mm, adjustment_z_mm:** Positional adjustment applied during the pick attempt (in millimeters).
- **roll_deg, pitch_deg, yaw_deg:** Angular adjustment applied (in degrees).
- **grasp_pressure_psi:** The vacuum pressure (in PSI) measured at the grasping step.
- **lift_pressure_psi:** The vacuum pressure (in PSI) measured after lifting the object.
- **grasp_success:** A Boolean flag indicating whether the grasp was successfully achieved.
- **lift_success:** A Boolean flag indicating whether the object was successfully lifted while maintaining a vacuum seal.

### Terminal Outputs

Throughout the run of the experiment, the script prints status messages to the terminal to provide real-time insight into the process, including:

- **Test Point Information:**  
  The current computed adjustments are printed prior to the movement command.
  
- **Pose Adjustments and Movements:**  
  Notifications when the robot moves to the adjusted pose, approaches the object, and when it returns to the home position.
  
- **Vacuum Seal Status:**  
  - A message indicating that the vacuum seal has been achieved, along with the corresponding pressure reading.  
  - A failure message if the required vacuum pressure threshold is not met.
  
- **Valve Activation:**  
  Messages indicating when the HFG valve is opened and closed during the pick attempt.
  
- **General Status:**  
  Any errors or exceptions encountered during a pick attempt.

These logging details ensure that you have both a persistent record in the CSV file for post-experiment analysis and real-time feedback in the terminal during testing.

---
## File Descriptions

### single_test_script.py

- **Purpose:**  
  Executes a single pick test. This script:
  - Loads parameters about adjustments and about the object being picked (called the test object).
  - Applies positional and angular adjustments to generate adjusted poses.
  - Commands the robot to move to the adjusted pose, enables vacuum sealing, checks for a seal, lifts the object, and then returns the outcome.
  - Logs intermediate and final results.

- **Key Functions:**
  - `wait_for_vacuum_seal(threshold: float, timeout_s: float) -> Tuple[bool, float]`:  
    Polls the vacuum sensor until the pressure drops below a threshold or a timeout is reached.
  - `run_pick_attempt(params: PickParameters, config: TestConfig, obj: TestObject) -> Tuple[bool, bool, float, float]`:  
    Executes the pick attempt and returns a tuple with grasp success, lift success, and corresponding vacuum pressures.
  - `execute_and_log_test(...)`:  
    Integrates the pick attempt execution and result logging.
### data_classes.py

- **Purpose:**  
  Contains all the data structures used across the experiments.
  
- **Key Classes:**
  - **TestConfig:**  
    Stores configuration parameters such as sampling strategy (currently only grid-based sampling implemented), ranges for position adjustments
 (X, Y, Z) and angles (roll, pitch, yaw), random seed, and approach settings.  
    - Provides a class method `from_yaml(yaml_path: str)` to load these parameters from a YAML file.
  - **ApproachSettings:**  
    Stores parameters related to vertical motions (e.g., use_vertical_approach, approach_height, velocity, and acceleration).
  - **TestObject:**  
    Defines the test object parameters, including its ID and type.
  - **PickParameters:**  
    Contains the adjustments
 and threshold parameters for a pick attempt.
  - **CupConfig:**  
    Configures cup-specific parameters like cup ID, type, diameter, vacuum threshold, and compatible test objects.
  - **TestResult:**  
    Captures the results from a single pick attempt (e.g., timestamp, applied adjustments
, vacuum pressure readings, and success flags).


### keyboard_yaw_control.py

- **Purpose:**  
  Lets the user change the gripper’s yaw manually via keyboard.
  
- **Key Components:**
  - **KeyboardYawController:**  
    Handles non-blocking keyboard inputs to increment/decrement the gripper’s yaw angle.
    - Press **'a'** to rotate counter-clockwise.
    - Press **'d'** to rotate clockwise.
    - Press **'q'** to exit interactive mode.
  - **interactive_yaw_control(hfg_agent):**  
    Helper function that starts the interactive yaw control process and returns the final yaw angle.


