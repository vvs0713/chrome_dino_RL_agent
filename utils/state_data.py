import numpy as np
import numpy.typing as npt
import time
from dataclasses import dataclass


@dataclass
class MakeStateDataConfig:
    """Configuration parameters for the MakeStateData class."""
    
    barrier_distance_y_coord: int = 123     # Y-coordinate used to determine distance to the barrier.
    barrier_distance_x_coord: int = 100     # X-coordinate used to determine distance to the barrier.
    bird_distance_y_coord: int = 87         # Y-coordinate used to determine distance to the bird.
    bird_distance_x_coord: int = 85         # X-coordinate used to determine distance to the bird.
    
    day_night_values_thresh: int = 100      # Threshold value for determining day/night on the image.
    meaning_values_thresh: int = 150        # Threshold value for determining significant pixels on the image.
    barrier_height_y_min_coord: int = 80    # Y-coordinate defining the topmost point of the barrier.
    barrier_height_y_max_coord: int = 120   # Y-coordinate defining the lowest point of the barrier.
    barrier_x_max_coord: int = 140          # X-coordinate from which barrier detection starts.
    barrier_pixel_values_thresh: int = 10   # Threshold for determining significant pixels after image standardization (likely duplicates the previous value).
    dino_coord_shift: int = 55              # Offset of the largest dinosaur coordinate.
    window_size: int = 5                    # Window size for morphological gap-filling of the barrier mask.
    
    game_time_prev: float = 0               # Previous game time value, used to calculate game speed.
    dist_prev: int = 0                      # Previous distance to the barrier value, used to calculate game speed.
    vw: float = 0.999                       # Weight coefficient for exponential smoothing of velocity (weight of the current value).
    velocity_estimation: float = 150        # Initial game speed value.
    jump_time: float = 0.6                  # Dinosaur jump duration; constant throughout the game.

    vel_min_time_thresh: float = 0.015      # Threshold for velocity calculation.

class MakeStateData:
    """Builds a dictionary of game state variables from a raw frame."""

    def __init__(self, params: MakeStateDataConfig) -> None:
        self.params: MakeStateDataConfig = params
        self.game_time_prev: float = self.params.game_time_prev
        self.dist_prev: int = self.params.dist_prev
        self.velocity_estimation: float = self.params.velocity_estimation
        self.duck_timer: float = 0
        self.jump_timer: float = 0

    def _calc_velocity(self, dist: int, game_time: float) -> None:
        """Estimate game velocity from barrier distance changes over time (exponential smoothing)."""
        if (self.game_time_prev == 0) | (self.dist_prev == 0):
            self.game_time_prev = game_time
            self.dist_prev = dist
        else:
            time_delta = game_time - self.game_time_prev
            if time_delta > self.params.vel_min_time_thresh:
                dist_delta = dist - self.dist_prev + 1
                if dist_delta < 0:
                    current_velocity = abs(dist_delta) / time_delta  # was: curent_velocity
                    self.velocity_estimation = (
                        self.params.vw * self.velocity_estimation
                        + (1 - self.params.vw) * current_velocity
                    )
                self.game_time_prev = game_time
                self.dist_prev = dist


    def _calc_distance(self, raw: npt.NDArray[np.uint8]) -> tuple[int, int]:
        """Return (barrier_distance, bird_distance) in pixels by scanning fixed rows."""
        # Extract ground obstacle distance
        barrier_distance = np.max([
            np.argmin(raw[self.params.barrier_distance_y_coord,
                          self.params.barrier_distance_x_coord:, 0]),  # Find dark obstacles
            np.argmax(raw[self.params.barrier_distance_y_coord,
                          self.params.barrier_distance_x_coord:, 0])   # Find light obstacles
        ])
        
        # Extract flying obstacle distance (limited search range)
        bird_distance = np.max([
            np.argmin(raw[self.params.bird_distance_y_coord,
                          self.params.bird_distance_x_coord:self.params.bird_distance_x_coord + 130, 0]),
            np.argmax(raw[self.params.bird_distance_y_coord,
                          self.params.bird_distance_x_coord:self.params.bird_distance_x_coord + 130, 0])
        ])
        
        return int(barrier_distance), int(bird_distance)

    def update_timers(self, action: int) -> None:
        """Update jump/duck timers based on the chosen action (1 = jump, 2 = duck, 3 = stop ducking)."""
        if self.duck_timer > 0:
            if action == 3:
                self.duck_timer = 0.0
        else:
            # Update timers for new actions
            if action == 1:  # Jump
                self.jump_timer = time.time()
            if action == 2:  # Start ducking
                self.duck_timer = time.time()

        if (time.time() - self.jump_timer) > self.params.jump_time:
            self.jump_timer = 0

    def _reset_timers(self) -> None:
        """Reset all action timers to zero."""
        self.jump_timer = 0.0
        self.duck_timer = 0.0

    def _calc_mean_distance_between_two_barriers(self, img: npt.NDArray[np.uint8]) -> tuple[int, int]:
        """Return start/end positions (in dinosaur-relative coordinates) of the nearest gap between two barriers.

        Note: renamed from `_calc_mean_distance_between_two_barries` (typo fix).
        """
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        img = img[self.params.barrier_height_y_min_coord : self.params.barrier_height_y_max_coord,
                  self.params.barrier_distance_x_coord:]
        if np.mean(img) > self.params.day_night_values_thresh:
            img = (img < self.params.meaning_values_thresh).astype(np.float16) * 255
        else:
            img = (img > self.params.meaning_values_thresh).astype(np.float16) * 255

        d = (np.mean(img, axis=0) > self.params.barrier_pixel_values_thresh).astype(np.int16)

        # Fill short gaps in the mask so that a barrier split by artifacts is treated as solid
        for i in range(len(d) - self.params.window_size):
            if (d[i] == 1) and (d[i + self.params.window_size] == 1):
                if np.sum(d[i:i + self.params.window_size + 1]) < self.params.window_size:
                    for j in range(self.params.window_size):
                        d[i + j] = 1

        start_pos, end_pos = 0, 0
        pos1, pos2 = 0, 0
        for index, value in enumerate(d):
            if (value == 1) and (start_pos == 0):
                start_pos = index
            elif (value == 0) and (start_pos != 0) and (end_pos == 0):
                end_pos = index
                pos1 = index
            elif (value == 1) and (end_pos > 0):
                pos2 = index
                break

        pos1 += (self.params.barrier_distance_x_coord - self.params.dino_coord_shift) if pos1 != 0 else pos1
        pos2 += (self.params.barrier_distance_x_coord - self.params.dino_coord_shift) if pos2 != 0 else pos2

        return pos1, pos2

    def calc_state_variables(self, img: npt.NDArray[np.uint8], start_time: float, resolution_scale_factor: int = 1) -> dict[str, float]:
        """Compute and return a dict of state variables describing the current game frame."""
        # Extract obstacle distances
        barrier_distance, bird_distance = self._calc_distance(img.copy())
        pos1, pos2 = self._calc_mean_distance_between_two_barriers(img.copy())  # renamed call


        #barrier_distance = barrier_distance * resolution_scale_factor
        #bird_distance = bird_distance * resolution_scale_factor
        #pos1 = pos1 * resolution_scale_factor
        #pos2 = pos2 * resolution_scale_factor

        # Calculate current velocity
        self._calc_velocity(barrier_distance, time.time() - start_time)

        res_dict: dict[str, float] = {}
        res_dict['barrier_distance'] = barrier_distance
        res_dict['bird_distance'] = bird_distance
        res_dict['pos1'] = pos1
        res_dict['pos2'] = pos2
        res_dict['velocity'] = self.velocity_estimation / resolution_scale_factor
        res_dict['jump_time'] = time.time() - self.jump_timer if self.jump_timer != 0 else 0
        res_dict['duck_timer'] = time.time() - self.duck_timer if self.duck_timer != 0 else 0
        res_dict['game_time'] = time.time() - start_time

        return res_dict