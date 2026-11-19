


@dataclass
class BotGameConfig:
    """Configuration parameters for the AlgoBot algorithm."""
    
    # Velocity estimation parameters (linear regression: velocity = intercept + slope * time)
    velocity_est_slope: float = 0.1201683
    velocity_est_intercept: float = 5.34245021
    velocity_max_value: float = 12

    # Game mechanics parameters
    jump_distance: float = 60     
    duck_time_limit: float = 0.55  # Maximum time for ducking
    jump_time_threshold = 0.89

    internal_params: dict = field(default_factory=lambda: {
                                                            'barrier_distance_y_coord': 140,   # Y-coordinate for ground obstacle detection
                                                            'barrier_distance_x_coord': 100,    # X-coordinate for ground obstacle detection start
                                                            'bird_distance_y_coord': 98,       # Y-coordinate for flying obstacle detection
                                                            'bird_distance_x_coord': 85,       # X-coordinate for flying obstacle detection start
                                                            'barrier_distance_min_value': 0,   # Minimum distance to trigger jump
                                                            'barrier_distance_max_value': 99   # Maximum distance to trigger jump
                                                            })


class AlgoBot():
    """Bot for playing Chrome Dino game using algorithmic decision making."""
    
    def __init__(self, config: GameConfig, params: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize the algorithmic bot.
        
        Args:
            config (GameConfig): Configuration object with algorithm parameters.
            params (Optional[Dict[str, float]]): Optional custom parameters to override defaults.
        
        Attributes:
            config (GameConfig): Bot configuration.
            params (Dict[str, float]): Active parameters for obstacle detection.
            jump_timer (float): Timestamp of last jump action.
            duck_timer (float): Timestamp of when ducking started.
        """
        self.config = config
        self.jump_timer: float = 0.0
        self.duck_timer: float = 0.0
        
        # Load parameters (custom or default)
        if params:
            try:
                # Validate that all required keys are present
                self.params = {key: params[key] for key in self.config.internal_params.keys()}
            except KeyError:
                print('Warning: Not all keys in params! Using internal params.')
                self.params = self.config.internal_params.copy()
        else:
            self.params = self.config.internal_params.copy()

        self.game_time_prev = 0
        self.dist_prev = 0
        self.vw = 0.8
        self.velocity_estimation = 1

        self.down_counter = 0
        self.base_jump_distance = 250
        self.jump_time = 0.6

        self.r_prev = 0

    def _calc_velocity(self, dist, game_time):
        if (self.game_time_prev == 0) | (self.dist_prev == 0):
            self.game_time_prev = game_time
            self.dist_prev = dist
        else:
            dist_delta = dist - self.dist_prev + 1
            if dist_delta < 0:
                time_delta = game_time - self.game_time_prev
                curent_velocity =  abs(dist_delta) / time_delta
                self.velocity_estimation = self.vw*self.velocity_estimation + (1 - self.vw)*curent_velocity         
            self.game_time_prev = game_time
            self.dist_prev = dist
    
    def _preprocess_img(self, raw: npt.NDArray[np.uint8]) -> Tuple[int, int]:
        """
        Extract obstacle distances from game screenshot.
        
        Args:
            raw (npt.NDArray[np.uint8]): Game area image (height, width, channels).
            
        Returns:
            Tuple[int, int]: (barrier_distance, bird_distance) in pixels.
            
        Algorithm:
            1. For ground obstacles: Scan horizontally from barrier_distance_x_coord
               at barrier_distance_y_coord to find color changes.
            2. For flying obstacles: Scan 130 pixels horizontally from bird_distance_x_coord
               at bird_distance_y_coord to find color changes.
        """
        # Extract ground obstacle distance
        barrier_distance = np.max([
            np.argmin(raw[self.params['barrier_distance_y_coord'], 
                         self.params['barrier_distance_x_coord']:, 0]),  # Find dark obstacles
            np.argmax(raw[self.params['barrier_distance_y_coord'],
                         self.params['barrier_distance_x_coord']:, 0])   # Find light obstacles
        ])
        
        # Extract flying obstacle distance (limited search range)
        bird_distance = np.max([
            np.argmin(raw[self.params['bird_distance_y_coord'], 
                         self.params['bird_distance_x_coord']:self.params['bird_distance_x_coord'] + 130, 0]),
            np.argmax(raw[self.params['bird_distance_y_coord'], 
                         self.params['bird_distance_x_coord']:self.params['bird_distance_x_coord'] + 130, 0])
        ])
        
        return int(barrier_distance), int(bird_distance)
    
    def _post_process_action(self, action: int, velocity: float, start_time: float, r: float, img: npt.NDArray[np.uint8]) -> int:
        """
        Apply timing constraints and update action timers.
        
        Args:
            action (int): Proposed action (0=nothing, 1=jump, 2=duck, 3=stop duck).
            velocity (float): Current game velocity.
            start_time (float): Game start timestamp.
            
        Returns:
            int: Final action after applying constraints.
            
        Logic:
            1. Prevent jumping too frequently based on velocity.
            2. Limit ducking duration to duck_time_limit.
            3. Update timers for jump and duck actions.
        """
        # Calculate time threshold based on game duration
        th = self.config.jump_time_threshold if time.time() - start_time < self.config.jump_distance else 0.0
        jump_time_threshold = self.base_jump_distance / self.velocity_estimation
        
        # Check if we should prevent jumping (cooldown period)
        #if self.down_counter > 0:
        #    action = 2
        #    self.down_counter -= 1
        #    if self.down_counter == 0:
        #       action = 3 
        #       self.jump_timer = 0
        #elif ((time.time() - self.jump_timer) > jump_time_threshold) and ((time.time() - self.jump_timer) < (jump_time_threshold + 0.05)) and (self.duck_timer == 0):
        #    if self.jump_timer != 0:
        #        action = 2
        #        self.down_counter = 8
        #        self.jump_timer = 0
        if ((time.time() - self.jump_timer) <= self.jump_time) and (self.duck_timer == 0): 
            
            #img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            #img = img[100:140, 45:65]
            #if np.mean(img) > 100:
            #    img = (img < 150).astype(np.float16)*255
            #else:
            #    img = (img > 150).astype(np.float16)*255
    
            #d = (np.mean(img, axis=0) > 10).astype(np.int16)
            #if np.sum(d) > 0:
            #    action = 2
            #    print('qqq')
            #    self.duck_timer = time.time()
            #else:
            action = 0

        if self.duck_timer > 0:
            if time.time() - self.duck_timer > self.config.duck_time_limit:
                action = 3  # Stop ducking after time limit
                self.duck_timer = 0.0
            else:
                action = 0  # Continue ducking (no new action needed)
        else:
            # Update timers for new actions
            if action == 1:  # Jump
                self.jump_timer = time.time()
            if action == 2:  # Start ducking
                self.duck_timer = time.time()

        if (time.time() - self.jump_timer) > self.jump_time:
            self.jump_timer = 0
        
        return action
        
    def reset_timers(self) -> None:
        """Reset all action timers to zero."""
        self.jump_timer = 0.0
        self.duck_timer = 0.0
        
    def calc_mean_distance_between_two_barries(self, img):
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        img = img[100:140, 100:]
        if np.mean(img) > 100:
            img = (img < 150).astype(np.float16)*255
        else:
            img = (img > 150).astype(np.float16)*255
            
        #if np.mean(img) > 100:
        #    d = (np.mean(img, axis=0) < 245).astype(np.int16)
        #else:
        d = (np.mean(img, axis=0) > 10).astype(np.int16)
            
        window_size = 5
        for i in range(len(d) - window_size):
            if (d[i] == 1) and (d[i+window_size] == 1):
                if np.sum(d[i:i+window_size+1]) < window_size:
                    for j in range(window_size):
                        d[i+j] = 1
                
        start_pos, end_pos = 0, 0
        pos1, pos2 = 0, 0
        for index, value in enumerate(d):
            if (value == 1) and (start_pos == 0):
                start_pos = index
            elif (value == 0) and (start_pos != 0) and (end_pos==0):
                end_pos = index
                pos1 = index
            elif (value == 1) and (end_pos > 0):
                pos2 = index
                break

        pos1 += 45 if pos1 != 0 else pos1
        pos2 += 45 if pos2 != 0 else pos2
        
        return pos1, pos2

    def select_action(self, img: npt.NDArray[np.uint8], start_time: float) -> int:
        """
        Main decision-making method: selects action based on game state.
        
        Args:
            img (npt.NDArray[np.uint8]): Current game screenshot.
            start_time (float): Game start timestamp.
            
        Returns:
            int: Action code: 0=nothing, 1=jump, 2=duck, 3=stop duck.
            
        Steps:
            1. Extract obstacle distances from image.
            2. Calculate current game velocity.
            3. Adjust detection thresholds based on velocity.
            4. Decide action based on obstacle positions.
            5. Apply timing constraints.
        """
        # Extract obstacle distances
        barrier_distance, bird_distance = self._preprocess_img(img.copy())
        pos1, pos2 = self.calc_mean_distance_between_two_barries(img)
        
        # Calculate current velocity
        self._calc_velocity(barrier_distance, time.time() - start_time)
        velocity = self.velocity_estimation
        if time.time() - start_time < 20:
            velocity = 320 / 0.75
            
        if (pos1 != 0) & (pos2 != 0):
            pos2 = min(pos2, 2*pos1)
            r = int((pos1 + pos2) / 2)
            r = pos1 + 0.2*(pos2 - pos1)
        elif (pos1 != 0) & (pos2 == 0): 
            pos2 = 2*pos1
            r = 0.9*pos1 + 0.1*pos2
            r = pos1 + 50
        else:
            r = 99999
        
        # Adjust detection threshold based on velocity (faster game = larger detection range)
        params_tmp = self.params.copy()
        #params_tmp['barrier_distance_max_value'] = barrier_distance_max_value
        self.r = r
        # Decision logic
        if (r < (0.75*velocity*self.jump_time)) & (bird_distance == 0):
            #print(r, velocity, self.jump_time, velocity*self.jump_time)
            action = 1  # Jump over ground obstacle
        elif (bird_distance != 0) and (barrier_distance == 0):
            action = 2  # Duck under flying obstacle
        else:
            action = 0  # No action needed

        # Apply timing constraints and update timers
        action = self._post_process_action(action, velocity, start_time, r, img)
        self.r_prev = r

        res_dict = {}
        res_dict['action'] = action
        res_dict['barrier_distance'] = barrier_distance
        res_dict['bird_distance'] = bird_distance
        res_dict['pos1'] = pos1
        res_dict['pos2'] = pos2
        res_dict['velocity'] = velocity
        res_dict['jump_time'] = time.time() - self.jump_timer if self.jump_timer != 0 else 0
        res_dict['duck_timer'] = time.time() - self.duck_timer if self.duck_timer != 0 else 0

        return res_dict