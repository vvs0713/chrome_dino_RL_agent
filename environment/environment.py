import numpy as np
import pyautogui
import cv2
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.window import WindowTypes 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from mss import mss
from typing import Tuple, Dict, Optional
import numpy.typing as npt


class EnvClass():
    """Class for interacting with Chrome Dino game using Selenium and computer vision."""
    
    def __init__(self) -> None:
        """
        Initializes the Chrome Dino game environment.
        
        Attributes:
            cap (mss): Screenshot capture object.
            game_location (Dict[str, int]): Coordinates of the game area on screen.
            actions_dict (Dict[int, str]): Dictionary mapping action codes to keys.
            game_over_area (List): [x_min, y_min, x_max, y_max]
            game_over_area_constants (List): Constants for setting game over flag
            done_flag (bool): Game completion flag.
            driver (Optional[webdriver.Chrome]): Chrome browser driver.
        """
        self.cap = mss()
        self.game_location = {'top': 230, 'left': 220, 'width': 620, 'height': 170}
        self.actions_dict = {0: 'nothing', 1: 'up', 2: 'down', 3: 'down'}
        self.done_flag = False
        self.driver: Optional[webdriver.Chrome] = None
        self.coord_ratio = (0.03888888888888889, 0.03958333333333333)
        self.size_ratio = (0.9518518518518518, 0.475)
        
        self.original_screen_width = 1920
        self.original_screen_height = 1080
        self.original_canvas_width = 600
        self.original_canvas_height = 150
        
    def init_environment(self) -> None:
        """
        Initializes the game environment: opens browser and launches Dino game.
        
        Opens Chrome browser and navigates to the Dino game page, then clicks
        to focus the game window.
        
        Raises:
            WebDriverException: If there's an issue with Chrome driver.
        """

        screen_width, screen_height = self.original_canvas_width, self.original_canvas_height
        
        try:
            options = Options()
            options.add_argument("--no-sandbox")
            options.add_argument("--kiosk")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--ozone-platform=x11")
            options.add_argument("--force-device-scale-factor=1")
            options.add_argument("--high-dpi-support=1")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)

            self.driver = webdriver.Chrome(options=options)
            self.driver.command_executor.set_timeout(1000)

            screen_width = self.driver.execute_script("return screen.width;")
            screen_height = self.driver.execute_script("return screen.height;")

            self.driver.get("chrome://dino")
        
        except WebDriverException as e:
            pass

        self.rescale_canvas_param(screen_width, screen_height)
        self._canvas_coords()
        pyautogui.click(200, 200) #activate Chrome
        
    def stop_environment(self) -> None:
        """
        Stops the environment: closes the browser.
        
        Closes the Chrome browser and quits the WebDriver.
        """
        if self.driver:
            self.driver.quit()
        
    def start(self) -> None:
        """
        Starts the game by pressing space.
        
        Sends a space key press to start the Dino game.
        """
        pyautogui.press('space')
        self.done_flag = False
    
    def refresh(self) -> None:
        """
        Refreshes the game page.
        
        Refreshes the browser page to reset the game state.
        """
        if self.driver:
            self.driver.refresh()
        pyautogui.keyUp('down', _pause=False)
        self.done_flag = False
        
    def apply_action(self, action: int) -> None:
        """
        Applies the specified action in the game.
        
        Args:
            action (int): Action code from actions_dict:
                0: do nothing
                1: jump (space)
                2: duck (hold down)
                3: stop duck (release down)
                
        Note:
            Uses _pause=False to prevent pyautogui delays.
        """
        if action == 1:
            pyautogui.press('space', _pause=False)
        elif action == 2:
            pyautogui.keyDown(self.actions_dict[action], _pause=False)
        elif action == 3:
            pyautogui.keyUp(self.actions_dict[action], _pause=False)
        else:
            pass  # action == 0: do nothing
            
    def check_done(self) -> None:
        if self.driver is None:
            self.done_flag = False
            return
        try:
            crashed = self.driver.execute_script("""
                if (typeof Runner !== 'undefined' && typeof Runner.getInstance === 'function') {
                    return Runner.getInstance().crashed;
                }
                return false;
            """)
            self.done_flag = bool(crashed)
        except Exception:
            self.done_flag = False

    def rescale_canvas_param(self, screen_width, screen_height):
        #self.k = np.sqrt((self.original_screen_width*self.original_screen_height)/(screen_width*screen_height))
        self.k = (self.original_screen_width)/(screen_width)


    def _canvas_coords(self):
        canvas_element = self.driver.find_element(By.TAG_NAME, "canvas")
        canvas_rect = canvas_element.rect
        self.canvas_width = canvas_rect['width']   # Текущая ширина поля в пикселях
        self.canvas_height = canvas_rect['height'] # Текущая высота поля в пикселях
        
        location = canvas_element.location_once_scrolled_into_view
        self.canvas_x = location['x']
        self.canvas_y = location['y']

        #self.k = ((self.original_canvas_width*self.original_canvas_height)/(self.canvas_width*self.canvas_height))
        

    def stop_fullscreen_mode(self):
        pyautogui.hotkey('alt', 'f4')
        
    def grab_image(self) -> npt.NDArray[np.uint8]:
        """
        Captures the game area image.
        
        Returns:
            npt.NDArray[np.uint8]: Three-channel game area image 
            with dimensions (height, width, 3) and uint8 type.
            
        Steps:
            1. Captures screenshot of entire screen.
            2. Crops to game area.
            3. Checks game state.
        """

        # Capture screenshot of entire screen
        raw = np.array(self.cap.grab(self.cap.monitors[1]))

        # Convert to RGB and resize
        raw = np.array(raw)[:, :, :3].astype(np.uint8)

        raw = raw[self.canvas_y : self.canvas_y + self.canvas_height,
                  self.canvas_x : self.canvas_x + self.canvas_width, :]
        
  
        #raw = cv2.resize(raw, (620, 170), interpolation=cv2.INTER_NEAREST)
        
        # Check if game has ended
        self.check_done()
        
        return raw