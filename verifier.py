# pylint: disable=broad-exception-caught, too-many-locals, too-many-statements
"""
Part 4: Metadata Verification GUI

Provides a graphical user interface (Tkinter) to randomly sample
and visually verify the correctness of generated NFT metadata
against their corresponding images. Allows users to 'Verify'
or 'Reject' pairs, logs rejections, and generates an analysis 
report highlighting potentially problematic traits or values.
"""
import os
import sys
import json
import random
import logging
import socket
import subprocess
import tkinter as tk
import urllib.request
import urllib.error
from io import BytesIO
from collections import Counter
from tkinter import ttk, messagebox
from typing import List, Dict, Optional
from PIL import Image, ImageTk, UnidentifiedImageError

BASE_DIR = "C:/bedlam/md_gen_v4"
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
MAIN_LOG_FILE = os.path.join(LOG_DIR, "verify_metadata.log")
REJECT_LOG_FILE = os.path.join(LOG_DIR, "verify_rejections.log")
ANALYSIS_LOG_FILE = os.path.join(LOG_DIR, "verify_analysis.log")
IMAGE_JSON_PATH = os.path.join(CONFIG_DIR, "images.json")
TOKEN_START = 1295
TOKEN_END = 10000
SAMPLE_SIZE = 100
IMAGE_DISPLAY_SIZE = (400, 400)
REQUEST_TIMEOUT = 20
INITIAL_WINDOW_WIDTH = 750
INITIAL_WINDOW_HEIGHT = 800

try:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
except OSError as e:
    print(f"CRITICAL: Error creating directories: {e}", \
        file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - \
        [%(funcName)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(MAIN_LOG_FILE, mode='a', \
            encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()
def load_metadata(token_id: int) -> Optional[Dict]:
    """
    Loads metadata JSON from the metadata 
    directory for a given token ID.
    Args:
        token_id: The integer ID of the token to 
        load metadata for.
    Returns:
        A dictionary containing the loaded metadata, 
        or None if loading fails.
    """
    metadata_path = os.path.join(METADATA_DIR, str(token_id))
    if not os.path.exists(metadata_path):
        logger.error("Metadata file not found for \
            token ID %d at %s",
                     token_id, metadata_path)
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON format in %s for \
            token ID %d: %s",
                     metadata_path, token_id, e)
        return None
    except (OSError, IOError) as e:
        logger.error("OS error reading metadata file %s for \
            token ID %d: %s",
                     metadata_path, token_id, e)
        return None
    except Exception as e:
        logger.error("Unexpected error loading metadata for \
            token ID %d: %s",
                     token_id, e, exc_info=True)
        return None

def load_image_from_url(url: str) -> Optional[Image.Image]:
    """
    Downloads and opens an image from a URL.
    Args:
        url: The URL string of the image to download.
    Returns:
        A PIL Image object if successful, otherwise None.
    """
    if not url:
        logger.error("No URL provided for image download.")
        return None
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) \
            as response:
            if response.status != 200:
                logger.error("Failed to load image from URL %s: \
                    HTTP Status %d",
                             url, response.status)
                return None
            image_data = response.read()
            with BytesIO(image_data) as image_stream:
                img = Image.open(image_stream).convert("RGB")
                return img
    except urllib.error.HTTPError as e:
        logger.error("HTTPError loading image from URL %s: %s - %s",
                     url, e.code, e.reason)
        return None
    except urllib.error.URLError as e:
        reason = e.reason if isinstance(e.reason, str) \
            else getattr(e.reason, 'strerror', str(e.reason))
        if isinstance(e.reason, socket.timeout) or 'timed out' \
            in reason.lower():
            logger.error("Timeout loading image from URL %s \
                after %d seconds",
                         url, REQUEST_TIMEOUT)
        else:
            logger.error("URLError loading image from URL %s: \
                %s", url, reason)
        return None
    except socket.timeout:
        logger.error("Socket timeout loading image from URL %s \
            after %d seconds",
                     url, REQUEST_TIMEOUT)
        return None
    except UnidentifiedImageError:
        logger.error("Cannot identify image file from URL %s \
            (invalid format?)", url)
        return None
    except (OSError, IOError, ValueError) as e:
        logger.error("OS/IO/Value error processing image from URL %s: \
            %s", url, e)
        return None
    except Exception as e:
        logger.error("Unexpected error loading image from URL %s: \
            %s",
                     url, e, exc_info=True)
        return None

def format_metadata_display(metadata: Optional[Dict]) -> str:
    """
    Formats the full metadata dictionary into a 
    pretty-printed JSON string.
    Args:
        metadata: The dictionary containing the 
        token's metadata.

    Returns:
        A formatted string representation of 
        the metadata, or an error message.
    """
    if not metadata:
        return "Error loading metadata."
    try:
        if 'attributes' in metadata and \
            isinstance(metadata['attributes'], list):
            pass
        return json.dumps(metadata, indent=2)
    except Exception as e:
        logger.error("Error formatting metadata for display: \
            %s", e)
        return f"Error formatting metadata:\n{str(metadata)}"

def open_file(filepath: str):
    """
    Opens a file using the system's default 
    application in a cross-platform way.
    Args:
        filepath: The absolute or relative path to the file to open.
    """
    try:
        if not os.path.exists(filepath):
            logger.warning("Cannot open file: File does not exist \
                at %s", filepath)
            messagebox.showwarning("File Not Found", \
                f"Could not find file:\n{filepath}")
            return
        if sys.platform == "win32":
            os.startfile(os.path.normpath(filepath))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", filepath])
        else:
            subprocess.Popen(["xdg-open", filepath])
        logger.info("Attempted to open file: %s", filepath)
    except Exception as e:
        logger.error("Failed to open file %s: %s", filepath, e)
        messagebox.showerror("Error Opening File", \
            f"Could not open file:\n{filepath}\n\nError: {e}")

class MetadataVerifierApp:
    """
    Main application class for the Tkinter-based metadata verification GUI.

    Handles UI setup, data loading, user interactions (verify/reject),
    and manages the verification workflow from start to completion analysis.
    """
    BG_COLOR = "#ECECEC"
    TEXT_COLOR = "#333333"
    HEADER_COLOR = "#1E3A5F"
    BUTTON_COLOR = "#F0F0F0"
    REJECT_COLOR = "#D32F2F"
    VERIFY_COLOR = "#388E3C"

    def __init__(self, master: tk.Tk):
        """
        Initializes the application, sets up styles, frames, and widgets.

        Args:
            master: The main Tkinter root window (often called root).
        """
        self.master = master
        self.master.title("Metadata Verifier")
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x_coord = int((screen_width / 2) - (INITIAL_WINDOW_WIDTH / 2))
        y_coord = int((screen_height / 2) - (INITIAL_WINDOW_HEIGHT / 2))
        x_coord = max(0, x_coord)
        y_coord = max(0, y_coord)
        self.master.geometry(f"{INITIAL_WINDOW_WIDTH}x \
            {INITIAL_WINDOW_HEIGHT}+{x_coord}+{y_coord}")
        self.master.minsize(600, 650)

        self.sampled_token_ids: List[int] = []
        self.rejected_token_ids: List[int] = []
        self.current_index: int = -1
        self.current_pil_image: Optional[Image.Image] = None
        self.current_metadata: Optional[Dict] = None
        self.img_tk: Optional[ImageTk.PhotoImage] = None
        self.image_map: Optional[Dict] = None

        self._load_image_map()
        if self.image_map is None:
            messagebox.showerror("Fatal Error",
                                  f"Could not load image map \
                                      from:\n{IMAGE_JSON_PATH}\n\n"
                                  "Please check the file exists \
                                      and is valid JSON.\n"
                                  "See log file for details.")
            logger.critical("Exiting due to failed image map load.")
            self.master.after(100, self.master.destroy)
            return

        self._setup_styles()
        self._setup_frames()
        self._setup_start_widgets()
        self._setup_progress_widgets()
        self._setup_verify_widgets()
        self._setup_complete_widgets()

        self.image_label.bind('<Configure>', self._on_resize)

        self.start_frame.pack(fill=tk.BOTH, expand=True)

    def _load_image_map(self):
        """Loads the 
            images.json
             file into
              self.image_map.
        """
        logger.info("Attempting to load image map from \
            %s", IMAGE_JSON_PATH)
        try:
            with open(IMAGE_JSON_PATH, 'r', encoding='utf-8') as f:
                self.image_map = json.load(f)
            if not isinstance(self.image_map, dict):
                logger.error("Image map loaded from %s \
                    is not a dictionary.", IMAGE_JSON_PATH)
                self.image_map = None
            else:
                logger.info("Successfully loaded image map with \
                    %d entries.", len(self.image_map))
        except FileNotFoundError:
            logger.error("Image map file not found: %s", IMAGE_JSON_PATH)
            self.image_map = None
        except json.JSONDecodeError as e:
            logger.error("Error decoding JSON from image map file %s: \
                %s", IMAGE_JSON_PATH, e)
            self.image_map = None
        except (OSError, IOError) as e:
            logger.error("OS error reading image map file %s: \
                %s", IMAGE_JSON_PATH, e)
            self.image_map = None
        except Exception as e:
            logger.error("Unexpected error loading image map %s: %s",
                         IMAGE_JSON_PATH, e, exc_info=True)
            self.image_map = None

    def _setup_styles(self):
        """
        Configures
         ttk styles
         for the
         application.
        """
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.master.configure(bg=self.BG_COLOR)

        self.style.configure("TFrame", background=self.BG_COLOR)
        self.style.configure("TLabel", padding=6, font=('Segoe UI', 10),
                             background=self.BG_COLOR, \
                                 foreground=self.TEXT_COLOR)
        self.style.configure("Header.TLabel", font=('Segoe UI', 14, 'bold'),
                             foreground=self.HEADER_COLOR, \
                                 background=self.BG_COLOR)
        self.style.configure("Status.TLabel", font=('Segoe UI', 11),
                             background=self.BG_COLOR, \
                                 foreground=self.TEXT_COLOR)
        self.style.configure("TButton", padding=(10, 5), \
                             font=('Segoe UI', 10, 'bold'),
                                 background=self.BUTTON_COLOR, \
                                     relief="raised")
        self.style.map("TButton", background=[('active', '#E0E0E0')])

        self.style.configure("TProgressbar", thickness=20)

        self.style.configure("Reject.TButton", foreground="white",
                             background=self.REJECT_COLOR)
        self.style.map("Reject.TButton", background=[('active', '#C62828')])

        self.style.configure("Verify.TButton", foreground="white",
                             background=self.VERIFY_COLOR)
        self.style.map("Verify.TButton", background=[('active', '#2E7D32')])

        self.style.configure("Close.TButton", foreground=self.TEXT_COLOR,
                             background="#D0D0D0")
        self.style.map("Close.TButton", background=[('active', '#BDBDBD')])


    def _setup_frames(self):
        self.start_frame = ttk.Frame(self.master, padding="30")
        self.progress_frame = ttk.Frame(self.master, padding="30")
        self.verify_frame = ttk.Frame(self.master, padding="15")
        self.complete_frame = ttk.Frame(self.master, padding="30")

    def _setup_start_widgets(self):
        """
        Sets up widgets
        for the initial
        start screen.
        """
        self.start_button = ttk.Button(self.start_frame, \
            text="Start Verifying", command= \
                self._start_verification, width=20)
        self.start_button.pack(expand=True, ipadx=10, ipady=5)

    def _setup_progress_widgets(self):
        """
        Sets up widgets
        for the progress
        bar screen
        during sampling.
        """
        self.progress_label = \
            ttk.Label(self.progress_frame,
                text="Sampling metadata files...",
                    style="Status.TLabel")
        self.progress_label.pack(pady=(20, 10))
        self.progress_bar = ttk.Progressbar(self.progress_frame, \
            orient='horizontal', mode='determinate', length=400)
        self.progress_bar.pack(pady=10, padx=20)

    def _setup_verify_widgets(self):
        """
         Sets up widgets for
         the main verification
         screen (image,
         metadata, buttons).
        """
        self.verify_frame.columnconfigure(0, weight=1, minsize=20)
        self.verify_frame.columnconfigure(1, weight=5)
        self.verify_frame.columnconfigure(2, weight=1, minsize=20)
        self.verify_frame.rowconfigure(0, weight=0, minsize=40)
        self.verify_frame.rowconfigure(1, weight=3)
        self.verify_frame.rowconfigure(2, weight=2)
        self.verify_frame.rowconfigure(3, weight=0, minsize=60)

        self.info_frame = ttk.Frame(self.verify_frame, padding=(0, 5))
        self.info_frame.grid(row=0, column=1, sticky="nsew")
        self.token_id_label = ttk.Label(self.info_frame, \
            text="Token ID: ---", style="Header.TLabel")
        self.token_id_label.pack(side=tk.LEFT, padx=(0, 20))
        self.index_label = ttk.Label(self.info_frame, \
            text="Sample: - / -", style="Status.TLabel")
        self.index_label.pack(side=tk.RIGHT)

        self.image_label = tk.Label(self.verify_frame, \
            text="Loading Image...", anchor=tk.CENTER,
            background="#DDDDDD", relief="groove", borderwidth=1)
        self.image_label.grid(row=1, column=1, \
            padx=10, pady=10, sticky="nsew")
        self.metadata_outer_frame = \
            ttk.Frame(self.verify_frame, padding=(10, 10))
        self.metadata_outer_frame.grid(row=2, column=1, \
            sticky="nsew")
        self.metadata_outer_frame.rowconfigure(0, weight=1)
        self.metadata_outer_frame.columnconfigure(0, weight=1)
        self.metadata_text = tk.Text(self.metadata_outer_frame, \
            wrap=tk.WORD, height=10, width=60, padx=10, pady=10, \
                font=('Consolas', 10), state=tk.DISABLED, \
                    borderwidth=1, relief="sunken", \
                        background="#FFFFFF")
        self.metadata_scrollbar = \
            ttk.Scrollbar(self.metadata_outer_frame, \
                orient=tk.VERTICAL, command=self.metadata_text.yview)
        self.metadata_text.config(yscrollcommand=self.metadata_scrollbar.set)
        self.metadata_text.grid(row=0, column=0, sticky="nsew")
        self.metadata_scrollbar.grid(row=0, column=1, sticky="ns")

        self.button_frame = ttk.Frame(self.verify_frame, padding=(0, 10))
        self.button_frame.grid(row=3, column=1, sticky="ew")
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.columnconfigure(1, weight=0)
        self.button_frame.columnconfigure(2, weight=0)
        self.button_frame.columnconfigure(3, weight=1)
        self.reject_button = ttk.Button(self.button_frame, text="Reject",
                                        command=self._reject_action,
                                        style="Reject.TButton", width=12)
        self.reject_button.grid(row=0, column=1, padx=15, ipady=2)
        self.verify_button = ttk.Button(self.button_frame, text="Verify",
                                        command=self._verify_action,
                                        style="Verify.TButton", width=12)
        self.verify_button.grid(row=0, column=2, padx=15, ipady=2)

    def _setup_complete_widgets(self):
        """
        Sets up widgets for 
        the completion screen.
        """
        self.complete_label = ttk.Label(self.complete_frame,
                                        text="Verification Complete!",
                                        style="Header.TLabel")
        self.complete_label.pack(pady=(20, 10))
        self.complete_info_label = \
            ttk.Label(self.complete_frame, text="",
                style="Status.TLabel", justify=tk.CENTER)
        self.complete_info_label.pack(pady=5)

        self.complete_button_frame = ttk.Frame(self.complete_frame)
        self.complete_button_frame.pack(pady=20)

        self.open_logs_button = \
            ttk.Button(self.complete_button_frame, \
                text="Open Log Files", \
                    command=self._open_logs, \
                        style="TButton", width=20)
        self.final_action_button = \
            ttk.Button(self.complete_button_frame, \
                text="Close", command=self.master.quit, \
                    style="Close.TButton", width=20)
        self.final_action_button.pack(side=tk.LEFT, \
            padx=10, ipady=5)

    def _on_resize(self, _event: tk.Event = None):
        """
        Handles window resize events to rescale 
        the displayed image proportionally
        within the image_label widget.
        Args:
            event: The Tkinter configure event 
            object (passed automatically).
        """
        if not self.current_pil_image:
            return

        try:
            widget_width = self.image_label.winfo_width()
            widget_height = self.image_label.winfo_height()

            pad_x = 10
            pad_y = 10

            target_width = max(1, widget_width - (2 * pad_x))
            target_height = max(1, widget_height - (2 * pad_y))

            if target_width <= 1 or target_height <= 1:
                return

            img_width, img_height = self.current_pil_image.size
            if img_height == 0:
                logger.warning("Cannot resize image with zero height.")
                return
            img_ratio = img_width / img_height

            widget_ratio = target_width / target_height

            if widget_ratio > img_ratio:
                new_height = target_height
                new_width = int(new_height * img_ratio)
            else:
                new_width = target_width
                new_height = int(new_width / img_ratio)

            new_width = max(1, new_width)
            new_height = max(1, new_height)

            resized_image = self.current_pil_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

            self.img_tk = ImageTk.PhotoImage(resized_image)

            self.image_label.config(image=self.img_tk)

        except (ValueError, MemoryError) as pil_error:
            logger.error("PIL error during image resize: %s", pil_error)
            self.image_label.config(image='')
        except Exception as e:
            logger.error("Unexpected error during image resize: \
                %s", e, exc_info=True)
            self.image_label.config(image='')

    def _start_verification(self):
        """
        Handles the 'Start Verifying'
        button click: transitions UI
        and starts sampling.
        """
        logger.info("Verification process started by user.")
        self.start_frame.pack_forget()
        self.progress_frame.pack(fill=tk.BOTH, expand=True)
        self.master.update_idletasks()

        self._get_samples()

        if not self.sampled_token_ids:
            messagebox.showerror("Error",
                                 f"No metadata files found in range \
                                     {TOKEN_START}-{TOKEN_END} "
                                 f"in directory:\n{METADATA_DIR}")
            logger.critical("Aborting verification due to no \
                available metadata files.")
            self.master.quit()
            return

        self.progress_frame.pack_forget()
        self.verify_frame.pack(fill=tk.BOTH, expand=True)
        self.current_index = 0
        self._load_and_display_item()

    def _get_samples(self):
        """
        Selects random token IDs
        from the available metadata
        files, updating progress.
        """
        logger.info("Selecting up to %d random tokens between %d and %d",
                    SAMPLE_SIZE, TOKEN_START, TOKEN_END)
        available_tokens = []

        if not os.path.isdir(METADATA_DIR):
            logger.error("Metadata directory not found: %s", METADATA_DIR)
            return

        total_range = TOKEN_END - TOKEN_START + 1
        self.progress_bar['maximum'] = total_range
        self.progress_bar['value'] = 0
        check_interval = max(1, total_range // 100)
        self.progress_label.config(text=f"Checking metadata files \
            ({TOKEN_START}-{TOKEN_END})...")
        self.master.update_idletasks()

        for i, token_id in enumerate(range(TOKEN_START, \
            TOKEN_END + 1)):
            if os.path.exists(os.path.join(METADATA_DIR, \
                str(token_id))):
                available_tokens.append(token_id)

            if i % check_interval == 0 or i == total_range - 1:
                self.progress_bar['value'] = i + 1
                self.master.update_idletasks()

        self.progress_bar['value'] = total_range
        self.progress_label.config(text="Selecting random sample...")
        self.master.update_idletasks()

        if not available_tokens:
            logger.error("No metadata files found in range \
                %d-%d in directory %s",
                         TOKEN_START, TOKEN_END, METADATA_DIR)
            return

        actual_sample_size = min(SAMPLE_SIZE, len(available_tokens))
        if actual_sample_size < SAMPLE_SIZE:
            logger.warning("Requested sample size %d > available tokens \
                (%d). Using %d.",
                            SAMPLE_SIZE, len(available_tokens), \
                                actual_sample_size)

        self.sampled_token_ids = random.sample(available_tokens, \
            actual_sample_size)
        logger.info("Selected %d token IDs for verification.", \
            len(self.sampled_token_ids))

        self.progress_label.config \
            (text=f"Selected {len(self.sampled_token_ids)} tokens. \
                Starting...")
        self.master.update_idletasks()
        self.master.after(500)

    def _load_and_display_item(self):
        """
        Loads data for the current
        token ID and updates the
        verification UI elements.
        """
        if not 0 <= self.current_index < len(self.sampled_token_ids):
            logger.error("Invalid current index: %d. \
                Cannot load item.", self.current_index)
            self._complete_verification()
            return

        current_token_id = self.sampled_token_ids[self.current_index]

        self.token_id_label.config(text=f"Token ID: {current_token_id}")
        self.index_label.config(text=f"Sample: {self.current_index + 1} / "
                                     f"{len(self.sampled_token_ids)}")

        self.image_label.config(image='', text="Loading...")
        self.metadata_text.config(state=tk.NORMAL)
        self.metadata_text.delete(1.0, tk.END)
        self.metadata_text.insert(tk.END, "Loading metadata...")
        self.metadata_text.config(state=tk.DISABLED)
        self.master.update_idletasks()

        self.current_metadata = load_metadata(current_token_id)
        formatted_meta = format_metadata_display(self.current_metadata)

        image_url = None
        if self.image_map:
            map_key = f"tokenId {current_token_id}"
            image_url = self.image_map.get(map_key)
            if not image_url:
                logger.warning("Image URL not found for key \
                    '%s' in image map.", map_key)
        else:
            logger.error("Image map is not loaded. \
                Cannot fetch image URL.")

        self.current_pil_image = load_image_from_url(image_url) \
            if image_url else None

        self.metadata_text.config(state=tk.NORMAL)
        self.metadata_text.delete(1.0, tk.END)
        self.metadata_text.insert(tk.END, formatted_meta)
        self.metadata_text.config(state=tk.DISABLED)
        self.metadata_text.yview_moveto(0.0)

        if self.current_pil_image:
            self.image_label.config(text="")
            self.image_label.event_generate('<Configure>')
        else:
            self.img_tk = None
            self.image_label.config(image='', text="Image Unavailable")
            logger.warning("Image could not be loaded or found \
                for token %d (URL: %s)",
                         current_token_id, image_url or "N/A")

        self.verify_button.config(state=tk.NORMAL)
        self.reject_button.config(state=tk.NORMAL)

    def _verify_action(self):
        """
        Handles the 'Verify' button
        click: logs verification
        and loads next item.
         """
        if 0 <= self.current_index < len(self.sampled_token_ids):
            logger.info("Token %d verified.", \
                self.sampled_token_ids[self.current_index])
            self._load_next_item()
        else:
            logger.warning("Verify action called with invalid index: \
                %d", self.current_index)

    def _reject_action(self):
        """
        Handles the 'Reject' button
        click: logs rejection details
        and loads next item.
        """
        if 0 <= self.current_index < len(self.sampled_token_ids):
            rejected_id = self.sampled_token_ids[self.current_index]
            logger.warning("Token %d REJECTED.", rejected_id)
            self.rejected_token_ids.append(rejected_id)

            try:
                with open(REJECT_LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"--- REJECTED Token ID: {rejected_id} ---\n")
                    f.write(format_metadata_display(self.current_metadata))
                    f.write("\n\n")
            except Exception as e:
                logger.error("Failed to write to rejection log %s: %s",
                             REJECT_LOG_FILE, e)

            self._load_next_item()
        else:
            logger.warning("Reject action called with invalid index: \
                %d", self.current_index)

    def _load_next_item(self):
        """
        Advances the index and
        loads the next item, or
        completes verification.
        """
        self.current_index += 1
        if self.current_index < len(self.sampled_token_ids):
            self.verify_button.config(state=tk.DISABLED)
            self.reject_button.config(state=tk.DISABLED)
            self.master.after(10, self._load_and_display_item)
        else:
            self._complete_verification()

    def _complete_verification(self):
        """
        Shows the completion screen,
        runs analysis, and
        configures final UI state.
        """
        logger.info("GUI verification loop completed.")
        self.verify_frame.pack_forget()
        self.complete_frame.pack(fill=tk.BOTH, expand=True)
        self.master.update_idletasks()

        analysis_needed = bool(self.rejected_token_ids)
        post_verification_analysis(self.rejected_token_ids)

        num_rejected = len(self.rejected_token_ids)
        num_sampled = len(self.sampled_token_ids)
        num_verified = num_sampled - num_rejected

        if analysis_needed:
            info_text = (f"Verification Complete!\n\n"
                         f"Verified: {num_verified}\n"
                         f"Rejected: {num_rejected}\n\n"
                         f"Rejection details and analysis saved to log files.")
            self.complete_info_label.config(text=info_text)
            self.final_action_button.config(text="Close",
                                            command=self.master.quit,
                                            style="Close.TButton")
            self.open_logs_button.pack(side=tk.LEFT, padx=10, ipady=5)
            messagebox.showwarning("Verification Complete",
                                   f"{num_rejected} item(s) were rejected. "
                                   "Please review the log files.")
        else:
            info_text = (f"Verification Complete!\n\n"
                         f"All {num_sampled} sampled items verified successfully!")
            self.complete_info_label.config(text=info_text)
            self.final_action_button.config(text="Close",
                                            command=self.master.quit,
                                            style="Verify.TButton")
            self.open_logs_button.pack_forget()
            messagebox.showinfo("Verification Complete",
                                "All sampled items verified successfully!")

        self.final_action_button.pack(side=tk.LEFT, padx=10, ipady=5)


    def _open_logs(self, open_main_log: bool = True):
        """
        Opens the relevant log files using the default system application.

        Args:
            open_main_log: Whether to open the main verification log.
        """
        files_to_open = []
        if open_main_log and os.path.exists(MAIN_LOG_FILE):
            files_to_open.append(MAIN_LOG_FILE)
        if self.rejected_token_ids:
            if os.path.exists(REJECT_LOG_FILE):
                files_to_open.append(REJECT_LOG_FILE)
            if os.path.exists(ANALYSIS_LOG_FILE):
                files_to_open.append(ANALYSIS_LOG_FILE)

        opened_count = 0
        if not files_to_open:
            messagebox.showinfo("No Logs Found", "Could not find any \
                relevant log files to open.")
            return

        for log_file in files_to_open:
            open_file(log_file)
            opened_count += 1

        if opened_count == 0:
            messagebox.showinfo("No Logs Opened", \
                "Could not open any log files.")


def post_verification_analysis(rejected_ids: List[int]):
    """
    Analyzes rejected tokens by counting trait occurrences and logs findings.
    Overwrites the analysis log file if rejections occurred.

    Args:
        rejected_ids: A list of token IDs that were marked as rejected.
    """
    logger.info("Starting post-verification analysis...")

    if not rejected_ids:
        logger.info("No tokens were rejected in this session. No analysis needed.")
        return

    logger.warning("%d tokens were rejected. Analyzing...", len(rejected_ids))
    trait_value_counts: Dict[str, Counter] = {}

    try:
        with open(ANALYSIS_LOG_FILE, 'w', encoding='utf-8') as f_analysis:
            f_analysis.write("Verification Analysis Report\n")
            f_analysis.write("============================\n")
            current_time = logging.Formatter('%(asctime)s').formatTime(
                logging.LogRecord(None, None, '', 0, '', (), None, None),
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            f_analysis.write(f"Timestamp: {current_time}\n")
            f_analysis.write(f"Total Rejected Tokens in this session: \
                {len(rejected_ids)}\n")
            rejected_ids_str = ', '.join(map(str, sorted(rejected_ids)))
            f_analysis.write(f"Rejected Token IDs: {rejected_ids_str}\n\n")
            f_analysis.write("Analysis of Traits in Rejected Tokens:\n")
            f_analysis.write("--------------------------------------\n")

            for token_id in sorted(rejected_ids):
                metadata = load_metadata(token_id)
                if metadata and isinstance(metadata.get('attributes'), list):
                    attributes = metadata['attributes']
                    if not attributes:
                        f_analysis.write(f" - Token {token_id}: No attributes \
                            found in metadata.\n")
                    else:
                        for attr in attributes:
                            if isinstance(attr, dict):
                                trait_type = attr.get('trait_type', \
                                    'Unknown Trait Type')
                                value = attr.get('value', 'Unknown Value')
                                if trait_type not in trait_value_counts:
                                    trait_value_counts[trait_type] = Counter()
                                trait_value_counts[trait_type][str(value)] += 1
                            else:
                                f_analysis.write(f" - Token {token_id}: "
                                                 f"Found non-dictionary item \
                                                     in attributes list: \
                                                         {attr}\n")
                else:
                    f_analysis.write(f" - Could not load/parse metadata \
                        or attributes for "
                                     f"rejected token {token_id}\n")

            if not trait_value_counts:
                f_analysis.write("\n - No valid attributes found across \
                    all rejected token "
                                 "metadata for analysis.\n")
            else:
                for trait_type, counter in sorted(trait_value_counts.items()):
                    f_analysis.write(f"\nTrait Type: '{trait_type}'\n")
                    for value, count in sorted(counter.items(),
                                               key=lambda item: \
                                                   (-item[1], item[0])):
                        f_analysis.write(f"  - Value '{value}': \
                            {count} rejection(s)\n")

            f_analysis.write("\n--- End of Report ---\n")
            logger.info("Analysis complete. Summary written to %s", \
                ANALYSIS_LOG_FILE)
            logger.info("Detailed rejection data is appended to %s", \
                REJECT_LOG_FILE)

    except Exception as e:
        logger.error("Failed during post-verification analysis: \
            %s", e, exc_info=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = MetadataVerifierApp(root)
    if hasattr(app, 'image_map') and app.image_map is not None:
        root.mainloop()
        logger.info("Application closed normally.")
    else:
        logger.info("Application did not start due to \
            initialization errors.")
        try:
            root.destroy()
        except tk.TclError:
            pass
