# pylint: disable=broad-exception-caught, too-many-locals,
# too-many-statements, import-error, wrong-import-position
"""
NFT Trait Classifier Trainer - 
Part 1 (Improved with GUI Verification)

This script reads the first 1294 NFT images 
and their corresponding metadata to train
visual classifiers for identifying trait values.
It incorporates improved feature
extraction (color histograms for backgrounds)
and optional *GUI-based* human verification and correction.
Models are saved using gzip-compressed pickle format
(.pkl.gz), including the necessary feature transformation step
(e.g., PCA or histogram function). Image URLs are accessed 
via an images.json file, and metadata JSON files are 
read from the "metadata" folder. Metadata files are named
by token ID (e.g., "1", "2") without an extension.

Requirements:
- Pillow
- numpy
- scikit-learn
- tqdm
- Tkinter (usually included with Python)
"""
import os
import json
import gzip
import pickle
import logging
import sys
import urllib.request
import urllib.error
import socket
import random
import tkinter as tk
from tkinter import ttk, messagebox
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageTk
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

BASE_DIR = "C:/bedlam/md_gen_v4"
LOG_DIR = os.path.join(BASE_DIR, "logs")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
CLASSIFIER_DIR = os.path.join(CONFIG_DIR, "classifiers")
NUM_TOKENS_TO_TRAIN = 1294
IMAGE_SIZE = (128, 128)
DISPLAY_IMAGE_SIZE = (400, 400)
IMAGE_JSON_PATH = os.path.join(CONFIG_DIR, "images.json")
REQUEST_TIMEOUT = 20
HISTOGRAM_BINS = 16
VERIFICATION_ENABLED = True
VERIFICATION_FREQUENCY = 100
VERIFICATION_SAMPLE_SIZE = 5
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(CLASSIFIER_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "trait_training_v2_gui.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def load_image(url: str) -> Optional[Tuple[Image.Image, Image.Image]]:
    """
    Downloads, opens, converts an image from a given URL.
    Returns both the original image (for display) 
    and a resized version (for features).
    Handles potential network and image processing errors.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    original_image: Optional[Image.Image] = None
    resized_image: Optional[Image.Image] = None
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            if response.status != 200:
                logger.error("Failed to load image from URL %s: HTTP Status %d",
                             url, response.status)
                return None
            image_data = response.read()
            with BytesIO(image_data) as image_stream:
                original_image = Image.open(image_stream).convert("RGB")
                resized_image = original_image.resize(IMAGE_SIZE, \
                    Image.Resampling.LANCZOS)
                return original_image, resized_image
    except urllib.error.HTTPError as e:
        logger.error("Failed to load image from URL %s: HTTPError %s - %s",
                     url, e.code, e.reason)
        return None
    except urllib.error.URLError as e:
        reason = e.reason if isinstance(e.reason, str) \
            else getattr(e.reason, 'strerror', str(e.reason))
        if isinstance(e.reason, socket.timeout) or 'timed out' in reason.lower():
            logger.error("Failed to load image from URL %s: Request timed out after \
                %d seconds", url, REQUEST_TIMEOUT)
        else:
            logger.error("Failed to load image from URL %s: URLError - %s", url, reason)
        return None
    except socket.timeout:
        logger.error("Failed to load image from URL %s: Socket timed out after %d seconds",
                     url, REQUEST_TIMEOUT)
        return None
    except UnidentifiedImageError:
        logger.error("Failed to load image from URL %s: Cannot identify image file", url)
        return None
    except (OSError, IOError, ValueError) as e:
        logger.error("Failed processing image from URL %s: %s", url, str(e))
        return None
    except Exception as e:
        logger.error("Unexpected error loading image from URL %s: \
            %s", url, e, exc_info=True)
        return None

def load_metadata(token_id: int) -> Optional[Dict]:
    """Loads metadata from the metadata directory for a given token ID."""
    metadata_path = os.path.join(METADATA_DIR, str(token_id))
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info("Metadata file not found for token ID %d: %s (Expected for training)",
                     token_id, metadata_path)
        return None
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in metadata file %s: %s", metadata_path, e)
        return None
    except OSError as e:
        logger.error("OS error reading metadata file %s: %s", metadata_path, e)
        return None
    except Exception as e:
        logger.error("Unexpected error loading metadata for token ID %d: %s",
                     token_id, e, exc_info=True)
        return None

def extract_traits(metadata: Optional[Dict]) -> Dict[str, str]:
    """Extracts a dictionary of trait_type: value from the metadata attributes."""
    if not metadata:
        return {}
    attributes = metadata.get("attributes", [])
    if not isinstance(attributes, list):
        logger.warning("Metadata attributes format is not a list for token %s. "
                       "Cannot extract traits.", metadata.get('name', 'Unknown'))
        return {}
    traits = {}
    for attr in attributes:
        if isinstance(attr, dict) and "trait_type" in attr and "value" in attr:
            traits[str(attr["trait_type"]).strip()] = str(attr["value"]).strip()
        else:
            logger.debug("Skipping invalid attribute item in token %s: %s",
                         metadata.get('name', 'Unknown'), attr)
    return traits

def image_to_vector(image: Image.Image) -> np.ndarray:
    """Converts an image into a flattened numpy array vector."""
    try:
        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        return np.asarray(image).flatten()
    except Exception as e:
        logger.error("Failed to convert image to vector: %s", e, exc_info=True)
        return np.array([])

def image_to_color_histogram(image: Image.Image, bins: int = HISTOGRAM_BINS) -> np.ndarray:
    """Calculates a flattened, normalized 3D color histogram for an RGB image."""
    try:
        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.asarray(image)
        hist_r = np.histogram(img_array[:, :, 0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:, :, 1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:, :, 2], bins=bins, range=(0, 256))[0]
        total_pixels = image.width * image.height
        if total_pixels == 0:
            logger.error("Cannot calculate histogram for zero-pixel image.")
            return np.array([])
        hist_r = hist_r / total_pixels
        hist_g = hist_g / total_pixels
        hist_b = hist_b / total_pixels
        return np.concatenate((hist_r, hist_g, hist_b))
    except Exception as e:
        logger.error("Failed to calculate color histogram: %s", e, exc_info=True)
        return np.array([])

def get_feature_extractor_and_data(
    trait_type: str,
    image_vector: np.ndarray,
    image_histogram: np.ndarray
) -> Optional[np.ndarray]:
    """Selects the appropriate feature vector based on trait type."""
    if trait_type == "Background":
        if image_histogram.size > 0:
            return image_histogram
        else:
            logger.warning("Requested histogram features for '%s', but they are empty.",
                           trait_type)
            return None
    else:
        if image_vector.size > 0:
            return image_vector
        else:
            logger.warning("Requested vector features for '%s', but they are empty.",
                           trait_type)
            return None

class VerificationWindow(tk.Toplevel):
    """GUI window for verifying and correcting token traits."""
    def __init__(self, master: tk.Tk, token_id: int, image: Image.Image,
                 initial_traits: Dict[str, str]):
        super().__init__(master)
        self.title(f"Verify & Correct Token {token_id}")
        self.grab_set()
        self.transient(master)
        self.token_id = token_id
        self.image = image
        self.initial_traits = initial_traits
        self.corrected_traits: Optional[Dict[str, str]] = None
        self.trait_widgets: Dict[str, Dict[str, Any]] = {}

        self.geometry("850x750")
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)

        img_frame = ttk.Frame(main_frame, style="TFrame")
        img_frame.grid(row=0, column=0, pady=10, sticky="n")
        try:
            display_img = image.copy()
            display_img.thumbnail(DISPLAY_IMAGE_SIZE, Image.Resampling.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(display_img)
            img_label = tk.Label(img_frame, image=self.img_tk, \
                relief="groove", borderwidth=1)
            img_label.pack()
        except Exception as e:
            img_label = tk.Label(
                img_frame, text=f"Error displaying image:\n{e}",
                fg="red", bg="#EEE", relief="groove", borderwidth=1,
                width=int(DISPLAY_IMAGE_SIZE[0]/8), \
                    height=int(DISPLAY_IMAGE_SIZE[1]/15)
            )
            img_label.pack(padx=10, pady=10)
            logger.error("Tkinter: Failed to display image for token %d: \
                %s", token_id, e)

        traits_outer_frame = ttk.Frame(main_frame, style="TFrame")
        traits_outer_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        traits_outer_frame.rowconfigure(0, weight=1)
        traits_outer_frame.columnconfigure(0, weight=1)

        traits_canvas = tk.Canvas(traits_outer_frame, \
            borderwidth=0, background="#ffffff")
        scrollbar = ttk.Scrollbar(traits_outer_frame, orient="vertical", \
            command=traits_canvas.yview)
        scrollable_frame = ttk.Frame(traits_canvas, style="TFrame")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: traits_canvas.configure(scrollregion=traits_canvas.bbox("all"))
        )

        canvas_window = traits_canvas.create_window((0, 0), \
            window=scrollable_frame, anchor="nw")
        traits_canvas.configure(yscrollcommand=scrollbar.set)

        traits_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        traits_canvas.bind('<Configure>', lambda e: \
            traits_canvas.itemconfig(canvas_window, width=e.width))

        header_label = ttk.Label(scrollable_frame, text="Verify / Correct Traits:",
                                 font=('Segoe UI', 12, 'bold'), style="TLabel")
        header_label.grid(row=0, column=0, columnspan=3, \
            pady=(5, 10), padx=10, sticky='w')

        row_num = 1
        for trait_type, value in sorted(initial_traits.items()):
            ttk.Label(scrollable_frame, text=f"{trait_type}:", \
                anchor="e", style="TLabel").grid(
                row=row_num, column=0, padx=(10, 5), pady=3, sticky='ew')

            correction_entry = ttk.Entry(scrollable_frame, width=40, \
                font=('Segoe UI', 10))
            correction_entry.insert(0, value)
            correction_entry.grid(row=row_num, column=1, padx=5, pady=3, sticky='ew')

            self.trait_widgets[trait_type] = {
                "entry": correction_entry,
            }
            row_num += 1

        scrollable_frame.columnconfigure(0, weight=1, pad=5)
        scrollable_frame.columnconfigure(1, weight=3, pad=5)

        button_frame = ttk.Frame(main_frame, style="TFrame")
        button_frame.grid(row=2, column=0, pady=(10, 5), sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=0)
        button_frame.columnconfigure(2, weight=0)
        button_frame.columnconfigure(3, weight=1)

        submit_button = ttk.Button(button_frame, text="Submit Corrections",
                                   command=self._submit, style="Verify.TButton", width=18)
        submit_button.grid(row=0, column=1, padx=10, pady=5, ipady=3)

        discard_button = ttk.Button(button_frame, text="Discard Token",
                                    command=self._discard, style="Reject.TButton", width=18)
        discard_button.grid(row=0, column=2, padx=10, pady=5, ipady=3)

        self.update_idletasks()
        master_width = master.winfo_screenwidth()
        master_height = master.winfo_screenheight()
        win_width = self.winfo_width()
        win_height = self.winfo_height()
        x_coord = int((master_width / 2) - (win_width / 2))
        y_coord = int((master_height / 2) - (win_height / 2))
        self.geometry(f'{win_width}x{win_height}+{x_coord}+{y_coord}')

        first_entry = next((widgets["entry"] for widgets \
            in self.trait_widgets.values()), None)
        if first_entry:
            first_entry.focus_set()

        self.bind('<Return>', lambda event=None: self._submit())
        self.bind('<Escape>', lambda event=None: self._discard())

    def _submit(self):
        """Collect corrected traits and close the window."""
        self.corrected_traits = {}
        found_change = False
        try:
            for trait_type, widgets in self.trait_widgets.items():
                corrected_value = widgets["entry"].get().strip()
                if corrected_value:
                    self.corrected_traits[trait_type] = corrected_value
                    if corrected_value != self.initial_traits.get(trait_type, ""):
                        found_change = True
                else:
                    if trait_type in self.initial_traits:
                        logger.warning("User submitted empty value for existing trait '%s' "
                                       "on token %d. Excluding trait.",
                                       trait_type, self.token_id)
                        found_change = True

            if not self.corrected_traits:
                if messagebox.askyesno("Confirm Empty Traits",
                                       "You haven't provided any trait \
                                           values (or cleared them all).\n"
                                       "Do you want to DISCARD this token instead?",
                                       parent=self, icon='warning'):
                    self.corrected_traits = None
                    logger.warning("User confirmed discarding token \
                        %d due to empty traits submission.",
                                   self.token_id)
                else:
                    return

            if self.corrected_traits is not None:
                if found_change:
                    logger.info("User submitted corrections for token %d.", self.token_id)
                else:
                    logger.info("User confirmed original traits for \
                        token %d (no changes made).",
                                self.token_id)

            self.destroy()

        except Exception as e:
            logger.error("Error submitting corrections for token %d: %s",
                         self.token_id, e, exc_info=True)
            messagebox.showerror("Error", f"An error occurred while \
                submitting:\n{e}", parent=self)
            if messagebox.askyesno("Error Handling",
                                   "An error occurred submitting.\nDiscard this token?",
                                   parent=self):
                self.corrected_traits = None
                self.destroy()

    def _discard(self):
        """Signal that the token should be discarded after confirmation."""
        if messagebox.askyesno("Confirm Discard",
                               "Are you sure you want to discard all data for this token?\n"
                               "It will NOT be used for training.",
                               parent=self, icon='question'):
            logger.warning("User chose to discard token %d via GUI button.", self.token_id)
            self.corrected_traits = None
            self.destroy()

    def wait_for_input(self) -> Optional[Dict[str, str]]:
        """Blocks execution until the window is closed and returns the result."""
        self.wait_window(self)
        return self.corrected_traits

def verify_and_correct_token_gui(token_id: int, image: Image.Image,
                                 traits: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Displays image and traits in a GUI, allows correction, returns corrected traits or None.
    Manages the temporary Tkinter root window.
    """
    print("-" * 60)
    print(f"Requesting verification/correction for Token ID: {token_id}")
    print("Check the pop-up window. You can edit traits and then 'Submit' or 'Discard'.")
    print("-" * 60)

    root = None
    try:
        root = tk.Tk()
        root.withdraw()

        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure("TFrame", background="#ECECEC")
        style.configure("TLabel", padding=6, font=('Segoe UI', 10),
                        background="#ECECEC", foreground="#333333")
        style.configure("Verify.TButton", foreground="white", background="#388E3C",
                        font=('Segoe UI', 10, 'bold'))
        style.map("Verify.TButton", background=[('active', '#2E7D32')])
        style.configure("Reject.TButton", foreground="white", background="#D32F2F",
                        font=('Segoe UI', 10, 'bold'))
        style.map("Reject.TButton", background=[('active', '#C62828')])

        dialog = VerificationWindow(root, token_id, image, traits)
        corrected_traits = dialog.wait_for_input()
        return corrected_traits

    except tk.TclError as e:
        logger.error("Tkinter TclError occurred: %s. Tkinter might not be properly "
                     "installed or configured.", e, exc_info=True)
        print("\nERROR: Could not display verification window (Tkinter TclError).")
        print("Ensure you have a graphical environment and Tkinter is installed correctly.")
        print("Falling back to simple console verification (y/n discard).")
        while True:
            confirm = input(f"Fallback: Discard data for token {token_id} "
                            "due to GUI error? (y/n): ").lower().strip()
            if confirm == 'y':
                logger.warning("User discarded token %d via fallback due to GUI TclError.",
                               token_id)
                return None
            if confirm == 'n':
                logger.info("User chose to keep token %d data via fallback \
                    despite GUI TclError.",
                            token_id)
                return traits
            print("Invalid input.")

    except Exception as e:
        logger.error("Failed to run verification GUI for token %d: %s",
                     token_id, e, exc_info=True)
        print(f"\nERROR: An unexpected error occurred launching the verification window "
              f"for token {token_id}.")
        print("Falling back to simple console verification (y/n discard).")
        while True:
            confirm = input(f"Fallback: Discard data for token {token_id} "
                            "due to GUI error? (y/n): ").lower().strip()
            if confirm == 'y':
                logger.warning("User discarded token %d via fallback due to GUI error.",
                               token_id)
                return None
            if confirm == 'n':
                logger.info("User chose to keep token %d data via fallback \
                    despite GUI error.",
                            token_id)
                return traits
            print("Invalid input.")
    finally:
        if root:
            try:
                root.destroy()
            except tk.TclError:
                pass

def train_classifiers(image_map: Dict[str, str]):
    """
    Trains and saves classifiers for each trait type using appropriate features.
    Includes optional GUI-based human verification and correction step.
    """
    training_data: Dict[str, List[Tuple[np.ndarray, str]]] = {}
    processed_tokens = 0
    verified_correct_count = 0
    corrected_count = 0
    discarded_count = 0
    skipped_verification_count = 0

    token_ids_to_process = list(range(1, NUM_TOKENS_TO_TRAIN + 1))
    verification_indices = set()
    verification_possible = VERIFICATION_ENABLED

    if VERIFICATION_ENABLED:
        if VERIFICATION_FREQUENCY > 0 and VERIFICATION_SAMPLE_SIZE > 0:
            num_to_verify = max(1, (NUM_TOKENS_TO_TRAIN // VERIFICATION_FREQUENCY)
                                * VERIFICATION_SAMPLE_SIZE)
            num_to_verify = min(num_to_verify, NUM_TOKENS_TO_TRAIN)
            try:
                verification_indices = set(random.sample(range(NUM_TOKENS_TO_TRAIN), \
                    k=num_to_verify))
                logger.info("GUI Verification enabled. Will attempt to \
                    verify/correct %d tokens.",
                            len(verification_indices))
            except ValueError as e:
                logger.error("Error during verification sampling (k=%d, N=%d): %s. "
                             "Disabling verification for this run.",
                             num_to_verify, NUM_TOKENS_TO_TRAIN, e)
                verification_possible = False
        else:
            logger.warning("Verification frequency or sample size is zero. "
                           "Disabling verification.")
            verification_possible = False

    if not verification_possible:
        logger.info("Verification is disabled for this run \
            (due to config or sampling error).")
        verification_indices = set()

    # --- Stage 1: Data Collection and Verification ---
    logger.info("--- Stage 1: Collecting Training Data \
        & Performing Verification ---")
    for idx, token_id in enumerate(tqdm(token_ids_to_process, \
        desc="Collecting Data", unit="token")):
        key = f"tokenId {token_id}"
        image_url = image_map.get(key)
        if not image_url:
            logger.warning("Missing image URL for token ID %d. \
                Skipping.", token_id)
            continue

        load_result = load_image(image_url)
        if load_result is None:
            logger.warning("Failed to load image for token ID %d. \
                Skipping.", token_id)
            continue
        original_image, resized_image = load_result

        metadata = load_metadata(token_id)
        if not metadata:
            continue

        initial_traits = extract_traits(metadata)
        if not initial_traits:
            logger.warning("No traits extracted for token ID \
                %d from metadata. Skipping.", token_id)
            continue

        current_traits = initial_traits
        needs_verification = verification_possible and (idx in verification_indices)

        if needs_verification:
            verification_result = verify_and_correct_token_gui(token_id, \
                original_image, initial_traits)

            if verification_result is None:
                discarded_count += 1
                logger.warning("Skipping data collection for token \
                    %d due to user discard in GUI.",
                               token_id)
                continue
            else:
                current_traits = verification_result
                if current_traits != initial_traits:
                    corrected_count += 1
                    logger.info("Applied user corrections for token %d.", \
                        token_id)
                else:
                    verified_correct_count += 1
                    logger.info("User confirmed original traits for token \
                        %d via GUI.", token_id)
        elif verification_possible:
            skipped_verification_count += 1

        vector = image_to_vector(resized_image)
        histogram = image_to_color_histogram(resized_image)

        if vector.size == 0 and histogram.size == 0:
            logger.warning("Skipping token ID %d due to all feature \
                extraction failing.", token_id)
            continue
        elif vector.size == 0:
            logger.warning("Vector feature extraction failed for token %d.", \
                token_id)
        elif histogram.size == 0:
            logger.warning("Histogram feature extraction failed for token %d.", \
                token_id)

        if not current_traits:
            logger.warning("Skipping token %d as it has no traits after potential "
                           "verification/correction.", token_id)
            continue

        token_data_added = False
        for trait_type, value in current_traits.items():
            if not value:
                logger.debug("Skipping empty value for trait '%s' on token %d.",
                             trait_type, token_id)
                continue

            features = get_feature_extractor_and_data(trait_type, vector, histogram)

            if features is not None and features.size > 0:
                training_data.setdefault(trait_type, []).append((features, value))
                token_data_added = True
            else:
                logger.warning("Could not get valid/required features for trait '%s' on "
                               "token %d. Skipping this trait.", trait_type, token_id)

        if token_data_added:
            processed_tokens += 1

    logger.info("--- Data Collection Summary ---")
    logger.info("Total tokens processed for data collection: %d / %d",
                processed_tokens, NUM_TOKENS_TO_TRAIN)
    if VERIFICATION_ENABLED:
        logger.info("Verification results: %d confirmed correct, \
            %d corrected, %d discarded, " "%d skipped check.", 
            verified_correct_count, corrected_count,
                    discarded_count, skipped_verification_count)
    if discarded_count > 0:
        logger.warning("Review the log for details on %d tokens discarded \
            during verification.",
                       discarded_count)
    if corrected_count > 0:
        logger.info("%d tokens had their traits corrected \
            during verification.", corrected_count)

    logger.info("--- Stage 2: Training Classifiers ---")
    classifiers: Dict[str, Pipeline] = {}

    all_trait_types = sorted(training_data.keys())
    logger.info("Found data for %d trait types: %s",
                len(all_trait_types), ", ".join(all_trait_types))

    for trait_type in tqdm(all_trait_types, desc="Training Classifiers", unit="trait"):
        entries = training_data.get(trait_type, [])
        sanitized_trait = trait_type.replace(' ', '_').replace('/', '_')
        model_path = os.path.join(CLASSIFIER_DIR, f"{sanitized_trait}.pkl.gz")

        loaded_successfully = False
        if os.path.exists(model_path):
            logger.info("Attempting to load existing classifier for '%s' from %s",
                        trait_type, model_path)
            try:
                with gzip.open(model_path, "rb") as f:
                    loaded_pipeline = pickle.load(f)
                if isinstance(loaded_pipeline, Pipeline) \
                    and hasattr(loaded_pipeline, 'predict'):
                    classifiers[trait_type] = loaded_pipeline
                    logger.info("Successfully loaded existing classifier \
                        for '%s'.", trait_type)
                    loaded_successfully = True
                else:
                    logger.warning("Loaded object for '%s' from %s is not a valid "
                                   "scikit-learn Pipeline. Retraining.", \
                                       trait_type, model_path)
            except (pickle.PickleError, EOFError, gzip.BadGzipFile,
                    OSError, ValueError, NotFittedError) as e:
                logger.error("Error loading/validating existing classifier \
                    for '%s' from %s: %s. "
                             "Retraining.", trait_type, model_path, e)
            except Exception as e:
                logger.error("Unexpected error loading existing classifier \
                    for '%s' from %s: %s. "
                             "Retraining.", trait_type, model_path, e, exc_info=True)

        if loaded_successfully:
            continue

        logger.info("Training new classifier for trait type '%s'.", trait_type)
        if not entries:
            logger.warning("No training data available for trait type '%s' "
                           "(perhaps all discarded?). Skipping training.", trait_type)
            continue

        try:
            features_list = [e[0] for e in entries]
            labels = [e[1] for e in entries]
            first_shape = features_list[0].shape
            if not all(f.shape == first_shape for f in features_list):
                logger.error("Inconsistent feature shapes detected for \
                    trait '%s'. Cannot train.",
                             trait_type)
                for i, f in enumerate(features_list):
                    if f.shape != first_shape:
                        logger.error("  - Feature %d shape: %s (Expected: %s)",
                                     i, f.shape, first_shape)
                continue
            feature_array = np.array(features_list)
        except IndexError:
            logger.error("Error accessing features for '%s' (IndexError). \
                Skipping training.",
                         trait_type)
            continue
        except ValueError as e:
            logger.error("Error creating numpy array for '%s': \
                Check feature consistency. "
                         "Error: %s. Skipping training.", trait_type, e)
            for i in range(min(5, len(features_list))):
                try:
                    logger.debug("  - Shape of feature %d for %s: %s",
                                 i, trait_type, features_list[i].shape)
                except IndexError:
                    pass
            continue

        n_samples = feature_array.shape[0]
        if n_samples == 0:
            logger.warning("Feature array is empty for '%s' after processing. \
                Skipping training.",
                           trait_type)
            continue
        n_features = feature_array.shape[1] if feature_array.ndim > 1 else 0

        unique_labels = set(labels)
        if len(unique_labels) < 2:
            logger.warning("Cannot train classifier for '%s': \
                Only one class ('%s') present "
                           "after data collection/verification. Skipping.",
                           trait_type, list(unique_labels)[0] \
                               if unique_labels else "N/A")
            continue

        pipeline_steps = []

        pipeline_steps.append(('scaler', StandardScaler()))

        if trait_type != "Background":
            max_pca_components = min(n_samples - 1, n_features)
            pca_n_components = min(50, max_pca_components)

            if pca_n_components > 1:
                logger.info("Using PCA with n_components=%d for trait '%s' "
                            "(n_samples=%d, n_features=%d)",
                            pca_n_components, trait_type, n_samples, n_features)
                pipeline_steps.append(('pca', \
                    PCA(n_components=pca_n_components, random_state=42)))
            else:
                logger.info("Skipping PCA for '%s': Insufficient samples/features or no "
                            "reduction possible (target components=%d, max=%d).",
                            trait_type, pca_n_components, max_pca_components)

        pipeline_steps.append(('svm', SVC(probability=True, random_state=42,
                                          class_weight='balanced')))

        pipeline = Pipeline(pipeline_steps)

        try:
            logger.info("Fitting pipeline for '%s' with %d samples...", \
                trait_type, n_samples)
            pipeline.fit(feature_array, labels)
            classifiers[trait_type] = pipeline

            logger.info("Saving trained classifier pipeline for '%s' to %s",
                        trait_type, model_path)
            with gzip.open(model_path, "wb") as f:
                pickle.dump(pipeline, f)
            logger.info("Successfully saved pipeline for '%s'.", trait_type)

        except (MemoryError, ValueError) as e:
            logger.error("Failed training classifier pipeline for '%s': \
                Sklearn error - %s. "
                         "Skipping save.", trait_type, e, exc_info=True)
        except (pickle.PickleError, OSError, IOError) as e:
            logger.error("Failed saving classifier pipeline for '%s' to %s: %s",
                         trait_type, model_path, e)
        except Exception as e:
            logger.error("Unexpected error training/saving pipeline for '%s': %s",
                         trait_type, e, exc_info=True)

    logger.info("--- Classifier Training Complete ---")
    trained_count = len(classifiers)
    total_possible_traits = len(all_trait_types)
    logger.info("Successfully trained/loaded %d out of %d trait classifiers.",
                trained_count, total_possible_traits)
    if trained_count < total_possible_traits:
        missing = set(all_trait_types) - set(classifiers.keys())
        logger.warning("Failed to train or load classifiers for: %s",
                       ", ".join(sorted(list(missing))))

def main() -> None:
    """Main execution function: Loads config, loads data, trains classifiers."""
    logger.info("Starting NFT Trait Classifier Trainer \
        (Part 1 - Improved with GUI Verification)")

    try:
        logger.info("Loading image map from %s", IMAGE_JSON_PATH)
        with open(IMAGE_JSON_PATH, "r", encoding="utf-8") as f:
            image_map = json.load(f)
        logger.info("Image map loaded successfully (%d entries).", len(image_map))
    except FileNotFoundError:
        logger.critical("CRITICAL: Image map file not found at %s. \
            Cannot proceed.",
                        IMAGE_JSON_PATH)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.critical("CRITICAL: Invalid JSON in image map file %s: %s. \
            Cannot proceed.",
                        IMAGE_JSON_PATH, e)
        sys.exit(1)
    except OSError as e:
        logger.critical("CRITICAL: OS error reading image map file %s: %s. \
            Cannot proceed.",
                        IMAGE_JSON_PATH, e)
        sys.exit(1)
    except Exception as e:
        logger.critical("CRITICAL: Unexpected error loading image map %s: %s. \
            Cannot proceed.",
                        IMAGE_JSON_PATH, e, exc_info=True)
        sys.exit(1)

    if not isinstance(image_map, dict):
        logger.critical("CRITICAL: Image map loaded from %s is not a dictionary. \
            Cannot proceed.",
                        IMAGE_JSON_PATH)
        sys.exit(1)
    if not image_map:
        logger.critical("CRITICAL: Image map loaded from %s is empty. \
            Cannot proceed.",
                        IMAGE_JSON_PATH)
        sys.exit(1)

    train_classifiers(image_map)

    logger.info("--- Script Finished ---")

if __name__ == "__main__":
    main()
