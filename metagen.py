# pylint: disable=broad-exception-caught, too-many-locals
"""
Part 2 - Trait Prediction & Metadata Generation for Unrevealed Tokens (Improved)
--------------------------------------------------------------------------------
This script uses pretrained classifier pipelines from Part 1 (Improved) to analyze
remaining unrevealed NFT images, predict their traits, and generate corresponding
metadata files. It performs the following tasks:

1. Loads pre-trained classifier pipelines from the specified directory.
2. Fetches images from IPFS based on image mappings in `images.json`.
3. Extracts appropriate features (vector or histogram) and predicts traits using pipelines.
4. Generates standardized metadata files per token ID (1295-10000).
5. Saves output to `metadata/` (no .json extension), with checkpoints and logging.

Free to use, no API keys or external paid services required.
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
from io import BytesIO
from time import sleep
from typing import Optional, Dict, Any, List
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

BASE_DIR = "C:/bedlam/md_gen_v4"
LOG_DIR = os.path.join(BASE_DIR, "logs")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CLASSIFIER_DIR = os.path.join(CONFIG_DIR, "classifiers")
IMAGE_MAP_FILE = os.path.join(CONFIG_DIR, "images.json")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "part2_checkpoint.json")
LOG_FILE = os.path.join(LOG_DIR, "trait_predictor_v2.log") # Consistent log name
REQUEST_TIMEOUT = 20
IMAGE_SIZE = (128, 128)
HISTOGRAM_BINS = 16
TOKEN_START_PREDICT = 1295
TOKEN_END_PREDICT = 10000
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def load_pickle_pipeline(file_path: str) -> Optional[Pipeline]:
    """
    Load a gzip-compressed
    pickle object (sklearn
    Pipeline) from file.
    """
    logger.debug("Attempting to load pipeline from: %s", file_path)
    try:
        with gzip.open(file_path, 'rb') as f:
            pipeline = pickle.load(f)
            if isinstance(pipeline, Pipeline) and hasattr(pipeline, 'predict'):
                try:
                    _ = getattr(pipeline[-1], "classes_", None)
                    pipeline.predict_proba(np.zeros((1, \
                        pipeline.steps[0][1].n_features_in_)))
                    logger.debug("Pipeline loaded and appears valid: %s", file_path)
                    return pipeline
                except NotFittedError:
                    logger.error("Loaded pipeline from %s is not fitted.", file_path)
                    return None
                except AttributeError:
                    logger.warning("Could not fully validate if pipeline from \
                        %s is fitted.", file_path)
                    return pipeline
            else:
                logger.error("Loaded object from %s is not a valid scikit-learn \
                    Pipeline.", file_path)
                return None
    except FileNotFoundError:
        logger.error("Pipeline file not found at %s.", file_path)
        return None
    except (pickle.PickleError, EOFError, gzip.BadGzipFile, OSError, ValueError) as e:
        logger.error("Failed to load or validate pipeline from %s: %s", file_path, e)
        return None
    except Exception as e:
        logger.error("Unexpected error loading pipeline from %s: %s", \
            file_path, e, exc_info=True)
        return None

def load_pipelines() -> Dict[str, Optional[Pipeline]]:
    """
    Load all classifier pipelines from the specified CLASSIFIER_DIR.
    Determines trait types based on filenames. Returns a dictionary mapping
    trait type (string) to the loaded Pipeline object or None if loading failed.
    """
    pipelines: Dict[str, Optional[Pipeline]] = {}
    trait_types: List[str] = []

    if not os.path.isdir(CLASSIFIER_DIR):
        logger.error("Classifier directory not found: %s. \
            Cannot load models.", CLASSIFIER_DIR)
        return {}

    logger.info("Scanning for classifier models in: %s", CLASSIFIER_DIR)
    try:
        model_files = [f for f in os.listdir(CLASSIFIER_DIR) if f.endswith('.pkl.gz')]
        trait_types = sorted([f.replace('.pkl.gz', '').replace('_', ' ') \
            for f in model_files])
    except OSError as e:
        logger.error("Error reading classifier directory %s: %s", CLASSIFIER_DIR, e)
        return {}

    if not trait_types:
        logger.error("No classifier files (.pkl.gz) found in %s.", CLASSIFIER_DIR)
        return {}

    logger.info("Found %d potential trait types: %s", \
        len(trait_types), ", ".join(trait_types))

    loaded_count = 0
    for trait in trait_types:
        sanitized_trait = trait.replace(' ', '_').replace('/', '_')
        model_path = os.path.join(CLASSIFIER_DIR, f"{sanitized_trait}.pkl.gz")

        logger.info("Attempting to load pipeline for trait '%s'...", trait)
        pipeline = load_pickle_pipeline(model_path)
        pipelines[trait] = pipeline
        if pipeline is not None:
            logger.info("Successfully loaded pipeline for trait '%s'.", trait)
            loaded_count += 1
        else:
            logger.error("Failed to load pipeline for trait \
                '%s' from %s.", trait, model_path)

    logger.info("Finished loading pipelines. \
        Successfully loaded %d out of %d.", loaded_count, len(trait_types))
    return pipelines

def get_checkpoint() -> int:
    """
    Return last processed 
    token ID from 
    checkpoint file.
    """
    default_start_token = TOKEN_START_PREDICT - 1
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                last_processed = data.get("last_processed", \
                    default_start_token)
                if not isinstance(last_processed, int) or \
                    last_processed < default_start_token:
                    logger.warning("Invalid checkpoint value '%s' found. \
                        Resetting.", last_processed)
                    return default_start_token
                logger.info("Checkpoint found. Last processed token: \
                    %d", last_processed)
                return last_processed
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Error reading checkpoint file %s: %s. \
                Starting from %d.",
                         CHECKPOINT_FILE, e, default_start_token)
            return default_start_token
        except Exception as e:
            logger.error("Unexpected error reading checkpoint file \
                %s: %s. Starting from %d.",
                          CHECKPOINT_FILE, e, default_start_token, exc_info=True)
            return default_start_token

    logger.info("No checkpoint file found. Starting from token %d.", \
        default_start_token + 1)
    return default_start_token

def update_checkpoint(token_id: int):
    """
    Save the last successfully
    processed token ID to
    the checkpoint file.
    """
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump({"last_processed": token_id}, f, indent=2)
        logger.debug("Checkpoint updated to token %d", token_id)
    except OSError as e:
        logger.error("Failed to update checkpoint file %s: %s", CHECKPOINT_FILE, e)
    except Exception as e:
        logger.error("Unexpected error updating checkpoint file %s: %s", \
            CHECKPOINT_FILE, e, exc_info=True)

def load_image(url: str) -> Optional[Image.Image]:
    """
    Downloads, opens, converts an image from a given URL.
    Returns the image resized to IMAGE_SIZE. Handles errors.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            if response.status != 200:
                logger.error("Failed to load image from URL %s: \
                    HTTP Status %d", url, response.status)
                return None
            image_data = response.read()
            with BytesIO(image_data) as image_stream:
                img = Image.open(image_stream).convert("RGB")
                if img.size != IMAGE_SIZE:
                    logger.debug("Resizing image from %s to %s", \
                        img.size, IMAGE_SIZE)
                    img = img.resize(IMAGE_SIZE, \
                        Image.Resampling.LANCZOS)
                return img
    except urllib.error.HTTPError as e:
        logger.error("Failed to load image from URL %s: HTTPError %s - %s", \
            url, e.code, e.reason)
        return None
    except urllib.error.URLError as e:
        reason = e.reason if isinstance(e.reason, str) \
            else getattr(e.reason, 'strerror', str(e.reason))
        if isinstance(e.reason, socket.timeout) or 'timed out' in reason.lower():
            logger.error("Failed to load image from URL %s: \
                Request timed out after %d seconds", url, REQUEST_TIMEOUT)
        else:
            logger.error("Failed to load image from URL %s: \
                URLError - %s", url, reason)
        return None
    except socket.timeout:
        logger.error("Failed to load image from URL %s: \
            Socket timed out after %d seconds", url, REQUEST_TIMEOUT)
        return None
    except UnidentifiedImageError:
        logger.error("Failed to load image from URL %s: \
            Cannot identify image file", url)
        return None
    except (OSError, IOError, ValueError) as e:
        logger.error("Failed processing image from URL %s: %s", url, e)
        return None
    except Exception as e:
        logger.error("Unexpected error loading image from URL \
            %s: %s", url, e, exc_info=True)
        return None

def image_to_vector(image: Image.Image) -> Optional[np.ndarray]:
    """
    Converts an image (assumed to be correct IMAGE_SIZE)
    into a flattened numpy array vector. Returns None on failure.
    """
    if image is None:
        logger.error("image_to_vector received None image.")
        return None
    try:
        if image.size != IMAGE_SIZE:
            logger.warning("Image provided to image_to_vector is not \
                the expected size %s. Resizing.", IMAGE_SIZE)
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        vector = np.asarray(image).flatten()
        logger.debug("Image converted to vector shape: %s", vector.shape)
        return vector
    except Exception as e:
        logger.error("Failed to convert image to vector: %s", e, exc_info=True)
        return None

def image_to_color_histogram(image: Image.Image, \
    bins: int = HISTOGRAM_BINS) -> Optional[np.ndarray]:
    """
    Calculates a flattened, normalized 3D color histogram for an RGB image
    (assumed to be correct IMAGE_SIZE). Returns None on failure.
    """
    if image is None:
        logger.error("image_to_color_histogram received None image.")
        return None
    try:
        if image.size != IMAGE_SIZE:
            logger.warning("Image provided to image_to_color_histogram \
                is not the expected size %s. Resizing.", IMAGE_SIZE)
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

        img_array = np.asarray(image)
        hist_r = np.histogram(img_array[:, :, 0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:, :, 1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:, :, 2], bins=bins, range=(0, 256))[0]

        total_pixels = image.width * image.height
        if total_pixels == 0:
            logger.error("Cannot calculate histogram for zero-pixel image.")
            return None

        hist_r = hist_r / total_pixels
        hist_g = hist_g / total_pixels
        hist_b = hist_b / total_pixels

        histogram_vector = np.concatenate((hist_r, hist_g, hist_b))
        logger.debug("Image converted to histogram shape: \
            %s", histogram_vector.shape)
        return histogram_vector
    except Exception as e:
        logger.error("Failed to calculate color histogram: \
            %s", e, exc_info=True)
        return None

def generate_metadata(token_id: int, traits: Dict[str, Any], \
    image_url: str, trait_order: List[str]) -> Dict[str, Any]:
    """
    Create the metadata JSON structure for a token, ordering traits correctly.
    Args:
        token_id: The ID of the token.
        traits: A dictionary of predicted trait_type: value pairs.
        image_url: The IPFS URL for the token's image.
        trait_order: A list of trait types in the desired output order.
    Returns:
        A dictionary representing the token's metadata.
    """
    attributes = []
    one_of_one_trait = traits.get("One Of One")
    special_trait = traits.get("Special")

    if one_of_one_trait and str(one_of_one_trait).lower() != "none":
        logger.info("Token %d identified as One Of One: '%s'", \
            token_id, one_of_one_trait)
        attributes = [{"trait_type": "One Of One", "value": \
            str(one_of_one_trait)}]
    elif special_trait and str(special_trait).lower() != "none":
        logger.info("Token %d identified as Special: \
            '%s'", token_id, special_trait)
        attributes = [{"trait_type": "Special", "value": \
            str(special_trait)}]
    else:
        logger.debug("Generating standard attributes for token %d \
            based on order: %s", token_id, trait_order)
        for trait in trait_order:
            if trait in ["One Of One", "Special"]:
                continue

            value = traits.get(trait)
            if value is not None and str(value).lower() != "none":
                attributes.append({"trait_type": trait, \
                    "value": str(value)})
            elif value is None:
                logger.debug("Trait '%s' not predicted for token %d.", \
                    trait, token_id)
            else:
                logger.debug("Skipping trait '%s' for token %d \
                    because predicted value was 'None'.", trait, token_id)

    if not attributes:
        logger.warning("No valid traits predicted or available for token %d. \
            Metadata will have empty attributes list.", token_id)

    metadata = {
        "name": f"Samsquanch #{token_id}",
        "description": "Dirty Samsquanch Fiens, Greasy Bastards",
        "attributes": attributes,
        "image": image_url,
        "external_url": "https://samsquanchfiens.xyz",
        "seller_fee_basis_points": 500,
        "fee_recipient": "0x0000000000000000000000000000000000000000"
    }
    logger.debug("Generated metadata structure for token %d: \
        %s", token_id, metadata)
    return metadata

def main():
    """
    Loads pipelines, processes tokens,
    predicts traits, and saves metadata.
    """
    logger.info("--- Starting Part 2: Trait Prediction & Metadata Generation ---")

    pipelines = load_pipelines()
    if not pipelines or all(p is None for p in pipelines.values()):
        logger.critical("CRITICAL: No valid classifier pipelines were loaded. \
            Cannot proceed. Ensure Part 1 ran successfully \
                and models exist in %s.", CLASSIFIER_DIR)
        sys.exit(1)

    valid_trait_types = sorted([trait for trait, \
        p in pipelines.items() if p is not None])
    if not valid_trait_types:
        logger.critical("CRITICAL: No pipelines loaded successfully. \
            Cannot proceed.")
        sys.exit(1)

    missing_traits = sorted(list(set(pipelines.keys()) \
        - set(valid_trait_types)))
    if missing_traits:
        logger.warning("Pipelines for the following traits \
            could not be loaded and will be skipped: \
                %s", ", ".join(missing_traits))
    logger.info("Will predict for traits: \
        %s", ", ".join(valid_trait_types))

    try:
        logger.info("Loading image map from %s", \
            IMAGE_MAP_FILE)
        with open(IMAGE_MAP_FILE, 'r', encoding='utf-8') as f:
            image_map = json.load(f)
        logger.info("Image map loaded successfully \
            (%d entries).", len(image_map))
    except FileNotFoundError:
        logger.critical("CRITICAL: Image map file not found: \
            %s. Cannot proceed.", IMAGE_MAP_FILE)
        sys.exit(1)
    except (json.JSONDecodeError, OSError) as e:
        logger.critical("CRITICAL: Error reading image map file %s: \
            %s. Cannot proceed.", IMAGE_MAP_FILE, e)
        sys.exit(1)
    except Exception as e:
        logger.critical("CRITICAL: Unexpected error loading image map %s: \
            %s. Cannot proceed.", IMAGE_MAP_FILE, e, exc_info=True)
        sys.exit(1)

    if not isinstance(image_map, dict):
        logger.critical("CRITICAL: Image map loaded from %s is not a dictionary. \
            Cannot proceed.", IMAGE_MAP_FILE)
        sys.exit(1)
    if not image_map:
        logger.critical("CRITICAL: Image map loaded from %s is empty. \
            Cannot proceed.", IMAGE_MAP_FILE)
        sys.exit(1)

    last_processed_token = get_checkpoint()
    start_token = last_processed_token + 1
    end_token = TOKEN_END_PREDICT
    logger.info("Starting prediction from token ID %d \
        up to %d (inclusive).", start_token, end_token)

    if start_token > end_token:
        logger.info("All tokens up to %d already processed based on checkpoint. \
            Nothing to do.", end_token)
        logger.info("--- Trait prediction script finished ---")
        return

    tokens_processed_this_run = 0
    tokens_failed_this_run = 0
    for token_id in range(start_token, end_token + 1):
        logger.info("----- Processing Token ID: \
            %d -----", token_id)
        key = f"tokenId {token_id}"
        ipfs_url = image_map.get(key)

        if not ipfs_url:
            logger.warning("No image URL found for token %d in %s. \
                Skipping.", token_id, IMAGE_MAP_FILE)
            tokens_failed_this_run += 1
            continue

        logger.debug("Image URL: %s", ipfs_url)

        try:
            image = load_image(ipfs_url)
            if image is None:
                logger.warning("Skipping token %d due to \
                    image loading failure.", token_id)
                tokens_failed_this_run += 1
                continue

            vector_features = image_to_vector(image)
            histogram_features = image_to_color_histogram(image)

            if vector_features is None and histogram_features is None:
                logger.warning("Skipping token %d due to \
                    all feature extraction failing.", token_id)
                tokens_failed_this_run += 1
                continue

            predicted_traits: Dict[str, Any] = {}
            prediction_successful = False
            for trait_type in valid_trait_types:
                pipeline = pipelines[trait_type]

                if trait_type == "Background":
                    features = histogram_features
                    feature_type = "histogram"
                else:
                    features = vector_features
                    feature_type = "vector"

                if features is None:
                    logger.warning("Skipping trait '%s' for token %d: \
                        Required %s features failed extraction.",
                                   trait_type, token_id, feature_type)
                    continue

                try:
                    features_reshaped = features.reshape(1, -1)
                    logger.debug("Predicting trait '%s' using %s \
                        features (shape: %s)...",
                                 trait_type, feature_type, features_reshaped.shape)

                    prediction = pipeline.predict(features_reshaped)[0]
                    predicted_traits[trait_type] = prediction
                    prediction_successful = True
                    logger.info("Token %d, Trait '%s': \
                        Predicted -> '%s'", token_id, trait_type, prediction)

                except NotFittedError:
                    logger.error("CRITICAL: Pipeline for trait '%s' is not fitted! \
                        Skipping prediction for token %d.",
                                 trait_type, token_id)
                except ValueError as ve:
                    try:
                        expected_features = pipeline.steps[0][1].n_features_in_
                    except (AttributeError, IndexError):
                        expected_features = "N/A"
                    logger.error("Prediction failed for trait '%s', token %d: \
                        %s. Check feature dimensions (Expected: %s, Got: %d).",
                                 trait_type, token_id, ve, expected_features, \
                                     features_reshaped.shape[1])
                except Exception as e:
                    logger.error("Unexpected error predicting trait \
                        '%s' for token %d: %s",
                                 trait_type, token_id, e, exc_info=True)

            if not prediction_successful:
                logger.warning("No traits were successfully predicted for token %d. \
                    Skipping metadata generation.", token_id)
                tokens_failed_this_run += 1
                continue

            metadata = generate_metadata(token_id, predicted_traits, \
                ipfs_url, valid_trait_types)
            output_path = os.path.join(METADATA_DIR, str(token_id))

            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Successfully generated and saved metadata \
                    for token %d to %s", token_id, output_path)
                update_checkpoint(token_id)
                tokens_processed_this_run += 1

            except OSError as e:
                logger.error("Failed to write metadata for token %d to %s: \
                    %s", token_id, output_path, e)
                tokens_failed_this_run += 1
            except TypeError as e:
                logger.error("Failed to serialize metadata for token %d: \
                    %s", token_id, e)
                tokens_failed_this_run += 1
            except Exception as e:
                logger.error("Unexpected error saving metadata for token %d: \
                    %s", token_id, e, exc_info=True)
                tokens_failed_this_run += 1

        except Exception as e:
            logger.error("Unhandled error processing token %d: \
                %s", token_id, e, exc_info=True)
            tokens_failed_this_run += 1
            sleep(1)

    logger.info("--- Trait prediction script finished ---")
    logger.info("Summary: Processed %d tokens successfully, \
        Failed/Skipped %d tokens in this run.",
                tokens_processed_this_run, tokens_failed_this_run)

if __name__ == "__main__":
    main()
