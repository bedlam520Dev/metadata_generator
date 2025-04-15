# pylint: disable=broad-exception-caught
"""
This script validates, dynamically reorders, and standardizes the indentation
of NFT metadata JSON files in a directory. It processes both individual token
metadata files (named by ID) and a combined metadata file (typically
'all_metadata.json').

For each file, it ensures that the `attributes` list follows a strict trait-type
priority defined by TRAIT_PRIORITY. It preserves extra or missing traits without
causing errors.

The script overwrites files in place after processing, ensuring consistent
indentation (indent=2) for all files, regardless of whether attribute reordering
occurred. It adheres strictly to Pylint standards.
"""
import os
import json
import logging
import sys

BASE_DIR = "C:/bedlam/md_gen_v4"
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "metadata_reorder_standardize.log")
TARGET_DIRECTORY = METADATA_DIR # Use METADATA_DIR constant
TRAIT_PRIORITY = [
    "Background",
    "Fur",
    "Clothing",
    "Mouth",
    "Moon Mark",
    "Eyes",
    "Accessories",
    "Special",
    "One Of One"
]

try:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TARGET_DIRECTORY, exist_ok=True)
except OSError as e:
    print(f"CRITICAL: Error creating directories: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def reorder_attributes(attributes: list) -> tuple[list, bool]:
    """
    Dynamically reorder the list of attribute dictionaries based on TRAIT_PRIORITY.
    Missing traits are skipped, and unknown ones are preserved at the end in their
    original relative order. Handles non-dict items gracefully.

    Args:
        attributes (list): List of trait dictionaries
                           (e.g., [{"trait_type": "Fur", "value": "..."}]).

    Returns:
        tuple[list, bool]: A tuple containing:
                           - The reordered list of trait dictionaries.
                           - A boolean indicating if any reordering occurred.
    """
    if not isinstance(attributes, list):
        logger.warning("Input 'attributes' is not a list. Skipping reordering.")
        return attributes, False

    original_order_map = {
        i: attr for i, attr in enumerate(attributes)
        if isinstance(attr, dict) and "trait_type" in attr
    }
    trait_map = {
        attr["trait_type"]: (i, attr) for i, attr in original_order_map.items()
    }

    reordered = []
    processed_indices = set()

    for trait in TRAIT_PRIORITY:
        if trait in trait_map:
            original_index, attr = trait_map.pop(trait)
            reordered.append(attr)
            processed_indices.add(original_index)

    remaining_sorted_by_index = sorted(
        trait_map.values(), key=lambda item: item[0]
    )
    for original_index, attr in remaining_sorted_by_index:
        reordered.append(attr)
        processed_indices.add(original_index)

    original_valid_attributes = [attributes[i] for i in sorted(original_order_map.keys())]
    was_reordered = reordered != original_valid_attributes

    non_conforming_items = [
        attr for i, attr in enumerate(attributes)
        if i not in original_order_map
    ]
    reordered.extend(non_conforming_items)

    return reordered, was_reordered

def process_json_file(file_path: str):
    """
    Load, check structure, reorder attributes if necessary, and save a JSON file
    with standardized indentation (indent=2). Handles both individual token files
    and the combined 'all_metadata.json' structure.
    Args:
        file_path (str): Full path to the JSON file.
    """
    logger.debug("Attempting to process file: %s", file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    except json.JSONDecodeError as error:
        logger.error("Error decoding JSON in %s: %s", file_path, error)
        return
    except OSError as error:
        logger.error("Error reading file %s: %s", file_path, error)
        return
    except Exception as error:
        logger.error("Unexpected error loading %s: %s", file_path, error, exc_info=True)
        return

    file_modified = False
    log_prefix = ""

    if isinstance(data, dict) and "tokens" in data and isinstance(data["tokens"], list):
        logger.debug("Processing combined file structure: %s", file_path)
        combined_file_changed_overall = False
        for token_obj in data["tokens"]:
            if (isinstance(token_obj, dict) and
                    "metadata" in token_obj and
                    isinstance(token_obj["metadata"], dict) and
                    "attributes" in token_obj["metadata"]):

                attributes = token_obj["metadata"].get("attributes", [])
                reordered_attrs, item_was_reordered = reorder_attributes(attributes)

                if item_was_reordered:
                    token_obj["metadata"]["attributes"] = reordered_attrs
                    combined_file_changed_overall = True
            else:
                token_id = token_obj.get('tokenId', 'Unknown') \
                    if isinstance(token_obj, dict) else 'Unknown'
                logger.warning("Skipping invalid token structure within %s (ID: %s)",
                                file_path, token_id)

        file_modified = combined_file_changed_overall
        log_prefix = "(combined) "

    elif isinstance(data, dict) and "attributes" in data:
        logger.debug("Processing individual file structure: %s", file_path)
        attributes = data.get("attributes", [])
        reordered_attrs, item_was_reordered = reorder_attributes(attributes)

        if item_was_reordered:
            data["attributes"] = reordered_attrs
            file_modified = True
        log_prefix = ""

    else:
        logger.warning("Skipped saving (unrecognized structure or \
            missing 'attributes'): %s", file_path)
        return

    try:
        with open(file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=2)

        action_taken = "Reordered attributes and saved" if file_modified else \
            "Checked attributes and saved (indent standardized)"
        logger.info("%s%s: %s", log_prefix, action_taken, os.path.basename(file_path))

    except OSError as error:
        logger.error("Error saving file %s: %s", file_path, error)
    except TypeError as error:
        logger.error("Error serializing data for %s: %s", file_path, error)
    except Exception as error:
        logger.error("Unexpected error saving %s: %s", file_path, error, exc_info=True)

def process_directory(directory_path: str):
    """
    Iterate through a directory and process all valid JSON metadata files
    (named numerically or ending in .json).
    Args:
        directory_path (str): Path to the folder containing metadata files.
    """
    if not os.path.isdir(directory_path):
        logger.critical("Target directory not found: %s. Cannot proceed.", directory_path)
        return

    logger.info("Starting processing in directory: %s", directory_path)
    processed_count = 0
    skipped_count = 0

    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if os.path.isfile(file_path):
                if filename.isdigit() or filename.lower().endswith(".json"):
                    logger.debug("Found potential metadata file: %s", filename)
                    process_json_file(file_path)
                    processed_count += 1
                else:
                    logger.debug("Skipping non-metadata file: %s", filename)
                    skipped_count += 1
            else:
                logger.debug("Skipping directory entry: %s", filename)
                skipped_count += 1
    except OSError as e:
        logger.error("Error reading directory %s: %s", directory_path, e)
    except Exception as e:
        logger.error("Unexpected error during directory processing %s: %s", \
            directory_path, e, exc_info=True)


    logger.info("Finished processing directory.")
    logger.info("Attempted to process %d files, skipped %d non-metadata items.",
                 processed_count, skipped_count)

if __name__ == "__main__":
    logger.info("--- Script Execution Started ---")
    process_directory(TARGET_DIRECTORY)
    logger.info("--- Script Execution Finished ---")
