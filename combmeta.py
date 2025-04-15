# pylint: disable=broad-exception-caught
"""
Part 3 - Combine All Metadata into a Nested JSON Structure
---------------------------------------------------------
This script will combine individual metadata files 
(from token 1 to 10000) into a single, properly nested
JSON structure. The output is a JSON file that contains
all token metadata under a single `"tokens"` key.
- Input: All metadata files located in `metadata/`.
- Output: Combined JSON file named `all_metadata.json`.
"""
import os
import json
import logging
import sys

BASE_DIR = "C:/bedlam/md_gen_v4"
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_FILE = os.path.join(METADATA_DIR, "all_metadata.json")
LOG_FILE = os.path.join(LOG_DIR, "combine_metadata.log")
os.makedirs(LOG_DIR, exist_ok=True)
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

def combine_metadata():
    """
    Combines individual metadata 
    files into a single nested 
    JSON structure.
    """
    logger.info("--- Starting Metadata Combination ---")
    combined_metadata = {"tokens": []}

    if not os.path.isdir(METADATA_DIR):
        logger.critical("CRITICAL: Metadata directory '%s' \
            not found. Cannot proceed.", METADATA_DIR)
        return

    logger.info("Scanning directory: %s", METADATA_DIR)
    try:
        filenames = sorted(
            [f for f in os.listdir(METADATA_DIR) if f.isdigit()],
            key=int
        )
        logger.info("Found %d potential metadata files (numeric names).", \
            len(filenames))
    except OSError as e:
        logger.critical("CRITICAL: Error listing directory '%s': \
            %s. Cannot proceed.", METADATA_DIR, e)
        return
    except Exception as e:
        logger.critical("CRITICAL: Unexpected error listing directory '%s': \
            %s. Cannot proceed.", METADATA_DIR, e, exc_info=True)
        return

    if not filenames:
        logger.warning("No metadata files with numeric names found in %s. \
            Output will be empty.", METADATA_DIR)

    files_processed = 0
    files_failed = 0
    for filename in filenames:
        token_id_str = filename
        try:
            token_id = int(token_id_str)
        except ValueError:
            logger.warning("Skipping file with non-integer name \
                '%s' that passed filter.", filename)
            files_failed += 1
            continue

        file_path = os.path.join(METADATA_DIR, filename)
        logger.debug("Processing file: %s (Token ID: %d)", \
            file_path, token_id)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            if not isinstance(metadata, dict):
                logger.warning("Metadata in file %s is not a dictionary. \
                    Skipping.", filename)
                files_failed += 1
                continue

            combined_metadata["tokens"].append({
                "tokenId": str(token_id),
                "metadata": metadata
            })
            files_processed += 1

        except json.JSONDecodeError as e:
            logger.error("Error decoding JSON from file %s: %s", filename, e)
            files_failed += 1
        except (OSError, IOError) as e:
            logger.error("Error reading file %s: %s", filename, e)
            files_failed += 1
        except Exception as e:
            logger.error("Unexpected error processing file %s: \
                %s", filename, e, exc_info=True)
            files_failed += 1

    logger.info("Finished processing individual files. Processed: \
        %d, Failed/Skipped: %d", files_processed, files_failed)

    logger.info("Attempting to write combined metadata to: %s", OUTPUT_FILE)
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(combined_metadata, f, indent=2)

        logger.info("Successfully combined %d tokens into %s", \
            len(combined_metadata["tokens"]), OUTPUT_FILE)

    except (OSError, IOError) as e:
        logger.error("CRITICAL: Error writing combined metadata to %s: \
            %s", OUTPUT_FILE, e)
    except TypeError as e:
        logger.error("CRITICAL: Error serializing combined metadata for %s: \
            %s", OUTPUT_FILE, e)
    except Exception as e:
        logger.error(
            "CRITICAL: Unexpected error writing combined metadata to %s: %s",
            OUTPUT_FILE, e, exc_info=True
        )

    logger.info("--- Metadata Combination Finished ---")

if __name__ == "__main__":
    combine_metadata()
