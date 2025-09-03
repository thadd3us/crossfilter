#!/usr/bin/env python3
"""
Script to pre-download SIGLIP2 models from HuggingFace for Docker warm cache.
This script only depends on the uv environment and does not import crossfilter code.
"""

import logging
import sys
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Download SIGLIP2 models from HuggingFace."""
    logger.info("Downloading SIGLIP2 models from HuggingFace...")

    # Import HuggingFace transformers
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    # SIGLIP2 model actually used in crossfilter source code
    models = [
        "google/siglip-so400m-patch14-384",
    ]

    for model_name in models:
        logger.info(f"Downloading model: {model_name}")

        # Download model - fail completely if this fails
        AutoModel.from_pretrained(model_name)
        logger.info(f"Model {model_name} downloaded successfully")

        # Download tokenizer if available
        AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Tokenizer for {model_name} downloaded successfully")

        # Download processor if available
        AutoProcessor.from_pretrained(model_name)
        logger.info(f"Processor for {model_name} downloaded successfully")

    # Clear the HuggingFace cache to save space.
    hf_cache = Path("~/.cache/huggingface/xet").expanduser()
    assert hf_cache.exists(), f"HuggingFace cache directory {hf_cache} does not exist"
    shutil.rmtree(hf_cache, ignore_errors=True)

    # Make sure it still works.
    AutoModel.from_pretrained(model_name, local_files_only=True)
    logger.info(f"Model {model_name} loaded successfully")

if __name__ == "__main__":
    main()
