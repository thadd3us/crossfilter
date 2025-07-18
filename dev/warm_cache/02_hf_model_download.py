#!/usr/bin/env python3
"""
Script to pre-download SIGLIP2 models from HuggingFace for Docker warm cache.
This script only depends on the uv environment and does not import crossfilter code.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Download SIGLIP2 models from HuggingFace."""
    logger.info("Downloading SIGLIP2 models from HuggingFace...")
    
    try:
        # Import HuggingFace transformers
        from transformers import AutoModel, AutoTokenizer, AutoProcessor
        
        # SIGLIP2 model actually used in crossfilter source code
        models = [
            "google/siglip-so400m-patch14-384",
        ]
        
        for model_name in models:
            logger.info(f"Downloading model: {model_name}")
            
            # Download model - fail completely if this fails
            model = AutoModel.from_pretrained(model_name)
            logger.info(f"Model {model_name} downloaded successfully")
            
            # Download tokenizer if available
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info(f"Tokenizer for {model_name} downloaded successfully")
            except Exception as e:
                logger.warning(f"No tokenizer found for {model_name}: {e}")
            
            # Download processor if available
            try:
                processor = AutoProcessor.from_pretrained(model_name)
                logger.info(f"Processor for {model_name} downloaded successfully")
            except Exception as e:
                logger.warning(f"No processor found for {model_name}: {e}")
                
        logger.info("SIGLIP2 model download completed")
        
    except ImportError as e:
        logger.error(f"Failed to import required libraries: {e}")
        logger.error("Make sure transformers is installed in the environment")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during model download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()