"""
Data Loading and Processing Script

Load and process dataset using YAML configuration with integrated export functionality.

Usage:
    python data_from_yaml.py --config ./examples/data/config.yaml
"""

import argparse
from pathlib import Path

from loguru import logger

import rm_gallery.core.data  # noqa: F401 - needed for core strategy registration
import rm_gallery.gallery.data  # noqa: F401 - needed for gallery strategy registration
from rm_gallery.core.data.build import create_build_module_from_yaml


def load_and_process_dataset(config_path: str):
    """
    Load and process dataset using YAML configuration

    Args:
        config_path: Path to the YAML configuration file
        output_dir: Optional output directory override
    """
    try:
        logger.info("🚀 Starting data processing...")
        logger.info(f"📄 Loading config: {config_path}")

        # Validate config file exists
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Create builder from YAML config
        builder = create_build_module_from_yaml(config_path)
        logger.info(f"🔧 Created builder: {builder.name}")

        # Run the complete pipeline (Load → Process → Export)
        logger.info("⚡ Starting pipeline execution...")
        dataset = builder.run()

        logger.success("✅ Pipeline completed successfully!")
        logger.info(f"📊 Dataset: {dataset.name}")
        logger.info(f"📈 Processed: {len(dataset)} samples")

        return dataset

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Load and process dataset using YAML configuration"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./examples/data/config.yaml",
        help="Path to YAML configuration file",
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    logger.info("🎯 Data Processing Pipeline")
    logger.info(f"📋 Config: {args.config}")

    dataset = load_and_process_dataset(args.config)

    if dataset:
        logger.success("🎉 Data processing completed successfully!")
        logger.info(
            "💡 Export files should be available in the configured export directory"
        )
        logger.info("📁 Check your YAML config for export settings")
    else:
        logger.error("❌ Data processing failed!")
        logger.info("💡 Check the error messages above for troubleshooting")


if __name__ == "__main__":
    main()
