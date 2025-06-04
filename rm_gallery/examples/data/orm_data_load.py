import json
from pathlib import Path

from loguru import logger

import rm_gallery.core.data.ops  # noqa: F401 - needed for strategy registration
from rm_gallery.core.data.build import create_build_module_from_yaml


def convert_to_jsonl_format(data):
    """Convert BaseData object to JSONL format"""
    return data.model_dump(mode="json")


def load_and_process_dataset():
    """load and process orm dataset"""
    config_path = "./rm_gallery/examples/data/orm_data_load.yaml"

    try:
        logger.info(f"Loading dataset from config: {config_path}")

        builder = create_build_module_from_yaml(config_path)
        logger.info(f"Created builder: {builder.name}")

        logger.info("Starting pipeline execution...")
        dataset = builder.run()

        output_path = f"{dataset.name.replace('-', '_')}_output.jsonl"
        export_dataset(dataset, output_path)
        return dataset

    except Exception as e:
        logger.error(f"❌ Error in dataset processing: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def export_dataset(dataset, output_path):
    if not dataset or not dataset.datas:
        logger.warning("No data to export")
        return

    output_path = Path(output_path)
    written_count = 0

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for data in dataset.datas:
                json_data = convert_to_jsonl_format(data)
                f.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                written_count += 1

        logger.info(f"✅ Successfully exported {written_count} items to {output_path}")

    except Exception as e:
        logger.error(f"❌ Error exporting to JSONL: {str(e)}")


def main():
    logger.info("🚀 Starting ORM data processing...")
    dataset = load_and_process_dataset()

    if dataset:
        logger.info("✅ ORM data processing completed successfully!")
    else:
        logger.error("❌ ORM data processing failed!")


if __name__ == "__main__":
    main()
