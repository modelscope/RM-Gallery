"""
Annotation Pipeline Example for RM Gallery

This module demonstrates how to use annotation as a stage in the build pipeline.
It provides examples of:
1. Data pipeline without annotation
2. Data pipeline with annotation integration
3. Exporting annotation results
4. Service health checks

Usage:
    python annotation_pipeline.py [--with-annotation] [--export-only] [--check-service]

example:
    # Run basic pipeline (without annotation)
    python annotation_pipeline.py

    # Run pipeline with annotation
    python annotation_pipeline.py --with-annotation --api-token YOUR_TOKEN

    # Check service status only
    python annotation_pipeline.py --check-service

    # Export annotation results only
    python annotation_pipeline.py --export-only --project-id 123 --api-token YOUR_TOKEN

    # Use custom configuration
    python annotation_pipeline.py --config config.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from loguru import logger

import rm_gallery.core.data.ops  # noqa: F401 - needed for strategy registration
from rm_gallery.core.data.annotation import create_annotation_module
from rm_gallery.core.data.build import create_build_module
from rm_gallery.core.data.config.label_studio_config import REWARD_BENCH_LABEL_CONFIG
from rm_gallery.core.data.load import create_load_module
from rm_gallery.core.data.process import OperatorFactory, create_process_module

# Configuration Constants
DEFAULT_CONFIG = {
    "data_source": {
        "path": "./data/preference-test-sets/data/anthropic_helpful-00000-of-00001.parquet",
        "limit": 2000,
    },
    "label_studio": {
        "server_url": "http://localhost:8080",
        "api_token": "your_api_token",  # Replace with actual token
        "project_title": "RM Gallery Quality Annotation",
    },
    "filters": {
        "conversation_turns": {"min_turns": 2, "max_turns": 6},
        "text_length": {"min_length": 50, "max_length": 2000},
        "data_juicer_length": {"min_len": 50, "max_len": 100},
        "group_split": {"train_ratio": 0.7, "test_ratio": 0.3},
    },
}


class AnnotationPipelineExample:
    """Main class for annotation pipeline examples"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG

    def create_load_module(self, limit: int = None):
        """Create and configure load module"""
        limit = limit or self.config["data_source"]["limit"]

        return create_load_module(
            name="preference-test-sets",
            config={},
            load_strategy_type="local",
            data_source="rewardbench",
            load_config={
                "path": self.config["data_source"]["path"],
                "limit": limit,
            },
        )

    def create_basic_process_module(self):
        """Create basic processing module with common filters"""
        return create_process_module(
            name="preference-test-sets-processor",
            config={},
            operators=[
                OperatorFactory.create_operator(
                    {
                        "type": "filter",
                        "name": "conversation_turn_filter",
                        "config": self.config["filters"]["conversation_turns"],
                    }
                ),
                OperatorFactory.create_operator(
                    {
                        "type": "filter",
                        "name": "rm_text_length_filter",
                        "config": self.config["filters"]["text_length"],
                    }
                ),
                OperatorFactory.create_operator(
                    {
                        "type": "data_juicer",
                        "name": "text_length_filter",
                        "config": self.config["filters"]["data_juicer_length"],
                    }
                ),
                OperatorFactory.create_operator(
                    {
                        "type": "group",
                        "name": "group_train",
                        "config": self.config["filters"]["group_split"],
                    }
                ),
            ],
        )

    def create_annotation_process_module(self):
        """Create processing module optimized for annotation"""
        return create_process_module(
            name="annotation-preprocessor",
            config={},
            operators=[
                OperatorFactory.create_operator(
                    {
                        "type": "filter",
                        "name": "conversation_turn_filter",
                        "config": self.config["filters"]["conversation_turns"],
                    }
                ),
                OperatorFactory.create_operator(
                    {
                        "type": "filter",
                        "name": "rm_text_length_filter",
                        "config": self.config["filters"]["text_length"],
                    }
                ),
            ],
        )

    def create_annotation_module(self, api_token: str = None):
        """Create annotation module with Label Studio integration"""
        api_token = api_token or self.config["label_studio"]["api_token"]

        if api_token == "your_api_token":
            logger.warning("⚠️  Please set a valid API token!")
            logger.info(
                "Get your token from: http://localhost:8080 -> Account & Settings"
            )

        return create_annotation_module(
            name="rm_gallery_annotation",
            api_token=api_token,
            server_url=self.config["label_studio"]["server_url"],
            project_title=self.config["label_studio"]["project_title"],
            label_config=REWARD_BENCH_LABEL_CONFIG,
        )

    def run_pipeline_without_annotation(self) -> bool:
        """Run data pipeline without annotation stage"""
        logger.info("🚀 Running pipeline without annotation...")

        try:
            load_module = self.create_load_module(
                limit=self.config["data_source"]["limit"]
            )
            process_module = self.create_basic_process_module()

            build_module = create_build_module(
                name="simple_pipeline",
                config={},
                load_module=load_module,
                process_module=process_module,
            )

            result = build_module.run()

            logger.success("✅ Pipeline completed successfully!")
            logger.info(f"📊 Processed {len(result)} items")
            logger.info(f"📋 Dataset: {result.name}")

            return True

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False

    def run_pipeline_with_annotation(self, api_token: str = None) -> Optional[Any]:
        """Run complete pipeline with annotation stage"""
        logger.info("🚀 Running pipeline with annotation...")

        # Check service first
        if not self.check_label_studio_service():
            logger.error("❌ Label Studio service is not available")
            return None

        try:
            # Create modules
            load_module = self.create_load_module()
            process_module = self.create_annotation_process_module()
            annotation_module = self.create_annotation_module(api_token)

            # Create build pipeline
            build_module = create_build_module(
                name="annotation_pipeline",
                config={"description": "Complete pipeline with annotation"},
                load_module=load_module,
                process_module=process_module,
                annotation_module=annotation_module,
            )

            # Run pipeline
            result = build_module.run()

            self._print_annotation_results(result)
            return result

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            logger.info("💡 Make sure Label Studio is running and API token is correct")
            return None

    def export_annotations(
        self, project_id: int, api_token: str = None
    ) -> Optional[Any]:
        """Export annotations from a completed project"""
        logger.info(f"📤 Exporting annotations from project {project_id}...")

        try:
            annotation_module = self.create_annotation_module(api_token)
            annotation_module.project_id = project_id

            annotated_dataset = annotation_module.export_annotations_to_dataset(
                filename="exported_annotations.json", include_original_data=True
            )

            logger.success(f"✅ Exported annotations: {annotated_dataset.name}")
            logger.info(f"📊 Annotated samples: {len(annotated_dataset.datas)}")
            logger.info(f"📝 Metadata: {annotated_dataset.metadata}")

            return annotated_dataset

        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return None

    def check_label_studio_service(self) -> bool:
        """Check if Label Studio service is running and accessible"""
        try:
            response = requests.get(
                f"{self.config['label_studio']['server_url']}/health", timeout=5
            )
            if response.status_code == 200:
                logger.success("✅ Label Studio service is running")
                return True
            else:
                logger.error(
                    f"❌ Label Studio service returned status {response.status_code}"
                )
                return False
        except Exception as e:
            logger.error(f"❌ Label Studio service is not running: {e}")
            logger.info("💡 Start it with: label-studio start")
            return False

    @staticmethod
    def _print_annotation_results(result):
        """Print annotation pipeline results"""
        logger.success("🎉 Pipeline completed successfully!")
        logger.info(f"📊 Result: {result.name}")
        logger.info(f"📋 Data count: {len(result.datas)}")
        logger.info(f"📝 Metadata: {result.metadata}")

        project_id = result.metadata.get("annotation_project_id")
        server_url = result.metadata.get("annotation_server_url")

        if project_id:
            logger.info("\n🎯 Annotation project created!")
            logger.info(f"📋 Project ID: {project_id}")
            logger.info(f"🌐 Access at: {server_url}/projects/{project_id}")
            logger.info("\n📝 Next steps:")
            logger.info("   1. Go to the URL above")
            logger.info("   2. Annotate some tasks")
            logger.info(
                f"   3. Export annotations using: --export-only --project-id {project_id}"
            )


class DataExporter:
    """Utility class for data export operations"""

    @staticmethod
    def convert_to_jsonl_format(data):
        """Convert BaseData object to JSONL format"""
        return data.model_dump(mode="json")

    @staticmethod
    def export_dataset_to_jsonl(dataset, output_path: str):
        """Export dataset to JSONL file"""
        if not dataset or not dataset.datas:
            logger.warning("No data to export")
            return

        output_path = Path(output_path)
        written_count = 0

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for data in dataset.datas:
                    json_data = DataExporter.convert_to_jsonl_format(data)
                    f.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                    written_count += 1

            logger.success(
                f"✅ Successfully exported {written_count} items to {output_path}"
            )

        except Exception as e:
            logger.error(f"❌ Error exporting to JSONL: {str(e)}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RM Gallery Annotation Pipeline Examples"
    )
    parser.add_argument(
        "--with-annotation",
        action="store_true",
        help="Run pipeline with annotation stage",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export annotations from existing project",
    )
    parser.add_argument(
        "--check-service",
        action="store_true",
        help="Only check Label Studio service status",
    )
    parser.add_argument(
        "--project-id", type=int, help="Project ID for exporting annotations"
    )
    parser.add_argument("--api-token", type=str, help="Label Studio API token")
    parser.add_argument("--config", type=str, help="Path to custom configuration file")

    return parser.parse_args()


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults"""
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            custom_config = json.load(f)
        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        config.update(custom_config)
        return config
    return DEFAULT_CONFIG


def main():
    """Main execution function"""
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)
    pipeline = AnnotationPipelineExample(config)

    logger.info("🚀 RM Gallery Annotation Pipeline Examples")
    logger.info("=" * 60)

    # Handle different execution modes
    if args.check_service:
        pipeline.check_label_studio_service()
        return

    if args.export_only:
        if not args.project_id:
            logger.error("❌ --project-id is required for export-only mode")
            return
        pipeline.export_annotations(args.project_id, args.api_token)
        return

    if args.with_annotation:
        # Check service first
        if not pipeline.check_label_studio_service():
            logger.error("⚠️  Please start Label Studio service first!")
            logger.info("💡 Run: label-studio start")
            return

        result = pipeline.run_pipeline_with_annotation(args.api_token)
        if result:
            logger.info("\n🎯 Pipeline completed! You can now:")
            logger.info("   • Annotate tasks in the web interface")
            logger.info("   • Export results with --export-only flag")
    else:
        # Run basic pipeline without annotation
        pipeline.run_pipeline_without_annotation()


if __name__ == "__main__":
    main()
