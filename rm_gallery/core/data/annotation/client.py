from typing import Any, Dict, List, Optional

import requests
from loguru import logger


class LabelStudioClient:
    """Label Studio API client"""

    def __init__(self, server_url: str, api_token: str):
        self.server_url = server_url
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/json",
        }

    def create_project(
        self, title: str, label_config: str, description: str = ""
    ) -> Optional[int]:
        """Create a new project in Label Studio"""
        try:
            project_data = {
                "title": title,
                "description": description,
                "label_config": label_config,
            }

            response = requests.post(
                f"{self.server_url}/api/projects/",
                json=project_data,
                headers=self.headers,
                timeout=30,
            )

            if response.status_code == 201:
                project = response.json()
                logger.info(f"Created project: {title} (ID: {project['id']})")
                return project["id"]
            else:
                logger.error(
                    f"Failed to create project: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return None

    def import_tasks(self, project_id: int, tasks: List[Dict[str, Any]]) -> bool:
        """Import tasks to a project"""
        try:
            response = requests.post(
                f"{self.server_url}/api/projects/{project_id}/import",
                json=tasks,
                headers=self.headers,
                timeout=60,
            )

            if response.status_code == 201:
                logger.info(
                    f"Successfully imported {len(tasks)} tasks to project {project_id}"
                )
                return True
            else:
                logger.error(
                    f"Failed to import tasks: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Error importing tasks: {e}")
            return False

    def export_annotations(
        self, project_id: int, export_type: str = "JSON"
    ) -> Optional[List[Dict[str, Any]]]:
        """Export annotations from a project"""
        try:
            response = requests.get(
                f"{self.server_url}/api/projects/{project_id}/export",
                params={"exportType": export_type},
                headers=self.headers,
                timeout=60,
            )

            if response.status_code == 200:
                annotations = response.json()
                logger.info(
                    f"Exported {len(annotations)} annotations from project {project_id}"
                )
                return annotations
            else:
                logger.error(f"Failed to export annotations: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error exporting annotations: {e}")
            return None

    def delete_project(self, project_id: int) -> bool:
        """Delete a project"""
        try:
            response = requests.delete(
                f"{self.server_url}/api/projects/{project_id}/",
                headers=self.headers,
                timeout=30,
            )

            if response.status_code == 204:
                logger.info(f"Successfully deleted project {project_id}")
                return True
            else:
                logger.error(f"Failed to delete project: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error deleting project: {e}")
            return False

    def get_projects(self) -> Optional[List[Dict[str, Any]]]:
        """Get all projects"""
        try:
            response = requests.get(
                f"{self.server_url}/api/projects/",
                headers=self.headers,
                timeout=30,
            )

            if response.status_code == 200:
                projects = response.json()
                logger.info(f"Retrieved {len(projects)} projects")
                return projects
            else:
                logger.error(f"Failed to get projects: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error getting projects: {e}")
            return None
