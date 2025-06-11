#!/usr/bin/env python3
"""
Unified Label Studio Manager
manage label studio service with docker or pip
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from loguru import logger as loguru_logger


class LabelStudioManager:
    """Label Studio Manager"""

    def __init__(
        self,
        port: int = 8080,
        username: str = "admin@example.com",
        password: str = "admin123",
        data_dir: str = "./label_studio_data",
        use_docker: bool = True,
        container_name: str = "rm-gallery-label-studio",
        image: str = "heartexlabs/label-studio:latest",
    ):
        self.port = port
        self.username = username
        self.password = password
        self.data_dir = data_dir
        self.use_docker = use_docker
        self.container_name = container_name
        self.image = image
        self.server_url = f"http://localhost:{port}"
        self.process = None
        self.api_token = None
        self.config_file = Path("label_studio_config.json")

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging system"""
        # Create logs directory
        logs_dir = os.path.join(os.path.abspath(self.data_dir), "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Setup loguru logger
        loguru_logger.remove()  # Remove default handler

        # Add console handler
        loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
        )

        # Add file handler
        log_file = os.path.join(logs_dir, "label_studio_manager.log")
        loguru_logger.add(
            log_file,
            rotation="10 MB",
            retention="3 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="INFO",
        )

    def start_service(self) -> bool:
        """Start Label Studio service"""
        try:
            # Check if service is already running
            if self._is_service_running():
                loguru_logger.info(
                    f"Label Studio service is already running at {self.server_url}"
                )
                return self._get_api_token()

            loguru_logger.info(f"Starting Label Studio service on port {self.port}...")

            # Choose startup method
            if self.use_docker and self._check_docker_available():
                success = self._start_with_docker()
            else:
                if self.use_docker:
                    loguru_logger.warning(
                        "Docker not available, falling back to pip installation"
                    )
                success = self._start_with_pip()

            if success:
                loguru_logger.info("✅ Label Studio service started successfully!")
                loguru_logger.info(f"🌐 Web interface: {self.server_url}")

                # Get API token and save configuration
                if self._get_api_token():
                    self._save_config()
                    return True
                else:
                    loguru_logger.error("Failed to get API token")
                    return False
            else:
                loguru_logger.error("❌ Failed to start Label Studio service")
                return False

        except Exception as e:
            loguru_logger.error(f"Error starting Label Studio service: {e}")
            return False

    def _check_docker_available(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _start_with_docker(self) -> bool:
        """Start Label Studio with Docker"""
        loguru_logger.info("Starting Label Studio with Docker...")

        # Stop existing container
        if self._container_exists():
            self._stop_container()

        # Pull image
        loguru_logger.info(f"Pulling Docker image: {self.image}...")
        result = subprocess.run(["docker", "pull", self.image], check=False)
        if result.returncode != 0:
            loguru_logger.error(f"Failed to pull Docker image: {self.image}")
            return False

        # Create data volume
        volume_name = f"{self.container_name}-data"
        subprocess.run(["docker", "volume", "create", volume_name], check=False)

        # Start container
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "--restart",
            "unless-stopped",
            "-p",
            f"{self.port}:8080",
            "-v",
            f"{volume_name}:/label-studio/data",
            "-e",
            f"LABEL_STUDIO_USERNAME={self.username}",
            "-e",
            f"LABEL_STUDIO_PASSWORD={self.password}",
            "-e",
            "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true",
            self.image,
        ]

        loguru_logger.info(f"Starting container: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            loguru_logger.error(f"Failed to start container: {result.stderr}")
            return False

        # Wait for service to be ready
        return self._wait_for_service()

    def _start_with_pip(self) -> bool:
        """Start Label Studio with pip"""
        loguru_logger.info("Starting Label Studio with pip...")

        # Check and install label-studio
        try:
            subprocess.run(
                ["label-studio", "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            loguru_logger.info("Installing Label Studio...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "label-studio"], check=False
            )
            if result.returncode != 0:
                loguru_logger.error("Failed to install Label Studio")
                return False

        # Create data directory
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        # Start command
        cmd = [
            "label-studio",
            "start",
            "--port",
            str(self.port),
            "--data-dir",
            self.data_dir,
            "--username",
            self.username,
            "--password",
            self.password,
            "--user-token",  # Generate user token
        ]

        loguru_logger.info(f"Running command: {' '.join(cmd)}")

        # Start service
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for service to be ready
        return self._wait_for_service()

    def _is_service_running(self) -> bool:
        """Check if service is already running"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _wait_for_service(self, max_wait: int = 300) -> bool:
        """Wait for service to be ready"""
        start_time = time.time()

        loguru_logger.info("Waiting for Label Studio to initialize...")

        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    loguru_logger.info("Label Studio is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass

            # 检查Docker容器是否仍在运行
            if self.use_docker and self._check_docker_available():
                if not self._container_running():
                    loguru_logger.error("Container stopped unexpectedly")
                    self._show_container_logs()
                    return False

            time.sleep(5)

        loguru_logger.error("Label Studio failed to start within timeout")
        return False

    def _get_api_token(self) -> bool:
        """Get API token"""
        max_retries = 10
        retry_delay = 3

        loguru_logger.info("Getting API token...")

        for attempt in range(max_retries):
            try:
                # 尝试登录获取token
                login_data = {"email": self.username, "password": self.password}

                response = requests.post(
                    f"{self.server_url}/api/users/login", json=login_data, timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    self.api_token = data.get("token")
                    if self.api_token:
                        loguru_logger.info("✅ Successfully obtained API token!")
                        return True
                    else:
                        loguru_logger.warning(
                            "Login successful but no token in response"
                        )

                elif response.status_code == 401:
                    loguru_logger.warning(
                        f"Login failed (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                else:
                    loguru_logger.warning(
                        f"Unexpected response: {response.status_code} - {response.text}"
                    )
                    time.sleep(retry_delay)

            except Exception as e:
                loguru_logger.warning(
                    f"Error getting API token (attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(retry_delay)

        loguru_logger.error(f"❌ Failed to get API token after {max_retries} attempts")
        return False

    def _save_config(self):
        """Save configuration to file"""
        config = {
            "server_url": self.server_url,
            "api_token": self.api_token,
            "username": self.username,
            "data_dir": self.data_dir,
            "use_docker": self.use_docker,
            "container_name": self.container_name if self.use_docker else None,
            "status": "running",
        }

        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        loguru_logger.info(f"💾 Configuration saved to {self.config_file}")
        loguru_logger.info(f"🔑 API Token: {self.api_token}")

    def _container_exists(self) -> bool:
        """Check if container exists"""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    f"name={self.container_name}",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return self.container_name in result.stdout.strip().split("\n")
        except:
            return False

    def _container_running(self) -> bool:
        """Check if container is running"""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"name={self.container_name}",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return self.container_name in result.stdout.strip().split("\n")
        except:
            return False

    def _stop_container(self):
        """Stop and delete container"""
        if self._container_exists():
            loguru_logger.info(f"Stopping container: {self.container_name}...")
            subprocess.run(["docker", "stop", self.container_name], check=False)
            loguru_logger.info(f"Removing container: {self.container_name}...")
            subprocess.run(["docker", "rm", self.container_name], check=False)

    def _show_container_logs(self):
        """Show container logs"""
        try:
            result = subprocess.run(
                ["docker", "logs", self.container_name],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                loguru_logger.error(
                    f"Container logs:\n{result.stdout}\n{result.stderr}"
                )
        except:
            pass

    def stop_service(self):
        """Stop service"""
        if self.use_docker:
            self._stop_container()
        else:
            if self.process:
                self.process.terminate()
                self.process.wait()
                loguru_logger.info("Label Studio service stopped")

        # Clear configuration file
        if self.config_file.exists():
            self.config_file.unlink()
            loguru_logger.info("Configuration file removed")

    def get_status(self) -> dict:
        """Get service status"""
        config = self.load_config()
        if config:
            return {"running": self._is_service_running(), "config": config}
        else:
            return {"running": False, "config": None}

    def load_config(self) -> Optional[dict]:
        """Load configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    return json.load(f)
            except Exception as e:
                loguru_logger.error(f"Error loading config: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Label Studio Unified Manager")
    parser.add_argument(
        "action", choices=["start", "stop", "status"], help="Action to perform"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for Label Studio service"
    )
    parser.add_argument(
        "--username", default="admin@example.com", help="Admin username"
    )
    parser.add_argument("--password", default="admin123", help="Admin password")
    parser.add_argument(
        "--data-dir",
        default="./label_studio_data",
        help="Data directory for Label Studio",
    )
    parser.add_argument(
        "--use-pip", action="store_true", help="Use pip instead of Docker"
    )
    parser.add_argument(
        "--container-name",
        default="rm-gallery-label-studio",
        help="Docker container name",
    )
    parser.add_argument(
        "--image", default="heartexlabs/label-studio:latest", help="Docker image to use"
    )

    args = parser.parse_args()

    manager = LabelStudioManager(
        port=args.port,
        username=args.username,
        password=args.password,
        data_dir=args.data_dir,
        use_docker=not args.use_pip,
        container_name=args.container_name,
        image=args.image,
    )

    if args.action == "start":
        if manager.start_service():
            loguru_logger.info("🎉 Label Studio is ready for use!")
            print("\n" + "=" * 60)
            print("🚀 Label Studio Successfully Started!")
            print("=" * 60)
            print(f"🌐 Web Interface: {manager.server_url}")
            print(f"📧 Username: {manager.username}")
            print(f"🔐 Password: {manager.password}")
            print(f"🔑 API Token: {manager.api_token}")
            print(f"📁 Data Directory: {manager.data_dir}")
            print(f"🐳 Deployment: {'Docker' if manager.use_docker else 'Pip'}")
            print("=" * 60)
            print("💡 Use this API token in your annotation modules")
            print("=" * 60)
        else:
            loguru_logger.error("❌ Failed to start Label Studio")
            sys.exit(1)

    elif args.action == "stop":
        manager.stop_service()
        loguru_logger.info("Service stopped")

    elif args.action == "status":
        status = manager.get_status()
        if status["config"]:
            print("\n" + "=" * 40)
            print("📊 Label Studio Status")
            print("=" * 40)
            print(f"🌐 Server: {status['config']['server_url']}")
            print(f"🔑 API Token: {status['config']['api_token']}")
            print(f"📁 Data Dir: {status['config']['data_dir']}")
            print(
                f"🐳 Deployment: {'Docker' if status['config']['use_docker'] else 'Pip'}"
            )
            print(f"✅ Status: {'Running' if status['running'] else 'Stopped'}")
            print("=" * 40)
        else:
            print("❌ Label Studio is not running or not configured")


if __name__ == "__main__":
    main()
