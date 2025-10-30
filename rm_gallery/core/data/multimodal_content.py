"""
Multimodal content definitions for vision-language reward modeling.

This module provides data structures for handling multimodal content including
images and text, with support for API-friendly formats (URL and base64).
Designed for seamless integration with VLM APIs like Qwen VL, GPT-4V, etc.
"""

import base64
import hashlib
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Union

import requests
from PIL import Image
from pydantic import BaseModel, Field, field_validator, model_validator


class ImageContent(BaseModel):
    """
    Represents image content with API-friendly formats.

    Supports two primary formats optimized for API transmission:
    - URL: Most efficient, recommended for public/accessible images
    - base64: Suitable for small images or when URL is not available

    Attributes:
        type: Format type - "url" or "base64"
        data: Image data as URL string or base64-encoded string
        metadata: Optional metadata (size, format, dimensions, etc.)
        detail: Detail level for API calls ("auto", "low", "high")

    Examples:
        >>> # From URL
        >>> img = ImageContent(
        ...     type="url",
        ...     data="https://example.com/image.jpg"
        ... )

        >>> # From base64
        >>> img = ImageContent(
        ...     type="base64",
        ...     data="iVBORw0KGgoAAAANSUhEUgAA..."
        ... )

        >>> # Convert between formats
        >>> pil_image = img.to_pil()
        >>> base64_str = img.to_base64()
    """

    type: Literal["url", "base64"] = Field(
        description="Image format type: 'url' or 'base64'"
    )
    data: str = Field(description="Image data: URL string or base64-encoded string")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata (size, format, dimensions, etc.)"
    )
    detail: Literal["auto", "low", "high"] = Field(
        default="auto", description="Detail level for API image processing"
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate that type is either 'url' or 'base64'."""
        if v not in {"url", "base64"}:
            raise ValueError(f"type must be 'url' or 'base64', got '{v}'")
        return v

    @field_validator("data")
    @classmethod
    def validate_data(cls, v: str, info) -> str:
        """Validate data format based on type."""
        if not v:
            raise ValueError("data cannot be empty")

        # Get type from values if available
        type_value = info.data.get("type") if info.data else None

        if type_value == "url":
            # Basic URL validation
            if not (
                v.startswith("http://")
                or v.startswith("https://")
                or v.startswith("data:image")
            ):
                raise ValueError(
                    f"URL must start with http://, https://, or data:image, got: {v[:50]}..."
                )
        elif type_value == "base64":
            # Validate base64 format
            # Support both raw base64 and data URL format
            data_to_validate = v

            # If it's a data URL, extract the base64 part
            if v.startswith("data:"):
                try:
                    # Format: data:image/png;base64,iVBORw0KG...
                    if ";base64," in v:
                        data_to_validate = v.split(";base64,", 1)[1]
                    else:
                        raise ValueError("data URL must contain ';base64,' separator")
                except Exception as e:
                    raise ValueError(f"Invalid data URL format: {str(e)}")

            try:
                # Try to decode to verify it's valid base64
                base64.b64decode(data_to_validate, validate=True)
            except Exception as e:
                raise ValueError(f"Invalid base64 data: {str(e)}")

        return v

    def to_pil(self, timeout: int = 30) -> Image.Image:
        """
        Convert image to PIL Image object.

        Args:
            timeout: Timeout in seconds for URL requests

        Returns:
            PIL Image object

        Raises:
            ValueError: If image data is invalid
            requests.RequestException: If URL fetch fails
        """
        try:
            if self.type == "url":
                # Handle data URLs
                if self.data.startswith("data:image"):
                    # Extract base64 data from data URL
                    base64_data = self.data.split(",", 1)[1]
                    image_bytes = base64.b64decode(base64_data)
                    return Image.open(BytesIO(image_bytes))
                else:
                    # Fetch from URL
                    response = requests.get(self.data, timeout=timeout)
                    response.raise_for_status()
                    return Image.open(BytesIO(response.content))
            else:  # base64
                # Support both raw base64 and data URL format
                data_to_decode = self.data
                if self.data.startswith("data:"):
                    # Extract base64 part from data URL
                    data_to_decode = self.data.split(",", 1)[1]

                image_bytes = base64.b64decode(data_to_decode)
                return Image.open(BytesIO(image_bytes))
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch image from URL: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to convert image to PIL: {str(e)}")

    def to_base64(self, format: Optional[str] = None, quality: int = 85) -> str:
        """
        Convert image to base64 string.

        Args:
            format: Image format (JPEG, PNG, etc.). If None, preserves original format.
            quality: Quality for lossy formats (1-100)

        Returns:
            Base64-encoded string
        """
        if self.type == "base64":
            return self.data

        # Convert to PIL and encode
        pil_image = self.to_pil()
        buffered = BytesIO()

        # Preserve original format if not specified
        if format is None:
            format = pil_image.format or "PNG"

        # Convert RGBA to RGB for JPEG (JPEG doesn't support transparency)
        if format.upper() == "JPEG" and pil_image.mode in ("RGBA", "LA", "P"):
            pil_image = pil_image.convert("RGB")

        # Save with appropriate quality
        save_kwargs = {"format": format}
        if format.upper() in ("JPEG", "WEBP"):
            save_kwargs["quality"] = quality

        pil_image.save(buffered, **save_kwargs)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def to_url(self) -> str:
        """
        Get URL representation of the image.

        Returns:
            URL string (for url type) or data URL (for base64 type)
        """
        if self.type == "url":
            return self.data
        else:
            # Create data URL from base64
            mime_type = self._guess_mime_type()
            return f"data:{mime_type};base64,{self.data}"

    def _guess_mime_type(self) -> str:
        """Guess MIME type from image data."""
        try:
            image_bytes = base64.b64decode(self.data)
            # Check magic bytes
            if image_bytes.startswith(b"\xff\xd8\xff"):
                return "image/jpeg"
            elif image_bytes.startswith(b"\x89PNG"):
                return "image/png"
            elif image_bytes.startswith(b"GIF"):
                return "image/gif"
            elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]:
                return "image/webp"
            else:
                return "image/jpeg"  # Default
        except:
            return "image/jpeg"

    def extract_metadata(self, force: bool = False) -> Dict[str, Any]:
        """
        Extract and cache image metadata.

        Args:
            force: Force re-extraction even if metadata exists

        Returns:
            Dictionary with metadata (width, height, format, size, etc.)
        """
        if self.metadata and not force:
            return self.metadata

        try:
            pil_image = self.to_pil()

            # Calculate size
            if self.type == "base64":
                size_bytes = len(base64.b64decode(self.data))
            else:
                size_bytes = None  # Can't easily get without downloading

            self.metadata = {
                "width": pil_image.width,
                "height": pil_image.height,
                "format": pil_image.format or "UNKNOWN",
                "mode": pil_image.mode,
                "size_bytes": size_bytes,
                "aspect_ratio": round(pil_image.width / pil_image.height, 2)
                if pil_image.height > 0
                else None,
            }

            return self.metadata
        except Exception as e:
            # Return minimal metadata on error
            return {"error": str(e)}

    def to_api_format(
        self, api_type: Literal["openai", "qwen", "anthropic"] = "openai"
    ) -> Dict[str, Any]:
        """
        Convert to API-specific format following official API specifications.

        Args:
            api_type: Target API type ("openai", "qwen", "anthropic")

        Returns:
            Dictionary formatted for the specified API

        Examples:
            >>> img = ImageContent(type="url", data="https://example.com/img.jpg")
            >>> img.to_api_format("openai")
            {'type': 'image_url', 'image_url': {'url': 'https://...', 'detail': 'auto'}}
        """
        if api_type == "openai":
            # OpenAI Vision API format
            # URL images: use image_url field
            # Base64 images: use data URL format in image_url
            if self.type == "url":
                # Standard HTTP/HTTPS URL
                return {
                    "type": "image_url",
                    "image_url": {"url": self.data, "detail": self.detail},
                }
            else:
                # Base64: wrap in data URL
                mime_type = self._guess_mime_type()
                data_url = f"data:{mime_type};base64,{self.data}"
                return {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": self.detail},
                }
        elif api_type == "qwen":
            # Qwen DashScope format
            # Reference: https://help.aliyun.com/zh/dashscope/developer-reference/qwen-vl-api
            if self.type == "url":
                # URL format: use 'image' field
                return {"image": self.data}
            else:
                # Base64 format: DashScope requires 'image' field with data URL format
                # Note: Some versions may use 'image_base64' field, but data URL is more universal
                mime_type = self._guess_mime_type()
                data_url = f"data:{mime_type};base64,{self.data}"
                return {"image": data_url}
        elif api_type == "anthropic":
            # Claude format
            # Reference: https://docs.anthropic.com/claude/docs/vision
            if self.type == "url":
                return {"type": "image", "source": {"type": "url", "url": self.data}}
            else:
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self._guess_mime_type(),
                        "data": self.data,
                    },
                }
        else:
            raise ValueError(f"Unsupported API type: {api_type}")

    def get_cache_key(self) -> str:
        """
        Generate a unique cache key for this image.

        Returns:
            SHA256 hash of the image data
        """
        if self.type == "url":
            # Use URL as cache key
            return hashlib.sha256(self.data.encode()).hexdigest()
        else:
            # Use hash of base64 data
            return hashlib.sha256(self.data.encode()).hexdigest()

    @classmethod
    def from_pil(
        cls, image: Image.Image, format: Optional[str] = None, quality: int = 85
    ) -> "ImageContent":
        """
        Create ImageContent from PIL Image object.

        Args:
            image: PIL Image object
            format: Output format (JPEG, PNG, etc.). If None, uses PNG for images with
                   transparency, JPEG otherwise.
            quality: Quality for lossy formats (only applies to JPEG/WEBP)

        Returns:
            ImageContent instance with base64 data
        """
        buffered = BytesIO()

        # Auto-detect format if not specified
        if format is None:
            # Use PNG for images with transparency, JPEG otherwise
            if image.mode in ("RGBA", "LA", "P") or (
                hasattr(image, "info") and "transparency" in image.info
            ):
                format = "PNG"
            else:
                format = "JPEG"

        # Convert RGBA to RGB for JPEG (JPEG doesn't support transparency)
        save_image = image
        if format.upper() == "JPEG" and image.mode in ("RGBA", "LA", "P"):
            save_image = image.convert("RGB")

        # Save with appropriate quality
        save_kwargs = {"format": format}
        if format.upper() in ("JPEG", "WEBP"):
            save_kwargs["quality"] = quality

        save_image.save(buffered, **save_kwargs)
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return cls(
            type="base64",
            data=base64_str,
            metadata={
                "width": image.width,
                "height": image.height,
                "format": format,
                "mode": image.mode,
                "size_bytes": len(buffered.getvalue()),
            },
        )

    @classmethod
    def from_file(cls, file_path: str) -> "ImageContent":
        """
        Create ImageContent from local file.

        Args:
            file_path: Path to image file

        Returns:
            ImageContent instance with base64 data
        """
        image = Image.open(file_path)
        format = image.format or "JPEG"
        return cls.from_pil(image, format=format)

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "examples": [
                {
                    "type": "url",
                    "data": "https://example.com/image.jpg",
                    "detail": "auto",
                },
                {
                    "type": "base64",
                    "data": "iVBORw0KGgoAAAANSUhEUgAA...",
                    "detail": "high",
                },
            ]
        }


class MultimodalContent(BaseModel):
    """
    Represents multimodal content combining text and images.

    This is the primary container for multimodal data, supporting:
    - Pure text content
    - Pure image content
    - Mixed text and image content
    - Multiple images

    Attributes:
        text: Optional text content
        images: List of image content objects

    Examples:
        >>> # Text only
        >>> content = MultimodalContent(text="Describe this image")

        >>> # Text + single image
        >>> content = MultimodalContent(
        ...     text="What's in this image?",
        ...     images=[ImageContent(type="url", data="https://...")]
        ... )

        >>> # Multiple images
        >>> content = MultimodalContent(
        ...     text="Compare these images",
        ...     images=[img1, img2, img3]
        ... )
    """

    text: Optional[str] = Field(default=None, description="Text content")
    images: List[ImageContent] = Field(
        default_factory=list, description="List of image content objects"
    )

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: List[ImageContent]) -> List[ImageContent]:
        """Validate images list."""
        if not isinstance(v, list):
            raise ValueError("images must be a list")
        return v

    @model_validator(mode="after")
    def validate_content(self) -> "MultimodalContent":
        """
        Validate content (lenient mode for flexibility).

        Note: We allow empty content for cases where it will be filled later.
        This is more flexible for dynamic content construction.
        """
        # Relaxed validation: allow empty content for gradual construction
        # Users can check has_text() and has_image() before using
        return self

    def has_text(self) -> bool:
        """Check if content has text."""
        return self.text is not None and len(self.text.strip()) > 0

    def has_image(self) -> bool:
        """Check if content has images."""
        return len(self.images) > 0

    def is_multimodal(self) -> bool:
        """Check if content is truly multimodal (has both text and images)."""
        return self.has_text() and self.has_image()

    def get_text(self) -> str:
        """Get text content (empty string if None)."""
        return self.text or ""

    def get_images(self) -> List[ImageContent]:
        """Get list of images."""
        return self.images

    def add_image(self, image: ImageContent) -> None:
        """
        Add an image to the content.

        Args:
            image: ImageContent object to add
        """
        self.images.append(image)

    def add_text(self, text: str, append: bool = True) -> None:
        """
        Add or set text content.

        Args:
            text: Text to add
            append: If True, append to existing text; if False, replace
        """
        if append and self.text:
            self.text += " " + text
        else:
            self.text = text

    def to_api_format(
        self,
        api_type: Literal["openai", "qwen", "anthropic"] = "openai",
        force_structured: bool = False,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Convert to API-specific message content format.

        Args:
            api_type: Target API type
            force_structured: If True, always return structured format even for text-only

        Returns:
            Formatted content for the API
            - OpenAI: string for text-only, list for multimodal
            - Qwen: always list format (per DashScope spec)
            - Anthropic: always list format

        Examples:
            >>> content = MultimodalContent(
            ...     text="What's in this image?",
            ...     images=[ImageContent(type="url", data="https://...")]
            ... )
            >>> content.to_api_format("openai")
            [
                {'type': 'text', 'text': "What's in this image?"},
                {'type': 'image_url', 'image_url': {'url': 'https://...'}}
            ]
        """
        if api_type == "openai":
            # OpenAI: text-only can be string, multimodal must be array
            if not self.has_image() and not force_structured:
                return self.get_text()

            result = []
            if self.has_text():
                result.append({"type": "text", "text": self.text})
            for image in self.images:
                result.append(image.to_api_format("openai"))
            return result

        elif api_type == "qwen":
            # Qwen DashScope: always uses array format per official docs
            # Reference: https://help.aliyun.com/zh/dashscope/developer-reference/qwen-vl-api
            result = []
            if self.has_text():
                result.append({"text": self.text})
            for image in self.images:
                result.append(image.to_api_format("qwen"))
            return result

        elif api_type == "anthropic":
            # Claude: always uses array format
            result = []
            if self.has_text():
                result.append({"type": "text", "text": self.text})
            for image in self.images:
                result.append(image.to_api_format("anthropic"))
            return result
        else:
            raise ValueError(f"Unsupported API type: {api_type}")

    def to_string(self, include_image_info: bool = True) -> str:
        """
        Convert to human-readable string representation.

        Args:
            include_image_info: Whether to include image count info

        Returns:
            String representation
        """
        parts = []
        if self.has_text():
            parts.append(self.text)
        if self.has_image() and include_image_info:
            parts.append(f"[{len(self.images)} image(s)]")
        return " ".join(parts)

    def __str__(self) -> str:
        """String representation."""
        return self.to_string()

    def estimate_tokens(self, model: str = "gpt-4o") -> int:
        """
        Estimate token count for API cost calculation.

        Args:
            model: Model name for token estimation

        Returns:
            Estimated token count

        Note:
            This is a rough estimation. Actual token usage may vary.
        """
        token_count = 0

        # Text tokens (rough estimate: 1 token ≈ 4 chars)
        if self.has_text():
            token_count += len(self.text) // 4

        # Image tokens (varies by detail level and image size)
        for image in self.images:
            metadata = image.metadata or {}
            width = metadata.get("width", 512)
            height = metadata.get("height", 512)

            if image.detail == "low":
                token_count += 85  # Fixed low-detail tokens
            else:
                # High detail: depends on image size
                # Approximate: 170 tokens per 512x512 tile
                tiles_x = (width + 511) // 512
                tiles_y = (height + 511) // 512
                token_count += 85 + (tiles_x * tiles_y * 170)

        return token_count

    class Config:
        """Pydantic config."""

        # Allow extra fields for flexibility (e.g., metadata, captions, etc.)
        # This provides better extensibility while still being safe due to explicit field checking
        extra = "allow"
        json_schema_extra = {
            "examples": [
                {"text": "Describe this image", "images": []},
                {
                    "text": "What's in this image?",
                    "images": [
                        {"type": "url", "data": "https://example.com/image.jpg"}
                    ],
                },
            ]
        }
