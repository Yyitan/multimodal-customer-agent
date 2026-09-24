import re
from pathlib import Path


IMAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


class KnowledgeImageCatalog:
    """Resolve knowledge image IDs without exposing arbitrary filesystem paths."""

    def __init__(self, image_dir: str | Path) -> None:
        self.image_dir = Path(image_dir).resolve()

    def resolve(self, image_id: str) -> Path | None:
        if not IMAGE_ID_PATTERN.fullmatch(image_id):
            return None

        for extension in SUPPORTED_IMAGE_EXTENSIONS:
            candidate = (self.image_dir / f"{image_id}{extension}").resolve()
            if self.image_dir not in candidate.parents:
                return None
            if candidate.is_file():
                return candidate
        return None

    def available(self, image_id: str) -> bool:
        return self.resolve(image_id) is not None
