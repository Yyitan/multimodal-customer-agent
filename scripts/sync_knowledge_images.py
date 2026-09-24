import argparse
import json
import shutil
from pathlib import Path

from app.config import PROJECT_ROOT, get_settings
from app.knowledge_images import SUPPORTED_IMAGE_EXTENSIONS


def load_required_image_ids(normalized_dir: Path) -> set[str]:
    image_ids: set[str] = set()
    for jsonl_path in sorted(normalized_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                image_ids.update(record.get("image_ids", []))
    return image_ids


def sync_images(source_dir: Path, destination_dir: Path) -> tuple[int, list[str]]:
    required_ids = load_required_image_ids(PROJECT_ROOT / "data" / "normalized")
    source_files = {
        path.stem: path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    }
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing: list[str] = []
    for image_id in sorted(required_ids):
        source = source_files.get(image_id)
        if source is None:
            missing.append(image_id)
            continue
        target = destination_dir / source.name
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
            copied += 1
    return copied, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把原始知识库插图同步到 API 可读取目录。",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=get_settings().knowledge_image_dir,
    )
    args = parser.parse_args()

    source_dir = args.source.expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"原始插图目录不存在：{source_dir}")

    copied, missing = sync_images(
        source_dir=source_dir,
        destination_dir=args.destination.expanduser().resolve(),
    )
    print(f"已同步图片：{copied}")
    print(f"知识库缺失图片：{len(missing)}")
    if missing:
        print("前 20 个缺失 ID：", ", ".join(missing[:20]))


if __name__ == "__main__":
    main()
