import base64
import binascii
import json
import re
import warnings
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError


DATA_URL_PATTERN = re.compile(
    r"^data:(image/(?:jpeg|jpg|png|webp));base64,"
    r"([A-Za-z0-9+/]*={0,2})$",
    re.IGNORECASE,
)

MIME_TO_FORMAT = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}

SESSION_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
)


class ImageValidationError(ValueError):
    """用户上传图片不符合安全约束。"""


def validate_image_data_urls(
    images: list[str],
    *,
    max_image_bytes: int,
    max_total_bytes: int,
    max_image_pixels: int,
) -> list[str]:
    """严格校验 Base64 Data URL、真实图片格式、大小和像素数。"""
    total_bytes = 0

    for index, image_data_url in enumerate(images, start=1):
        if not isinstance(image_data_url, str):
            raise ImageValidationError(
                f"第 {index} 张图片必须是字符串。"
            )

        match = DATA_URL_PATTERN.fullmatch(image_data_url)
        if match is None:
            raise ImageValidationError(
                "图片必须是合法的 JPEG、PNG 或 WEBP Base64 Data URL。"
            )

        mime_type = match.group(1).lower()
        encoded_payload = match.group(2)
        if not encoded_payload:
            raise ImageValidationError(
                f"第 {index} 张图片内容不能为空。"
            )

        estimated_bytes = len(encoded_payload) * 3 // 4
        if estimated_bytes > max_image_bytes:
            raise ImageValidationError(
                f"第 {index} 张图片超过单张大小限制。"
            )

        try:
            image_bytes = base64.b64decode(
                encoded_payload,
                validate=True,
            )
        except (binascii.Error, ValueError) as error:
            raise ImageValidationError(
                f"第 {index} 张图片不是合法 Base64。"
            ) from error

        if not image_bytes:
            raise ImageValidationError(
                f"第 {index} 张图片内容不能为空。"
            )
        if len(image_bytes) > max_image_bytes:
            raise ImageValidationError(
                f"第 {index} 张图片超过单张大小限制。"
            )

        total_bytes += len(image_bytes)
        if total_bytes > max_total_bytes:
            raise ImageValidationError("全部图片总大小超过限制。")

        _validate_decoded_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
            image_index=index,
            max_image_pixels=max_image_pixels,
        )

    return images


def _validate_decoded_image(
    *,
    image_bytes: bytes,
    mime_type: str,
    image_index: int,
    max_image_pixels: int,
) -> None:
    """使用 Pillow 验证图片结构，并核对声明格式与真实格式。"""
    expected_format = MIME_TO_FORMAT[mime_type]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )
            with Image.open(BytesIO(image_bytes)) as image:
                actual_format = image.format
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ImageValidationError(
                        f"第 {image_index} 张图片尺寸无效。"
                    )
                if width * height > max_image_pixels:
                    raise ImageValidationError(
                        f"第 {image_index} 张图片像素数量超过限制。"
                    )
                if actual_format != expected_format:
                    raise ImageValidationError(
                        f"第 {image_index} 张图片声明格式与真实格式不一致。"
                    )
                image.verify()

            # verify() 不解码全部像素，重新打开并 load() 检查截断文件。
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
    except ImageValidationError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise ImageValidationError(
            f"第 {image_index} 张图片已损坏或不是受支持的图片。"
        ) from error


class RequestBodyLimitMiddleware:
    """在进入 FastAPI 解析前限制指定接口的请求体大小。"""

    def __init__(
        self,
        app: Any,
        max_body_bytes: int,
        paths: set[str] | None = None,
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.paths = paths or {"/chat"}

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("path") not in self.paths
        ):
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope.get("headers", []))
        if (
            content_length is not None
            and content_length > self.max_body_bytes
        ):
            await self._send_too_large(send)
            return

        buffered_messages: list[dict] = []
        received_bytes = 0
        while True:
            message = await receive()
            buffered_messages.append(message)
            if message.get("type") != "http.request":
                break

            received_bytes += len(message.get("body", b""))
            if received_bytes > self.max_body_bytes:
                await self._send_too_large(send)
                return
            if not message.get("more_body", False):
                break

        message_index = 0

        async def replay_receive() -> dict:
            nonlocal message_index
            if message_index < len(buffered_messages):
                message = buffered_messages[message_index]
                message_index += 1
                return message
            return {
                "type": "http.request",
                "body": b"",
                "more_body": False,
            }

        await self.app(scope, replay_receive, send)

    async def _send_too_large(self, send: Any) -> None:
        body = json.dumps(
            {"detail": "请求体过大。"},
            ensure_ascii=False,
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )


def _content_length(headers: list[tuple[bytes, bytes]]) -> int | None:
    for name, value in headers:
        if name.lower() != b"content-length":
            continue
        try:
            return int(value)
        except ValueError:
            return None
    return None
