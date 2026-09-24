import base64
from io import BytesIO

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from PIL import Image

from app.input_security import (
    ImageValidationError,
    RequestBodyLimitMiddleware,
    validate_image_data_urls,
)


def make_image_data_url(
    image_format: str = "PNG",
    mime_type: str = "image/png",
    size: tuple[int, int] = (1, 1),
) -> tuple[str, int]:
    """生成真实的小图片，供安全校验测试使用。"""
    output = BytesIO()
    Image.new("RGB", size, color=(255, 255, 255)).save(
        output,
        format=image_format,
    )
    raw_bytes = output.getvalue()
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}", len(raw_bytes)


def raises_image_error(function) -> bool:
    try:
        function()
    except ImageValidationError:
        return True
    return False


def main() -> None:
    checks: dict[str, bool] = {}
    png_data_url, png_size = make_image_data_url()

    validated = validate_image_data_urls(
        [png_data_url],
        max_image_bytes=1024,
        max_total_bytes=2048,
        max_image_pixels=100,
    )
    checks["合法PNG通过严格校验"] = validated == [png_data_url]

    checks["非法Base64被拒绝"] = raises_image_error(
        lambda: validate_image_data_urls(
            ["data:image/png;base64,%%%"],
            max_image_bytes=1024,
            max_total_bytes=2048,
            max_image_pixels=100,
        )
    )

    mismatched_data_url = png_data_url.replace(
        "data:image/png",
        "data:image/jpeg",
        1,
    )
    checks["声明MIME与真实格式不一致时拒绝"] = raises_image_error(
        lambda: validate_image_data_urls(
            [mismatched_data_url],
            max_image_bytes=1024,
            max_total_bytes=2048,
            max_image_pixels=100,
        )
    )

    checks["单张图片超过字节限制时拒绝"] = raises_image_error(
        lambda: validate_image_data_urls(
            [png_data_url],
            max_image_bytes=png_size - 1,
            max_total_bytes=2048,
            max_image_pixels=100,
        )
    )

    checks["多张图片超过总字节限制时拒绝"] = raises_image_error(
        lambda: validate_image_data_urls(
            [png_data_url, png_data_url],
            max_image_bytes=1024,
            max_total_bytes=png_size * 2 - 1,
            max_image_pixels=100,
        )
    )

    checks["图片像素数量超过限制时拒绝"] = raises_image_error(
        lambda: validate_image_data_urls(
            [png_data_url],
            max_image_bytes=1024,
            max_total_bytes=2048,
            max_image_pixels=0,
        )
    )

    test_app = FastAPI()

    @test_app.post("/upload")
    async def upload(request: Request) -> dict[str, int]:
        body = await request.body()
        return {"size": len(body)}

    test_app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=32,
        paths={"/upload"},
    )

    with TestClient(test_app) as client:
        small_response = client.post(
            "/upload",
            content=b"small-body",
        )
        large_response = client.post(
            "/upload",
            content=b"x" * 33,
        )

    checks["限制内请求体正常通过"] = (
        small_response.status_code == 200
        and small_response.json()["size"] == len(b"small-body")
    )
    checks["超大请求体在解析前返回413"] = (
        large_response.status_code == 413
        and large_response.json()["detail"] == "请求体过大。"
    )

    print("====输入安全测试====")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n输入安全测试：全部通过")


if __name__ == "__main__":
    main()
