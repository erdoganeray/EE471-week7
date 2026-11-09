"""Turns an image into the JSON body that Cog's POST /predictions expects, for sending from Postman.

Cog only accepts JSON, and files go inside it as data URIs. Pasting megabytes of base64 into Postman
is impractical, so this writes the body to a file that Postman sends with Body -> binary.

Usage:  python make_request.py sneaker.png      -> writes sneaker.json next to the image
"""
import base64
import json
import mimetypes
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python make_request.py <image file>")

    image = Path(sys.argv[1])
    mime_type = mimetypes.guess_type(image.name)[0]
    if not mime_type or not mime_type.startswith("image/"):
        sys.exit(f"{image.name} does not look like an image file")

    data_uri = f"data:{mime_type};base64,{base64.b64encode(image.read_bytes()).decode()}"
    body = image.with_suffix(".json")
    body.write_text(json.dumps({"input": {"image": data_uri}}))
    print(f"Wrote {body} ({body.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
