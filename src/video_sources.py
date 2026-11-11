"""Safe, size-limited downloads of supported public video URLs."""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlsplit

MAX_VIDEO_BYTES = 150 * 1024 * 1024
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class VideoURLDownloadError(ValueError):
    """A public video URL could not be safely downloaded or decoded."""


def validate_public_video_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise VideoURLDownloadError("Enter a public HTTP or HTTPS video URL.")
    if parsed.username or parsed.password:
        raise VideoURLDownloadError("URLs containing usernames or passwords are not supported.")

    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise VideoURLDownloadError("Local and private-network URLs are not supported.")
    try:
        addresses = {
            ipaddress.ip_address(record[4][0].split("%")[0])
            for record in socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        }
    except (OSError, ValueError) as error:
        raise VideoURLDownloadError("The video host could not be resolved. Check that the public URL is available.") from error
    if not addresses or any(not address.is_global for address in addresses):
        raise VideoURLDownloadError("Local and private-network URLs are not supported.")
    return url.strip()


def download_public_video(url: str, destination: Path) -> Path:
    safe_url = validate_public_video_url(url)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        import yt_dlp

        options = {
            "format": "best[ext=mp4]/bestvideo[ext=mp4]/bestvideo[ext=webm]/best",
            "outtmpl": str(destination / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "max_filesize": MAX_VIDEO_BYTES,
            "socket_timeout": 20,
            "retries": 1,
        }
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(safe_url, download=True)
            if not info:
                raise VideoURLDownloadError("The video provider returned no downloadable video information.")
    except VideoURLDownloadError:
        raise
    except Exception as error:
        message = str(error).strip() or error.__class__.__name__
        raise VideoURLDownloadError(f"The video URL could not be downloaded: {message}") from error

    candidates = [
        path for path in destination.glob("source.*")
        if path.suffix.casefold() in SUPPORTED_VIDEO_SUFFIXES and path.is_file()
    ]
    if not candidates:
        raise VideoURLDownloadError("The URL did not provide a supported video file (MP4, AVI, MOV, MKV, or WebM).")
    video_path = candidates[0]
    if video_path.stat().st_size > MAX_VIDEO_BYTES:
        video_path.unlink(missing_ok=True)
        raise VideoURLDownloadError("The downloaded video exceeds the 150 MB local-demo limit.")
    return video_path