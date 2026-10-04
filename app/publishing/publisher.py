"""Explicit publishing integrations for YouTube and Instagram Reels."""
from __future__ import annotations

import os
import pathlib
import time
from urllib.parse import quote, urlparse

import requests


def publish_video(video_path: pathlib.Path, platforms: list[str]) -> dict[str, dict[str, object]]:
    """Publish *video_path* to requested platforms and return per-platform results."""
    results: dict[str, dict[str, object]] = {}
    for platform in platforms:
        try:
            if platform == "youtube":
                remote_id = _publish_youtube(video_path)
                message = f"YouTube video ID: {remote_id}"
            elif platform == "instagram":
                remote_id = _publish_instagram_reel(video_path)
                message = f"Instagram media ID: {remote_id}"
            else:
                raise ValueError(f"Unsupported platform: {platform}")
            results[platform] = {"success": True, "message": message}
        except Exception as exc:
            results[platform] = {"success": False, "message": str(exc)}
    return results


def _publish_youtube(video_path: pathlib.Path) -> str:
    client_secrets = pathlib.Path(os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", "credentials/youtube_client_secret.json"))
    token_path = pathlib.Path(os.getenv("YOUTUBE_TOKEN_FILE", "credentials/youtube_token.json"))
    if not client_secrets.is_file():
        raise RuntimeError(
            f"YouTube OAuth client file not found at {client_secrets}. "
            "Set YOUTUBE_CLIENT_SECRETS_FILE to your Google OAuth Desktop client JSON."
        )

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    credentials = None
    if token_path.is_file():
        credentials = Credentials.from_authorized_user_file(str(token_path), scopes)
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), scopes)
            credentials = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json(), encoding="utf-8")

    service = build("youtube", "v3", credentials=credentials)
    title = video_path.stem.replace("_", " ")[:100]
    request = service.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": "Gaming highlight edited with Gaming Video Agent.",
                "categoryId": "20",
            },
            "status": {
                "privacyStatus": os.getenv("YOUTUBE_PRIVACY_STATUS", "private"),
                "selfDeclaredMadeForKids": False,
            },
        },
        media_body=MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True),
    )
    response = request.execute()
    return response["id"]


def _publish_instagram_reel(video_path: pathlib.Path) -> str:
    access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
    instagram_user_id = os.getenv("INSTAGRAM_USER_ID")
    public_base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    missing = [
        name for name, value in (
            ("INSTAGRAM_ACCESS_TOKEN", access_token),
            ("INSTAGRAM_USER_ID", instagram_user_id),
            ("PUBLIC_BASE_URL", public_base_url),
        ) if not value
    ]
    if missing:
        raise RuntimeError(
            "Instagram Reels publishing requires " + ", ".join(missing) +
            ". PUBLIC_BASE_URL must expose this dashboard's /output files over HTTPS."
        )
    if urlparse(public_base_url).scheme != "https":
        raise RuntimeError("PUBLIC_BASE_URL must use HTTPS so Instagram can fetch the video.")

    version = os.getenv("INSTAGRAM_GRAPH_API_VERSION", "v23.0")
    base_url = f"https://graph.facebook.com/{version}"
    video_url = f"{public_base_url}/output/{quote(video_path.name)}"
    create_response = requests.post(
        f"{base_url}/{instagram_user_id}/media",
        data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": f"{video_path.stem.replace('_', ' ')} #gaming #reels",
            "access_token": access_token,
        },
        timeout=30,
    )
    create_response.raise_for_status()
    creation_id = create_response.json()["id"]

    status_url = f"{base_url}/{creation_id}"
    for _ in range(24):
        status_response = requests.get(
            status_url,
            params={"fields": "status_code", "access_token": access_token},
            timeout=30,
        )
        status_response.raise_for_status()
        status = status_response.json().get("status_code")
        if status == "FINISHED":
            break
        if status == "ERROR":
            raise RuntimeError("Instagram could not process the Reel video container.")
        time.sleep(5)
    else:
        raise RuntimeError("Instagram Reel processing did not finish within two minutes.")

    publish_response = requests.post(
        f"{base_url}/{instagram_user_id}/media_publish",
        data={"creation_id": creation_id, "access_token": access_token},
        timeout=30,
    )
    publish_response.raise_for_status()
    return publish_response.json()["id"]
