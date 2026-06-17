"""Draft export providers for generated outreach messages.

The providers in this module only serialize already-generated outreach content.
They intentionally do not classify contacts or generate narrative copy.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

DRAFT_COLUMNS = ["To", "Subject", "Body"]
DEFAULT_SENDER_NAME = "Helmut von Keyserling"
DEFAULT_SENDER_EMAIL = "helmut.vonkeyserling@metabolon.com"
SENDER_NOT_CONFIGURED_NOTE = ""
GRAPH_AUTHORITY = "https://login.microsoftonline.com/common"
GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
GRAPH_SCOPES = ["Mail.ReadWrite"]
TOKEN_CACHE_SERVICE = "andy-bot-microsoft-graph"
TOKEN_CACHE_USERNAME = "outlook-drafts"
TOKEN_CACHE_PATH = Path.home() / ".andy_bot" / "ms_graph_token_cache.json"


def is_outlook_graph_configured() -> bool:
    """Return whether Microsoft Graph draft creation has a client ID configured."""
    return bool(os.getenv("MS_GRAPH_CLIENT_ID", "").strip())


class OutlookGraphAuthRequired(RuntimeError):
    """Raised when the user must complete Microsoft device-code authentication."""

    def __init__(self, flow: dict[str, str]) -> None:
        self.flow = flow
        super().__init__(
            "Authenticate Outlook once, then click Create Outlook Drafts again. "
            f"Open {flow['verification_uri']} and enter code {flow['user_code']}."
        )


class DraftProvider(ABC):
    """Interface for exporting or creating email drafts."""

    @abstractmethod
    def export(self, drafts: pd.DataFrame) -> bytes:
        """Return a serialized draft artifact for the supplied draft rows."""


class CSVDraftProvider(DraftProvider):
    """Export draft rows to an Outlook-import-friendly CSV."""

    def export(self, drafts: pd.DataFrame) -> bytes:
        return drafts[DRAFT_COLUMNS].to_csv(index=False).encode("utf-8-sig")


class EMLDraftProvider(DraftProvider):
    """Export draft rows to a ZIP file containing one .eml file per contact."""

    def __init__(self, sender_email: str | None = None) -> None:
        self.sender_email = (
            os.getenv("METABOLON_SENDER_EMAIL", "")
            if sender_email is None
            else sender_email
        )

    def export(self, drafts: pd.DataFrame) -> bytes:
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as archive:
            draft_rows = drafts[DRAFT_COLUMNS].itertuples(index=False)
            for position, draft in enumerate(draft_rows, start=1):
                message = EmailMessage()
                message["To"] = str(draft.To)
                message["Subject"] = str(draft.Subject)
                body = str(draft.Body)
                if self.sender_email:
                    message["From"] = f"{DEFAULT_SENDER_NAME} <{DEFAULT_SENDER_EMAIL}>"
                message.set_content(body)
                archive.writestr(f"contact_{position:03d}.eml", message.as_bytes())
        return zip_buffer.getvalue()


class OutlookGraphDraftProvider(DraftProvider):
    """Create Outlook drafts with Microsoft Graph without sending them."""

    def __init__(self, client_id: str | None = None) -> None:
        self.client_id = client_id or os.getenv("MS_GRAPH_CLIENT_ID", "")
        if not self.client_id:
            raise ValueError(
                "Set MS_GRAPH_CLIENT_ID to an Azure public-client app ID with "
                "Mail.ReadWrite permission."
            )
        import msal

        self.msal = msal
        self.cache = msal.SerializableTokenCache()
        self._load_cache()
        self.app = msal.PublicClientApplication(
            self.client_id,
            authority=GRAPH_AUTHORITY,
            token_cache=self.cache,
        )

    def has_cached_account(self) -> bool:
        """Return whether an Outlook account is available in the token cache."""
        return bool(self.app.get_accounts())

    def begin_device_authentication(self) -> dict[str, str]:
        """Start Microsoft device-code authentication for delegated Mail.ReadWrite."""
        flow = self.app.initiate_device_flow(scopes=GRAPH_SCOPES)
        if "user_code" not in flow:
            raise RuntimeError("Could not start Microsoft Graph device authentication.")
        return flow

    def connect(self) -> bool:
        """Return True when Outlook is connected, otherwise raise with a device-code flow."""
        accounts = self.app.get_accounts()
        if accounts:
            result = self.app.acquire_token_silent(GRAPH_SCOPES, account=accounts[0])
            if result and "access_token" in result:
                self._save_cache()
                return True
        raise OutlookGraphAuthRequired(self.begin_device_authentication())

    def export(self, drafts: pd.DataFrame) -> bytes:
        created_count = self.create_drafts(drafts)
        return json.dumps({"created": created_count}).encode("utf-8")

    def create_drafts(self, drafts: pd.DataFrame) -> int:
        """Create one unsent Outlook draft for each supplied draft row."""
        access_token = self._get_access_token()
        created_count = 0
        for draft in drafts[DRAFT_COLUMNS].itertuples(index=False):
            import requests

            response = requests.post(
                f"{GRAPH_BASE_URL}/me/messages",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "subject": str(draft.Subject),
                    "body": {
                        "contentType": "Text",
                        "content": str(draft.Body),
                    },
                    "toRecipients": [
                        {"emailAddress": {"address": str(draft.To).strip()}}
                    ],
                },
                timeout=30,
            )
            if response.status_code != 201:
                raise RuntimeError(
                    f"Microsoft Graph draft creation failed ({response.status_code}): {response.text}"
                )
            created_count += 1
        return created_count

    def _get_access_token(self) -> str:
        accounts = self.app.get_accounts()
        result = None
        if accounts:
            result = self.app.acquire_token_silent(GRAPH_SCOPES, account=accounts[0])
        if not result:
            raise OutlookGraphAuthRequired(self.begin_device_authentication())
        if "access_token" not in result:
            raise RuntimeError(
                result.get(
                    "error_description",
                    "Could not authenticate with Microsoft Graph.",
                )
            )
        self._save_cache()
        return result["access_token"]

    def complete_device_authentication(self, flow: dict[str, str]) -> str:
        """Complete a previously initiated device-code flow and persist the access token."""
        result = self.app.acquire_token_by_device_flow(flow)
        if "access_token" not in result:
            raise RuntimeError(
                result.get(
                    "error_description",
                    "Could not authenticate with Microsoft Graph.",
                )
            )
        self._save_cache()
        return result["access_token"]

    def _load_cache(self) -> None:
        cache_blob = None
        try:
            import keyring

            cache_blob = keyring.get_password(TOKEN_CACHE_SERVICE, TOKEN_CACHE_USERNAME)
        except Exception:
            cache_blob = None
        if cache_blob is None and TOKEN_CACHE_PATH.exists():
            cache_blob = TOKEN_CACHE_PATH.read_text(encoding="utf-8")
        if cache_blob:
            self.cache.deserialize(cache_blob)

    def _save_cache(self) -> None:
        if not self.cache.has_state_changed:
            return
        cache_blob = self.cache.serialize()
        try:
            import keyring

            keyring.set_password(TOKEN_CACHE_SERVICE, TOKEN_CACHE_USERNAME, cache_blob)
        except Exception:
            TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            TOKEN_CACHE_PATH.write_text(cache_blob, encoding="utf-8")
            TOKEN_CACHE_PATH.chmod(0o600)


class GmailDraftProvider(DraftProvider):
    """Placeholder for future Gmail draft creation."""

    def export(self, drafts: pd.DataFrame) -> bytes:
        raise NotImplementedError("Gmail draft creation is not implemented yet.")
