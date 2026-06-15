"""Draft export providers for generated outreach messages.

The providers in this module only serialize already-generated outreach content.
They intentionally do not classify contacts or generate narrative copy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from email.message import EmailMessage
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

DRAFT_COLUMNS = ["To", "Subject", "Body"]


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

    def export(self, drafts: pd.DataFrame) -> bytes:
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as archive:
            draft_rows = drafts[DRAFT_COLUMNS].itertuples(index=False)
            for position, draft in enumerate(draft_rows, start=1):
                message = EmailMessage()
                message["To"] = str(draft.To)
                message["Subject"] = str(draft.Subject)
                message.set_content(str(draft.Body))
                archive.writestr(f"contact_{position:03d}.eml", message.as_bytes())
        return zip_buffer.getvalue()


class OutlookGraphDraftProvider(DraftProvider):
    """Placeholder for future Microsoft Graph draft creation."""

    def export(self, drafts: pd.DataFrame) -> bytes:
        raise NotImplementedError(
            "Outlook Graph API draft creation is not implemented yet."
        )


class GmailDraftProvider(DraftProvider):
    """Placeholder for future Gmail draft creation."""

    def export(self, drafts: pd.DataFrame) -> bytes:
        raise NotImplementedError("Gmail draft creation is not implemented yet.")
