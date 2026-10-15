"""Per-document text preprocessing."""

import re

_HEADER_RE = re.compile(
    r"(?m)^\s*(?:From|Subject|Lines|Organization|Distribution|"
    r"NNTP-Posting-Host|Path|Newsgroups|Date|Message-ID|References|"
    r"Reply-To|Sender|Keywords|Summary|Approved|Followup-To|Xref|"
    r"In-Reply-To|Content-Type|MIME-Version|Content-Transfer-Encoding)"
    r"\s*:.*$"
)


def preprocess_text(text):
    """Remove common news/email headers and normalize whitespace."""
    raw = str(text)
    cleaned = _HEADER_RE.sub(" ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else raw


def preprocess_many(texts):
    return [preprocess_text(text) for text in texts]

