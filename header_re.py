"""Canonical rating-descriptor header handling for the Arabic hotel sentiment data.

Single source of truth used by preprocessing, tests, and inference.
Rule (identical to audit r2): a header is a Booking.com rating descriptor at
position 0 of the review, immediately followed by a period. Strip happens on the
RAW text FIRST; normalization is applied AFTER stripping. Because the corpus also
contains a few headers already written in normalized orthography (e.g. 'رايع.'),
a second stripping pass on the normalized text with the DERIVED normalized forms
removes those; both passes loop until no header remains at position 0.
"""
import re

# Raw orthography exactly as it appears in balanced-reviews.txt (audit r2 pattern, byte-identical).
RAW_HEADER_RE = re.compile(r"^(استثنائي|رائع|جيد جداً|جيد جدا|جيد|مقبول|ضعيف جداً|ضعيف جدا|ضعيف|مخيب للأمل|ممتاز)\.\s*")
RAW_DESCRIPTOR_FORMS = ["استثنائي", "رائع", "جيد جداً", "جيد جدا", "جيد", "مقبول",
                        "ضعيف جداً", "ضعيف جدا", "ضعيف", "مخيب للأمل", "ممتاز"]

DIACRITICS = re.compile(r"[ً-ْٰ]")

def normalize(text):
    """v3/audit-identical normalization: diacritics, tatweel, alef, yaa, hamza folding."""
    t = DIACRITICS.sub("", str(text)); t = t.replace("ـ", "")
    t = re.sub("[أإآٱ]", "ا", t); t = re.sub("ى", "ي", t)
    t = re.sub("ؤ", "و", t); t = re.sub("ئ", "ي", t)
    return t.strip()

# Normalized descriptor forms are DERIVED from the raw forms (never handwritten),
# longest first so prefix forms can never shadow a longer form.
NORM_DESCRIPTOR_FORMS = sorted({normalize(f) for f in RAW_DESCRIPTOR_FORMS}, key=len, reverse=True)
NORM_HEADER_RE = re.compile(r"^(" + "|".join(NORM_DESCRIPTOR_FORMS) + r")\.\s*")

def _strip_loop(pattern, text, max_rounds=5):
    cur = str(text); n = 0
    while n < max_rounds:
        new = pattern.sub("", cur, count=1)
        if new == cur:
            break
        cur = new; n += 1
    return cur, n

def strip_header_raw(text):
    """Remove one raw-orthography header from RAW text. Returns (stripped_text, was_stripped)."""
    t = str(text)
    new = RAW_HEADER_RE.sub("", t, count=1)
    return new, (new != t)

def strip_header_raw_all(text, max_rounds=5):
    """Strip repeated raw-orthography headers (e.g. 'ممتاز. ممتاز. ...') until none remains.
    Returns (stripped_text, n_headers_removed)."""
    return _strip_loop(RAW_HEADER_RE, text, max_rounds)

def strip_header_norm_all(text, max_rounds=5):
    """Strip repeated normalized-orthography headers from ALREADY-normalized text.
    Returns (stripped_text, n_headers_removed)."""
    return _strip_loop(NORM_HEADER_RE, text, max_rounds)

def preprocess(text):
    """Full inference-time preprocessing: strip raw headers -> normalize -> strip normalized headers."""
    stripped, _ = strip_header_raw_all(text)
    normed = normalize(stripped)
    final, _ = strip_header_norm_all(normed)
    return final.strip()
