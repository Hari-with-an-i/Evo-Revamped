import re

# --- Compiled patterns (module-level for reuse) ---

_MD_IMAGE = re.compile(r'!\[[^\]]*\]\([^)]*\)')
_MD_LINK = re.compile(r'\[([^\]]*)\]\([^)]*\)')
_MD_HEADER = re.compile(r'^#{1,6}\s+.*$', re.MULTILINE)
_MD_HR = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)
_MD_EMPHASIS = re.compile(r'\*{1,3}([^*\n]*)\*{1,3}')
_MD_EMPHASIS2 = re.compile(r'_{1,3}([^_\n]*)_{1,3}')
_BARE_URL = re.compile(r'https?://\S+')
_MD_LIST_MARKER = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)

_BOILERPLATE = re.compile(
    r'(?i)('
    r'skip to (?:main )?content'
    r'|cookie policy'
    r'|accept (?:all )?cookies?'
    r'|privacy policy'
    r'|terms of (?:service|use)'
    r'|sign (?:up|in)'
    r'|log(?:ged)? in'
    r'|subscribe(?: now)?'
    r'|already have an account'
    r'|create an account'
    r'|enable javascript'
    r'|javascript (?:is )?(?:required|disabled)'
    r'|this (?:site|page) uses cookies'
    r'|we use cookies'
    r'|read more'
    r'|share this'
    r'|follow us'
    r'|newsletter'
    r'|advertisement'
    r'|sponsored content'
    r'|©\s*\d{4}'
    r'|all rights reserved'
    r'|[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}'
    r')'
)


def strip_html(text: str) -> str:
    """Remove HTML tags, Markdown navigation/boilerplate, and collapse whitespace."""
    # 1. Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Remove Markdown images
    text = _MD_IMAGE.sub(' ', text)
    # 3. Convert Markdown links to text only
    text = _MD_LINK.sub(r'\1', text)
    # 4. Remove Markdown headers (navigation/section labels in scraped content)
    text = _MD_HEADER.sub(' ', text)
    # 5. Remove horizontal rules
    text = _MD_HR.sub(' ', text)
    # 6. Unwrap bold/italic emphasis
    text = _MD_EMPHASIS.sub(r'\1', text)
    text = _MD_EMPHASIS2.sub(r'\1', text)
    # 7. Remove bare URLs
    text = _BARE_URL.sub(' ', text)
    # 8. Remove list markers (keep text after marker)
    text = _MD_LIST_MARKER.sub('', text)
    # 9. Remove boilerplate phrases
    text = _BOILERPLATE.sub(' ', text)
    # 10. Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
