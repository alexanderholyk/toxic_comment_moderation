import hashlib

def text_hash(s: str) -> str:
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()