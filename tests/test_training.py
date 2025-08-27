# tests/test_training.py
from src.utils.hashing import text_hash

def test_text_hash_is_hex_and_stable():
    h1 = text_hash("Hello")
    h2 = text_hash("Hello")
    h3 = text_hash("hello")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64
    int(h1, 16)  # parses as hex