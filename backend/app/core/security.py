from cryptography.fernet import Fernet

from app.utils.hashing import derive_fernet_key


def build_fernet(key: str) -> Fernet:
    if len(key) < 16:
        raise ValueError("Encryption key must be at least 16 characters.")
    derived = derive_fernet_key(key)
    return Fernet(derived)
