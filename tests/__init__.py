def have_dolfinx() -> bool:
    try:
        import dolfinx  # noqa: F401

        return True
    except Exception:
        return False


__all__ = ["have_dolfinx"]
