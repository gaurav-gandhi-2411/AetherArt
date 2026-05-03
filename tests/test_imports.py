def test_imports():
    import aetherart  # noqa: F401
    from aetherart import config, logger, model, utils  # noqa: F401

    assert config is not None
