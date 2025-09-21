def test_imports():
    import aetherart  # noqa: F401
    from aetherart import config, logger, utils, model  # noqa: F401
    assert config is not None
