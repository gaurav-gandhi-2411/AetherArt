from aetherart.utils import safe_get

def test_safe_get():
    d = {"a": 1}
    assert safe_get(d, "a") == 1
    assert safe_get(d, "b", 2) == 2
