from week23_project_optimize.core.hashing import stable_json_sha256


def test_stable_json_hash_ignores_key_order():
    assert stable_json_sha256({"a": 1, "b": 2}) == stable_json_sha256({"b": 2, "a": 1})
