"""Tests for feature engineering and transformation logic with edge cases."""

import pytest

from prodml.features import (
    create_vectorizer,
    dict_to_feature_dict,
    encode_store_and_fwd_flag,
)


@pytest.mark.parametrize(
    "flag_input,expected_encoding",
    [
        ("Y", 1),
        ("y", 1),
        ("N", 0),
        ("n", 0),
        (1, 1),
        (0, 0),
        (True, 1),
        (False, 0),
        (None, 0),  # Edge case: None category
        ("", 0),  # Edge case: empty string
        ("UNKNOWN", 0),  # Edge case: unseen category
    ],
)
def test_encode_store_and_fwd_flag_edge_cases(flag_input, expected_encoding):
    """Test encoding of categorical store_and_fwd_flag including missing and unknown categories."""
    assert encode_store_and_fwd_flag(flag_input) == expected_encoding


@pytest.mark.parametrize(
    "edge_case_dict,expected_checks",
    [
        # Edge case 1: Missing category (key omitted entirely)
        (
            {"trip_distance": 2.5, "fare_amount": 10.0},
            {"store_and_fwd_flag_encoded": 0, "trip_distance": 2.5},
        ),
        # Edge case 2: Zero distance
        (
            {"trip_distance": 0.0, "fare_amount": 5.0, "store_and_fwd_flag": "N"},
            {"trip_distance": 0.0, "store_and_fwd_flag_encoded": 0},
        ),
        # Edge case 3: Unseen PU_DO pair via explicit key
        (
            {"trip_distance": 3.0, "PU_DO": "9999_8888"},
            {"trip_distance": 3.0, "PU_DO": "9999_8888"},
        ),
        # Edge case 4: Unseen PU_DO pair via location IDs
        (
            {"trip_distance": 1.5, "PULocationID": 9999, "DOLocationID": 8888},
            {"trip_distance": 1.5, "PU_DO": "9999_8888"},
        ),
        # Edge case 5: Extra unmodeled keys
        (
            {"trip_distance": 4.0, "unexpected_external_id": "abc-xyz"},
            {"trip_distance": 4.0},
        ),
    ],
)
def test_dict_to_feature_dict_edge_cases(edge_case_dict, expected_checks):
    """Test feature dictionary normalization across edge cases."""
    result = dict_to_feature_dict(edge_case_dict)
    for key, val in expected_checks.items():
        assert result[key] == val
    if "unexpected_external_id" in edge_case_dict:
        assert "unexpected_external_id" not in result


def test_prepare_features_vectorizer_integration():
    """Verify that DictVectorizer handles unseen categories and prepared features gracefully."""
    training_records = [
        {"trip_distance": 1.0, "fare_amount": 5.0, "PU_DO": "100_101"},
        {"trip_distance": 2.0, "fare_amount": 10.0, "PU_DO": "100_102"},
    ]
    dv = create_vectorizer()
    X_train = dv.fit_transform(training_records)
    assert X_train.shape[0] == 2

    # Query with completely unseen PU_DO pair
    unseen_records = [{"trip_distance": 1.5, "fare_amount": 7.5, "PU_DO": "999_999"}]
    X_test = dv.transform(unseen_records)
    # The unseen PU_DO category is ignored without error
    assert X_test.shape[0] == 1
    assert X_test.shape[1] == X_train.shape[1]
