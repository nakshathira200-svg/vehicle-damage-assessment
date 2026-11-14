import pytest

from src.rag.chunker import (
    classify_policy_scope,
    pack_sections,
)
from src.rag.retriever import query_policy_scope


@pytest.mark.parametrize(
    "text, expected_scope",
    [
        ("IMT15 Personal Accident Cover", "personal_accident"),
        ("IMT16 Personal Accident to Passengers", "personal_accident"),
        ("IMT19 Cover for Imported Vehicles", "own_damage"),
        ("IMT22 Compulsory Deductible", "own_damage"),
        ("IMT24 Electrical Electronic Fittings", "add_on"),
    ],
)
def test_endorsement_scope_classification(text, expected_scope):
    scope, _ = classify_policy_scope(text)
    assert scope == expected_scope


def test_general_exceptions_reset_personal_accident_scope():
    scope, title = classify_policy_scope(
        "GENERAL EXCEPTIONS: The Company shall not be liable.",
        "personal_accident",
        "Personal Accident",
    )
    assert scope == "general"
    assert title == "General Exceptions"


def test_general_exceptions_are_separate_packed_chunk():
    sections = [
        {
            "heading": "PERSONAL ACCIDENT COVER",
            "text": "The insurer pays compensation for bodily injury.",
        },
        {
            "heading": "GENERAL EXCEPTIONS",
            "text": "The Company shall not be liable for specified claims.",
        },
        {
            "heading": "1. Driver clause",
            "text": "The vehicle must be driven by an authorised driver.",
        },
    ]

    chunks = pack_sections(sections)

    assert len(chunks) >= 3
    assert "GENERAL EXCEPTIONS" not in chunks[0]
    assert chunks[1].startswith("GENERAL EXCEPTIONS")
    assert "Driver clause" in chunks[2]


@pytest.mark.parametrize(
    "query, expected_scope",
    [
        (
            "Vehicle damage while driving under the influence of alcohol",
            "own_damage",
        ),
        ("Personal accident compensation for owner driver", "personal_accident"),
        ("What does engine protect flood cover?", "add_on"),
        ("Liability to third parties", "third_party"),
    ],
)
def test_query_scope_classification(query, expected_scope):
    assert query_policy_scope(query) == expected_scope
