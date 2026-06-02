"""Regression tests for the Gen 1 internal-index -> species mapping.

These are pure-data tests (no emulator needed). Run with:
    python -m pytest test_species_mapping.py -q
or just:
    python test_species_mapping.py
"""

from pokemon_agent.memory.red import (
    INTERNAL_TO_DEX,
    SPECIES_NAMES,
    species_name_from_index,
    _dedupe_types,
)


def test_internal_to_dex_is_complete_bijection():
    # Gen 1 has exactly 151 real species; the internal-index table must
    # cover every dex number 1..151 exactly once.
    assert len(INTERNAL_TO_DEX) == 151
    covered = set(INTERNAL_TO_DEX.values())
    assert covered == set(range(1, 152)), (
        f"missing dex numbers: {sorted(set(range(1, 152)) - covered)}"
    )


def test_known_internal_indices_resolve():
    # Anchored against pokered's PokedexOrder (data/pokemon/dex_order.asm).
    cases = {
        1: "Rhydon",      # internal 0x01
        21: "Mew",        # internal 0x15
        112: "Weedle",    # internal 0x70
        133: "Magikarp",  # internal 0x85
        153: "Bulbasaur", # internal 0x99
        176: "Charmander",# internal 0xB0
        177: "Squirtle",  # internal 0xB1 — the original reported bug
    }
    for idx, name in cases.items():
        assert species_name_from_index(idx) == name, (
            f"internal index {idx} should be {name}"
        )


def test_unknown_index_falls_back_labelled():
    # MissingNo./unused slots resolve to a labelled placeholder, not a crash.
    assert species_name_from_index(255) == "???(255)"


def test_squirtle_dex_number():
    assert INTERNAL_TO_DEX[177] == 7
    assert SPECIES_NAMES[7] == "Squirtle"


def test_dedupe_types_collapses_monotype():
    assert _dedupe_types(["Water", "Water"]) == ["Water"]


def test_dedupe_types_keeps_dual_type_order():
    assert _dedupe_types(["Grass", "Poison"]) == ["Grass", "Poison"]


if __name__ == "__main__":
    test_internal_to_dex_is_complete_bijection()
    test_known_internal_indices_resolve()
    test_unknown_index_falls_back_labelled()
    test_squirtle_dex_number()
    test_dedupe_types_collapses_monotype()
    test_dedupe_types_keeps_dual_type_order()
    print("All species-mapping tests passed.")
