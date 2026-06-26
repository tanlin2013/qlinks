from dataclasses import dataclass

import numpy as np

from qlinks.basis import Basis, full_basis_from_layout
from qlinks.operators import ConstantDiagonalOperator
from qlinks.qec import (
    CageSectorCollection,
    CodeSpace,
    LocalErrorSet,
    diagnose_cage_collection_code_candidate,
    union_basis_from_sector_bases,
)
from qlinks.variables import LocalSpace, VariableLayout


@dataclass(frozen=True)
class _Record:
    support: np.ndarray
    local_state: np.ndarray
    signature: tuple[int, int]


@dataclass(frozen=True)
class _CageResult:
    records: list[_Record]


@dataclass(frozen=True)
class _BuildResult:
    basis: Basis


def _basis_from_configs(layout, configs):
    return Basis.from_states(layout, np.asarray(configs, dtype=np.int64), sort=True)


def _sector_inputs():
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    ambient_basis = full_basis_from_layout(layout, sort=True)
    sector_a_basis = _basis_from_configs(layout, [[0, 0], [0, 1]])
    sector_b_basis = _basis_from_configs(layout, [[1, 0], [1, 1]])

    idx_a_00 = sector_a_basis.require_index(np.asarray([0, 0], dtype=np.int64))
    idx_a_01 = sector_a_basis.require_index(np.asarray([0, 1], dtype=np.int64))
    idx_b_11 = sector_b_basis.require_index(np.asarray([1, 1], dtype=np.int64))

    result_a = _CageResult(
        records=[
            _Record(
                support=np.asarray([idx_a_00], dtype=np.int64),
                local_state=np.asarray([1.0], dtype=np.complex128),
                signature=(0, 4),
            ),
            _Record(
                support=np.asarray([idx_a_01], dtype=np.int64),
                local_state=np.asarray([1.0], dtype=np.complex128),
                signature=(0, 6),
            ),
        ]
    )
    result_b = _CageResult(
        records=[
            _Record(
                support=np.asarray([idx_b_11], dtype=np.int64),
                local_state=np.asarray([1.0], dtype=np.complex128),
                signature=(0, 4),
            ),
        ]
    )
    return layout, ambient_basis, sector_a_basis, sector_b_basis, result_a, result_b


def test_union_basis_from_sector_bases_contains_all_sector_states() -> None:
    _layout, _ambient, sector_a_basis, sector_b_basis, _result_a, _result_b = _sector_inputs()

    union = union_basis_from_sector_bases((sector_a_basis, sector_b_basis), sort=True)

    assert union.n_states == 4
    assert union.require_index(np.asarray([0, 0], dtype=np.int64)) == 0
    assert union.require_index(np.asarray([1, 1], dtype=np.int64)) == 3


def test_cage_sector_collection_filters_signature_and_embeds_records() -> None:
    _layout, ambient, sector_a_basis, sector_b_basis, result_a, result_b = _sector_inputs()

    collection = CageSectorCollection.from_sector_results(
        [
            ((0, 0), sector_a_basis, result_a, "w00"),
            ((2, 0), sector_b_basis, result_b, "w20"),
        ],
        signature=(0, 4),
        ambient_basis=ambient,
    )

    assert len(collection) == 2
    assert collection.common_signature == (0, 4)
    assert collection.counts_by_sector == {"(0, 0)": 1, "(2, 0)": 1}

    rows = collection.to_ambient_row_vectors()
    idx_00 = ambient.require_index(np.asarray([0, 0], dtype=np.int64))
    idx_11 = ambient.require_index(np.asarray([1, 1], dtype=np.int64))
    np.testing.assert_allclose(rows[:, [idx_00, idx_11]], np.eye(2))

    code = CodeSpace.from_cage_collection(collection)
    assert code.dimension == 2
    assert code.ambient_dimension == ambient.n_states
    assert code.labels == (((0, 0), (0, 4), 0), ((2, 0), (0, 4), 0))


def test_cage_sector_collection_accepts_build_results_and_builds_union_basis() -> None:
    _layout, _ambient, sector_a_basis, sector_b_basis, result_a, result_b = _sector_inputs()

    collection = CageSectorCollection.from_sector_results(
        [
            ((0, 0), _BuildResult(sector_a_basis), result_a),
            ((2, 0), _BuildResult(sector_b_basis), result_b),
        ],
        signature=(0, 4),
    )

    assert collection.ambient_basis.n_states == 4
    assert len(collection.signatures) == 1
    assert "Cage sector collection" in collection.format_summary()


def test_diagnose_cage_collection_code_candidate_uses_collection_code_space() -> None:
    layout, ambient, sector_a_basis, sector_b_basis, result_a, result_b = _sector_inputs()
    collection = CageSectorCollection.from_sector_results(
        [
            ((0, 0), sector_a_basis, result_a),
            ((2, 0), sector_b_basis, result_b),
        ],
        signature=(0, 4),
        ambient_basis=ambient,
    )
    errors = LocalErrorSet.from_operators(
        [ConstantDiagonalOperator(layout=layout, coefficient=1.0, name="I")],
        names=["I"],
    )

    report = diagnose_cage_collection_code_candidate(
        collection=collection,
        errors=errors,
        max_weight=1,
        include_error_algebra=True,
    )

    assert report.signature == (0, 4)
    assert report.record_count == 2
    assert report.code_dimension == 2
    assert report.qec_candidate
    assert report.error_algebra is not None
    assert report.metadata["source"] == "cage_sector_collection"


@dataclass(frozen=True)
class _SectorBuildResult:
    basis: Basis
    model: object


@dataclass(frozen=True)
class _FakeWindingModel:
    layout: object
    winding_x: int | None = None
    winding_y: int | None = None

    def allowed_sector_labels(self):
        return {"winding_x": (0, 2), "winding_y": (0,)}

    def build(self, **_kwargs):
        if self.winding_x is None and self.winding_y is None:
            return _SectorBuildResult(full_basis_from_layout(self.layout, sort=True), self)

        configs = {
            (0, 0): [[0, 0]],
            (2, 0): [[1, 0]],
        }[(self.winding_x, self.winding_y)]
        return _SectorBuildResult(_basis_from_configs(self.layout, configs), self)


def _fake_cage_result_factory(build_result, _config):
    _ = build_result
    return _CageResult(
        records=[
            _Record(
                support=np.asarray([0], dtype=np.int64),
                local_state=np.asarray([1.0], dtype=np.complex128),
                signature=(0, 4),
            )
        ]
    )


def test_cage_sector_collection_from_model_sectors_builds_and_searches_sectors() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    model = _FakeWindingModel(layout)

    collection = CageSectorCollection.from_model_sectors(
        model,
        [(0, 0), (2, 0)],
        signature=(0, 4),
        cage_result_factory=_fake_cage_result_factory,
        ambient_basis_mode="model",
        source_name_prefix="sector=",
    )

    assert len(collection) == 2
    assert collection.ambient_basis.n_states == 4
    assert collection.counts_by_sector == {"(0, 0)": 1, "(2, 0)": 1}
    assert collection.metadata["source"] == "model_sector_collection"
    assert collection.metadata["sector_fields"] == ("winding_x", "winding_y")

    rows = collection.to_ambient_row_vectors()
    idx_00 = collection.ambient_basis.require_index(np.asarray([0, 0], dtype=np.int64))
    idx_10 = collection.ambient_basis.require_index(np.asarray([1, 0], dtype=np.int64))
    np.testing.assert_allclose(rows[:, [idx_00, idx_10]], np.eye(2))


def test_cage_sector_collection_from_model_sectors_accepts_mapping_labels() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    model = _FakeWindingModel(layout)

    collection = CageSectorCollection.from_model_sectors(
        model,
        [
            {"winding_x": 0, "winding_y": 0},
            {"winding_x": 2, "winding_y": 0},
        ],
        cage_result_factory=_fake_cage_result_factory,
    )

    assert len(collection) == 2
    assert collection.ambient_basis.n_states == 2
