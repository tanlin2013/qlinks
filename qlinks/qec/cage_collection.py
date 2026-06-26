from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from qlinks.basis import Basis
from qlinks.encoded import BinaryEncodedBasis, decode_binary_code, encode_binary_config

BasisLike = Basis | BinaryEncodedBasis


@dataclass(frozen=True, slots=True)
class CageSectorSource:
    """One sector-resolved cage-search output used by :class:`CageSectorCollection`.

    The ``basis`` is the sector-restricted basis in which the corresponding
    ``cage_result`` records are expressed.  The sector label is intentionally
    duck-typed so callers may use ``(winding_x, winding_y)``, a dictionary-like
    label, or any other stable object that is convenient in notebooks.
    """

    sector_label: Any
    basis: BasisLike
    cage_result: Any
    source_name: str | None = None

    @classmethod
    def from_item(cls, item: CageSectorSource | Sequence[Any]) -> CageSectorSource:
        """Normalize a source item.

        Accepted tuple/list forms are ``(sector_label, basis_or_build_result,
        cage_result)`` and ``(sector_label, basis_or_build_result, cage_result,
        source_name)``.  ``basis_or_build_result`` may be a basis directly or an
        object exposing a ``basis`` attribute, such as ``ModelBuildResult``.
        """
        if isinstance(item, CageSectorSource):
            return item

        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            raise TypeError(
                "sector sources must be CageSectorSource objects or tuples "
                "(sector_label, basis_or_build_result, cage_result[, source_name])."
            )

        if len(item) not in {3, 4}:
            raise ValueError(
                "sector source tuples must have length 3 or 4: "
                "(sector_label, basis_or_build_result, cage_result[, source_name])."
            )

        sector_label = item[0]
        basis = _extract_basis(item[1])
        cage_result = item[2]
        source_name = None if len(item) == 3 else None if item[3] is None else str(item[3])
        return cls(
            sector_label=sector_label,
            basis=basis,
            cage_result=cage_result,
            source_name=source_name,
        )


@dataclass(frozen=True, slots=True)
class CollectedCageRecord:
    """One cage record together with its source sector and source basis."""

    sector_label: Any
    signature: tuple[int, int] | None
    record_index: int
    record: Any = field(repr=False)
    basis: BasisLike = field(repr=False)
    source_name: str | None = None

    @property
    def label(self) -> tuple[Any, tuple[int, int] | None, int]:
        """Default label used when constructing a :class:`CodeSpace`."""
        return (self.sector_label, self.signature, self.record_index)

    def source_label(self) -> str:
        """Human-readable source label for summaries and notebooks."""
        prefix = "" if self.source_name is None else f"{self.source_name}:"
        return (
            f"{prefix}sector={self.sector_label!r}, "
            f"signature={self.signature}, rec={self.record_index}"
        )

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact summary of this collected record."""
        return {
            "sector_label": repr(self.sector_label),
            "signature": self.signature,
            "record_index": self.record_index,
            "source_name": self.source_name,
            "source_basis_dimension": self.basis.n_states,
        }


@dataclass(frozen=True, slots=True)
class CageSectorCollection:
    """Cage records collected across diagonal/topological sectors.

    Each collected record may come from a different sector-restricted basis.  The
    collection embeds them into ``ambient_basis`` when it builds row vectors.  If
    no ambient basis is provided, :meth:`from_sector_results` uses the union of
    all source-sector basis states.  For QEC leakage diagnostics, passing a
    larger common basis, such as the full constrained basis before winding-sector
    filtering, is usually preferable.
    """

    entries: tuple[CollectedCageRecord, ...]
    ambient_basis: BasisLike
    signature_filter: tuple[tuple[int, int], ...] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_sector_results(
        cls,
        sector_results: Sequence[CageSectorSource | Sequence[Any]],
        *,
        signature: tuple[int, int] | None = None,
        signatures: Sequence[tuple[int, int]] | None = None,
        ambient_basis: BasisLike | Any | None = None,
        sort_ambient_basis: bool = True,
        metadata: Mapping[str, object] | None = None,
    ) -> CageSectorCollection:
        """Collect cage records from several sector-resolved cage-search results.

        Args:
            sector_results: Sequence of ``CageSectorSource`` objects or tuples
                ``(sector_label, basis_or_build_result, cage_result[, source_name])``.
            signature: Optional single cage signature, such as ``(0, 4)``, to keep.
            signatures: Optional collection of signatures to keep.  Mutually
                exclusive with ``signature``.
            ambient_basis: Optional common basis for embedding all sector records.
                May be a basis directly or an object exposing ``basis``.
            sort_ambient_basis: Whether to sort the automatically built union basis.
            metadata: Optional user metadata attached to the collection.

        Returns:
            CageSectorCollection with provenance-preserving entries.
        """
        if signature is not None and signatures is not None:
            raise ValueError("Pass either signature or signatures, not both.")

        sources = tuple(CageSectorSource.from_item(item) for item in sector_results)
        if len(sources) == 0:
            raise ValueError("At least one sector result is required.")

        _require_compatible_layouts(source.basis for source in sources)

        signature_filter = _normalize_signature_filter(signature, signatures)
        entries: list[CollectedCageRecord] = []
        for source in sources:
            for record_index, record in enumerate(_iter_records(source.cage_result)):
                record_signature = _record_signature(record)
                if signature_filter is not None and record_signature not in signature_filter:
                    continue
                entries.append(
                    CollectedCageRecord(
                        sector_label=source.sector_label,
                        signature=record_signature,
                        record_index=record_index,
                        record=record,
                        basis=source.basis,
                        source_name=source.source_name,
                    )
                )

        if ambient_basis is None:
            common_basis = union_basis_from_sector_bases(
                tuple(source.basis for source in sources),
                sort=sort_ambient_basis,
            )
        else:
            common_basis = _extract_basis(ambient_basis)
            _require_compatible_layouts((*(source.basis for source in sources), common_basis))

        return cls(
            entries=tuple(entries),
            ambient_basis=common_basis,
            signature_filter=signature_filter,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_mapping(
        cls,
        sector_results: Mapping[Any, tuple[Any, Any] | Any],
        **kwargs: Any,
    ) -> CageSectorCollection:
        """Collect from a mapping ``sector_label -> (basis_or_build_result, cage_result)``.

        Values may also be objects with ``basis`` and ``cage_result`` attributes.
        """
        items: list[tuple[Any, Any, Any]] = []
        for sector_label, value in sector_results.items():
            if isinstance(value, tuple) and len(value) == 2:
                items.append((sector_label, value[0], value[1]))
                continue
            basis = _extract_basis(value)
            cage_result = getattr(value, "cage_result")
            items.append((sector_label, basis, cage_result))
        return cls.from_sector_results(items, **kwargs)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(
        self,
        index: int | slice,
    ) -> CollectedCageRecord | tuple[CollectedCageRecord, ...]:
        if isinstance(index, slice):
            return self.entries[index]
        return self.entries[int(index)]

    @property
    def labels(self) -> tuple[tuple[Any, tuple[int, int] | None, int], ...]:
        """Default labels for ``CodeSpace.from_cage_collection``."""
        return tuple(entry.label for entry in self.entries)

    @property
    def sector_labels(self) -> tuple[Any, ...]:
        """Distinct sector labels in first-appearance order."""
        labels: list[Any] = []
        for entry in self.entries:
            if entry.sector_label not in labels:
                labels.append(entry.sector_label)
        return tuple(labels)

    @property
    def signatures(self) -> tuple[tuple[int, int], ...]:
        """Distinct non-null cage signatures in sorted order."""
        signatures = {entry.signature for entry in self.entries if entry.signature is not None}
        return tuple(sorted(signatures))

    @property
    def counts_by_sector(self) -> dict[str, int]:
        """Number of collected records per sector, keyed by ``repr(sector_label)``."""
        return dict(Counter(repr(entry.sector_label) for entry in self.entries))

    @property
    def counts_by_signature(self) -> dict[tuple[int, int] | None, int]:
        """Number of collected records per cage signature."""
        return dict(Counter(entry.signature for entry in self.entries))

    @property
    def common_signature(self) -> tuple[int, int] | None:
        """Return the unique signature if all entries share one, otherwise ``None``."""
        signatures = {entry.signature for entry in self.entries}
        if len(signatures) == 1:
            return next(iter(signatures))
        return None

    def by_signature(self, signature: tuple[int, int]) -> CageSectorCollection:
        """Return a collection containing only records with the requested signature."""
        signature_normalized = _normalize_signature(signature)
        return self.select(signatures=(signature_normalized,))

    def by_sector(self, sector_label: Any) -> CageSectorCollection:
        """Return a collection containing only records from one sector label."""
        return self.select(sector_labels=(sector_label,))

    def select(
        self,
        *,
        signatures: Sequence[tuple[int, int]] | None = None,
        sector_labels: Sequence[Any] | None = None,
    ) -> CageSectorCollection:
        """Return a provenance-preserving filtered collection."""
        signature_filter = None
        if signatures is not None:
            signature_filter = tuple(_normalize_signature(sig) for sig in signatures)
            signature_set = set(signature_filter)
        else:
            signature_set = None

        sector_filter = None if sector_labels is None else tuple(sector_labels)

        entries = []
        for entry in self.entries:
            if signature_set is not None and entry.signature not in signature_set:
                continue
            if sector_filter is not None and not any(
                entry.sector_label == label for label in sector_filter
            ):
                continue
            entries.append(entry)

        return CageSectorCollection(
            entries=tuple(entries),
            ambient_basis=self.ambient_basis,
            signature_filter=signature_filter,
            metadata=dict(self.metadata),
        )

    def to_ambient_row_vectors(self) -> npt.NDArray[np.complex128]:
        """Embed collected records into ``ambient_basis`` as row vectors."""
        rows = np.zeros((len(self.entries), self.ambient_basis.n_states), dtype=np.complex128)
        for row_index, entry in enumerate(self.entries):
            rows[row_index, :] = embed_record_in_basis(
                record=entry.record,
                source_basis=entry.basis,
                target_basis=self.ambient_basis,
            )
        return rows

    def to_summary_dict(self, *, max_entries: int = 10) -> dict[str, object]:
        """Return a compact summary of this cross-sector collection."""
        preview = tuple(entry.to_summary_dict() for entry in self.entries[:max_entries])
        return {
            "n_records": len(self.entries),
            "ambient_dimension": self.ambient_basis.n_states,
            "n_sectors": len(self.sector_labels),
            "sector_labels": tuple(repr(label) for label in self.sector_labels),
            "signatures": self.signatures,
            "common_signature": self.common_signature,
            "signature_filter": self.signature_filter,
            "counts_by_sector": self.counts_by_sector,
            "counts_by_signature": self.counts_by_signature,
            "metadata": dict(self.metadata),
            "preview_records": preview,
            "n_preview_records": len(preview),
        }

    def to_text(self, *, max_entries: int = 10) -> str:
        """Return a human-readable cross-sector collection summary."""
        from qlinks.qec.reporting import format_key_value_lines

        summary = self.to_summary_dict(max_entries=max_entries)
        lines = [
            format_key_value_lines(
                "Cage sector collection",
                (
                    ("records", summary["n_records"]),
                    ("ambient dimension", summary["ambient_dimension"]),
                    ("sectors", summary["sector_labels"]),
                    ("signatures", summary["signatures"]),
                    ("common signature", summary["common_signature"]),
                    ("counts by sector", summary["counts_by_sector"]),
                    ("counts by signature", summary["counts_by_signature"]),
                ),
            )
        ]
        preview = self.entries[:max_entries]
        if preview:
            lines.append("preview records")
            lines.extend(f"  - {entry.source_label()}" for entry in preview)
            if len(self.entries) > len(preview):
                lines.append(f"  ... {len(self.entries) - len(preview)} more records")
        return "\n".join(lines)

    def format_summary(self, *, max_entries: int = 10) -> str:
        return self.to_text(max_entries=max_entries)

    def __str__(self) -> str:
        return self.to_text(max_entries=5)

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_entries: int = 10):
        """Return a rich renderable collection summary."""
        from rich.console import Group

        from qlinks.qec.reporting import add_summary_rows, require_rich

        _group, Panel, Table, _text = require_rich("CageSectorCollection")
        summary = self.to_summary_dict(max_entries=max_entries)
        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        add_summary_rows(
            overview,
            (
                ("records", summary["n_records"]),
                ("ambient dimension", summary["ambient_dimension"]),
                ("sectors", summary["sector_labels"]),
                ("signatures", summary["signatures"]),
                ("common signature", summary["common_signature"]),
            ),
        )

        table = Table(title="Preview records")
        table.add_column("sector")
        table.add_column("signature")
        table.add_column("record", justify="right")
        table.add_column("source")
        for entry in self.entries[:max_entries]:
            table.add_row(
                repr(entry.sector_label),
                str(entry.signature),
                str(entry.record_index),
                "" if entry.source_name is None else entry.source_name,
            )
        if len(self.entries) > max_entries:
            table.caption = f"Showing {max_entries} of {len(self.entries)} records"

        return Panel(Group(overview, table), title="Cage sector collection")


def union_basis_from_sector_bases(
    bases: Sequence[BasisLike],
    *,
    sort: bool = True,
) -> BasisLike:
    """Build a union basis containing all states from sector-restricted bases."""
    if len(bases) == 0:
        raise ValueError("At least one basis is required.")

    _require_compatible_layouts(bases)
    layout = bases[0].layout

    if all(isinstance(basis, BinaryEncodedBasis) for basis in bases):
        codes: set[int] = set()
        for basis in bases:
            assert isinstance(basis, BinaryEncodedBasis)
            codes.update(int(code) for code in basis.codes.tolist())
        return BinaryEncodedBasis.from_codes(layout, codes, sort=sort)

    configs: dict[bytes, npt.NDArray[np.int64]] = {}
    encoder = Basis.empty(layout).encoder
    for basis in bases:
        for index in range(basis.n_states):
            config = _basis_config(basis, index)
            configs[encoder.encode(config)] = config

    if configs:
        states = np.vstack(list(configs.values()))
    else:
        states = np.empty((0, layout.n_variables), dtype=np.int64)
    return Basis.from_states(layout, states, sort=sort)


def embed_record_in_basis(
    *,
    record: Any,
    source_basis: BasisLike,
    target_basis: BasisLike,
) -> npt.NDArray[np.complex128]:
    """Embed one cage-record-like object from ``source_basis`` to ``target_basis``."""
    source_vector = _record_vector(record, source_basis)
    target_vector = np.zeros(target_basis.n_states, dtype=np.complex128)
    nonzero_indices = np.flatnonzero(np.abs(source_vector) > 0)

    for source_index in nonzero_indices:
        target_index = _map_basis_index(source_basis, target_basis, int(source_index))
        target_vector[target_index] += source_vector[source_index]

    return target_vector


def _record_vector(record: Any, basis: BasisLike) -> npt.NDArray[np.complex128]:
    full_state = getattr(record, "full_state", None)
    if full_state is not None:
        vector = np.asarray(full_state, dtype=np.complex128)
        if vector.shape != (basis.n_states,):
            raise ValueError(
                f"Record full_state has shape {vector.shape}, expected ({basis.n_states},)."
            )
        return vector

    support = np.asarray(getattr(record, "support"), dtype=np.int64)
    local_state = np.asarray(getattr(record, "local_state"), dtype=np.complex128)
    if support.ndim != 1 or local_state.ndim != 1 or support.size != local_state.size:
        raise ValueError("Each cage record must provide matching support/local_state arrays.")

    vector = np.zeros(basis.n_states, dtype=np.complex128)
    for index, coefficient in zip(support.tolist(), local_state.tolist()):
        index = int(index)
        if index < 0 or index >= basis.n_states:
            raise IndexError(f"support index {index} outside [0, {basis.n_states}).")
        vector[index] += complex(coefficient)
    return vector


def _map_basis_index(source_basis: BasisLike, target_basis: BasisLike, source_index: int) -> int:
    if isinstance(target_basis, BinaryEncodedBasis):
        code = _basis_code(source_basis, source_index)
        target_index = target_basis.get_index(code)
    else:
        config = _basis_config(source_basis, source_index)
        target_index = target_basis.get_index(config)

    if target_index is None:
        raise KeyError(
            "Source cage record cannot be embedded because one support state is missing "
            "from the ambient basis. Pass a larger ambient_basis to CageSectorCollection."
        )
    return int(target_index)


def _basis_code(basis: BasisLike, basis_index: int) -> int:
    if isinstance(basis, BinaryEncodedBasis):
        return basis.code(basis_index)
    return encode_binary_config(basis.state(basis_index, copy=False))


def _basis_config(basis: BasisLike, basis_index: int) -> npt.NDArray[np.int64]:
    if isinstance(basis, BinaryEncodedBasis):
        return decode_binary_code(basis.code(basis_index), basis.n_variables)
    return basis.state(basis_index, copy=True)


def _extract_basis(obj: Any) -> BasisLike:
    basis = getattr(obj, "basis", obj)
    if not isinstance(basis, (Basis, BinaryEncodedBasis)):
        raise TypeError("Expected a Basis, BinaryEncodedBasis, or object exposing .basis.")
    return basis


def _iter_records(cage_result: Any) -> tuple[Any, ...]:
    records = getattr(cage_result, "records", None)
    if records is not None:
        return tuple(records)
    return tuple(cage_result)


def _record_signature(record: Any) -> tuple[int, int] | None:
    signature = getattr(record, "signature", None)
    if signature is None:
        return None
    return _normalize_signature(signature)


def _normalize_signature(signature: tuple[int, int]) -> tuple[int, int]:
    if len(signature) != 2:
        raise ValueError("cage signatures must have the form (kappa, potential_value).")
    return (int(signature[0]), int(signature[1]))


def _normalize_signature_filter(
    signature: tuple[int, int] | None,
    signatures: Sequence[tuple[int, int]] | None,
) -> tuple[tuple[int, int], ...] | None:
    if signature is not None:
        return (_normalize_signature(signature),)
    if signatures is None:
        return None
    return tuple(_normalize_signature(sig) for sig in signatures)


def _require_compatible_layouts(bases: Iterable[BasisLike]) -> None:
    bases = tuple(bases)
    if len(bases) == 0:
        return

    reference = bases[0].layout
    for basis in bases[1:]:
        layout = basis.layout
        if layout.n_variables != reference.n_variables:
            raise ValueError("All sector bases must use layouts with the same n_variables.")
        for variable_index in range(reference.n_variables):
            ref_values = reference.local_space(variable_index).values
            values = layout.local_space(variable_index).values
            if not np.array_equal(ref_values, values):
                raise ValueError("All sector bases must use compatible local spaces.")
