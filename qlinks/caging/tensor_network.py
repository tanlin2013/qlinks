"""Tensor-network ansatzes for constrained square-lattice QDM states.

The first tensor-network layer implemented here is a rectangular *vertex
PEPS*.  A tile owns every oriented ``+x`` and ``+y`` link whose source site is
inside the tile.  These owned links form a disjoint physical partition of a
periodic square lattice.  The virtual legs carry the occupations of links
entering the tile from the left and from below, while the right and upper
virtual legs copy the corresponding owned boundary-link occupations to the
neighbouring tiles.

Consequently, the structural tensor enforces the dimer constraint exactly,
without expanding the global constrained basis.  Variational tensor entries
then assign amplitudes to the locally allowed vertex configurations.  The
result can be materialized as a :class:`quimb.tensor.PEPS`, or evaluated exactly
on small qlinks bases for residual diagnostics and optimizer prototyping.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy import sparse as scipy_sparse

from qlinks.caging.singlet_product import (
    SquareQDMTwoPlaquetteSingletBlock,
    quimb_available,
)

SquareQDMTileTensorParameterization = Literal["entry", "physical"]


def autograd_available() -> bool:
    """Return whether the optional Autograd optimization backend is installed."""
    return importlib.util.find_spec("autograd") is not None


def _require_quimb() -> Any:
    if not quimb_available():
        raise ImportError(
            "quimb is not installed. Install qlinks with the 'tn' extra, "
            "for example `pip install 'qlinks[tn]'`."
        )
    import quimb.tensor as qtn  # type: ignore[import-not-found]

    return qtn


def _square_dimensions(model: object) -> tuple[int, int]:
    lattice = getattr(model, "lattice", None)
    lx = getattr(lattice, "lx", None)
    ly = getattr(lattice, "ly", None)
    if lx is None or ly is None:
        raise TypeError("The tensor-network square-QDM API requires a square lattice.")
    return int(lx), int(ly)


def _site_id(lattice: object, x: int, y: int) -> int:
    site_id = getattr(lattice, "site_id", None)
    if site_id is None:
        raise TypeError("The tensor-network square-QDM API requires lattice.site_id().")
    return int(site_id(int(x), int(y)))


def _outgoing_link_id(lattice: object, site_id: int, kind: str) -> int:
    matches = [
        int(link_id)
        for link_id in lattice.outgoing_links(int(site_id))
        if str(lattice.links[int(link_id)].kind) == str(kind)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one outgoing {kind!r} link at site {site_id}, found {matches}.")
    return matches[0]


def _incoming_link_id(lattice: object, site_id: int, kind: str) -> int:
    matches = [
        int(link_id)
        for link_id in lattice.incoming_links(int(site_id))
        if str(lattice.links[int(link_id)].kind) == str(kind)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one incoming {kind!r} link at site {site_id}, found {matches}.")
    return matches[0]


def _pattern_from_bits(bits: Sequence[int]) -> int:
    value = 0
    for position, bit in enumerate(bits):
        value |= int(bit) << position
    return int(value)


def _tensor_coordinate_key(coordinate: npt.ArrayLike) -> tuple[int, int, int, int, int]:
    values = np.asarray(coordinate, dtype=np.int64).reshape(-1)
    if values.size != 5:
        raise ValueError("A PEPS tensor coordinate must have five entries in urdlp order.")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class SquareQDMRectangularTileTensorBasis:
    """Locally allowed entries of one rectangular square-QDM vertex tensor.

    The tensor-index order is ``(up, right, down, left, physical)``.  The
    physical index labels configurations of all ``+x`` and ``+y`` links whose
    source lies inside the tile.  Thus physical links are disjoint between
    translated tiles.  A tensor entry is retained exactly when the incoming
    left/down occupations complete every site in the tile to one dimer and the
    outgoing right/up patterns agree with the owned physical configuration.
    """

    tile_shape: tuple[int, int]
    origin: tuple[int, int]
    owned_link_ids: npt.NDArray[np.int64]
    owned_link_keys: tuple[tuple[int, int, str], ...]
    physical_configurations: npt.NDArray[np.int8]
    entry_coordinates: npt.NDArray[np.int64]
    required_count: int = 1
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tile_shape = tuple(int(value) for value in self.tile_shape)
        origin = tuple(int(value) for value in self.origin)
        if len(tile_shape) != 2 or min(tile_shape) <= 0:
            raise ValueError("tile_shape must contain two positive integers.")
        if len(origin) != 2:
            raise ValueError("origin must contain two integers.")
        owned_link_ids = np.asarray(self.owned_link_ids, dtype=np.int64)
        physical = np.asarray(self.physical_configurations, dtype=np.int8)
        coordinates = np.asarray(self.entry_coordinates, dtype=np.int64)
        if owned_link_ids.ndim != 1:
            raise ValueError("owned_link_ids must be one-dimensional.")
        if physical.ndim != 2 or physical.shape[1] != owned_link_ids.size:
            raise ValueError("physical_configurations has the wrong shape.")
        if coordinates.ndim != 2 or coordinates.shape[1] != 5:
            raise ValueError("entry_coordinates must have shape (n_entries, 5).")
        if len(self.owned_link_keys) != owned_link_ids.size:
            raise ValueError("owned_link_keys must align with owned_link_ids.")
        if coordinates.shape[0] == 0:
            raise ValueError("At least one locally allowed tensor entry is required.")
        if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
            raise ValueError("entry_coordinates must be unique.")
        tensor_shape = self.tensor_shape
        if np.any(coordinates < 0):
            raise ValueError("entry_coordinates must be non-negative.")
        for axis, size in enumerate(tensor_shape):
            if np.any(coordinates[:, axis] >= size):
                raise ValueError(
                    f"entry coordinate axis {axis} exceeds tensor shape {tensor_shape}."
                )
        object.__setattr__(self, "tile_shape", tile_shape)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "owned_link_ids", owned_link_ids.copy())
        object.__setattr__(self, "physical_configurations", physical.copy())
        object.__setattr__(self, "entry_coordinates", coordinates.copy())
        object.__setattr__(self, "required_count", int(self.required_count))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def tile_lx(self) -> int:
        return int(self.tile_shape[0])

    @property
    def tile_ly(self) -> int:
        return int(self.tile_shape[1])

    @property
    def physical_dimension(self) -> int:
        return int(self.physical_configurations.shape[0])

    @property
    def n_entries(self) -> int:
        return int(self.entry_coordinates.shape[0])

    @property
    def up_dimension(self) -> int:
        return 1 << self.tile_lx

    @property
    def right_dimension(self) -> int:
        return 1 << self.tile_ly

    @property
    def down_dimension(self) -> int:
        return 1 << self.tile_lx

    @property
    def left_dimension(self) -> int:
        return 1 << self.tile_ly

    @property
    def tensor_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.up_dimension,
            self.right_dimension,
            self.down_dimension,
            self.left_dimension,
            self.physical_dimension,
        )

    @property
    def compression_ratio(self) -> float:
        full_dimension = 1 << int(self.owned_link_ids.size)
        return float(self.physical_dimension / full_dimension)

    def entry_coordinate_lookup(self) -> dict[tuple[int, int, int, int, int], int]:
        return {
            _tensor_coordinate_key(coordinate): index
            for index, coordinate in enumerate(self.entry_coordinates)
        }

    def physical_configuration_lookup(self) -> dict[tuple[int, ...], int]:
        return {
            tuple(int(bit) for bit in config): index
            for index, config in enumerate(self.physical_configurations)
        }

    def tensor_data_from_parameters(
        self,
        parameters: npt.ArrayLike,
        *,
        parameterization: SquareQDMTileTensorParameterization = "entry",
    ) -> npt.NDArray[np.complex128]:
        """Embed a compact parameter vector into the masked dense PEPS tensor."""
        values = np.asarray(parameters, dtype=np.complex128).reshape(-1)
        if parameterization == "entry":
            if values.size != self.n_entries:
                raise ValueError(f"entry parameters must have size {self.n_entries}.")
            entry_values = values
        elif parameterization == "physical":
            if values.size != self.physical_dimension:
                raise ValueError(f"physical parameters must have size {self.physical_dimension}.")
            entry_values = values[self.entry_coordinates[:, 4]]
        else:
            raise ValueError("parameterization must be 'entry' or 'physical'.")
        data = np.zeros(self.tensor_shape, dtype=np.complex128)
        coordinate_tuple = tuple(self.entry_coordinates[:, axis] for axis in range(5))
        data[coordinate_tuple] = entry_values
        return data

    def entry_parameters_from_tensor_data(
        self,
        tensor_data: npt.ArrayLike,
    ) -> npt.NDArray[np.complex128]:
        data = np.asarray(tensor_data, dtype=np.complex128)
        if data.shape != self.tensor_shape:
            raise ValueError(f"tensor_data must have shape {self.tensor_shape}.")
        coordinate_tuple = tuple(self.entry_coordinates[:, axis] for axis in range(5))
        return np.asarray(data[coordinate_tuple], dtype=np.complex128)

    def structural_tensor_data(self) -> npt.NDArray[np.complex128]:
        return self.tensor_data_from_parameters(np.ones(self.n_entries, dtype=np.complex128))


@dataclass(frozen=True, slots=True)
class SquareQDMPEPSAnsatz:
    """One translationally invariant constrained PEPS unit-cell tensor."""

    tile_basis: SquareQDMRectangularTileTensorBasis
    parameters: npt.NDArray[np.complex128]
    parameterization: SquareQDMTileTensorParameterization = "entry"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parameters = np.asarray(self.parameters, dtype=np.complex128).reshape(-1)
        expected = (
            self.tile_basis.n_entries
            if self.parameterization == "entry"
            else self.tile_basis.physical_dimension
        )
        if parameters.size != expected:
            raise ValueError(
                f"parameters has size {parameters.size}, expected {expected} "
                f"for {self.parameterization!r} parameterization."
            )
        object.__setattr__(self, "parameters", parameters.copy())
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def tensor_data(self) -> npt.NDArray[np.complex128]:
        return self.tile_basis.tensor_data_from_parameters(
            self.parameters,
            parameterization=self.parameterization,
        )

    @property
    def n_parameters(self) -> int:
        return int(self.parameters.size)

    def with_parameters(self, parameters: npt.ArrayLike) -> SquareQDMPEPSAnsatz:
        return SquareQDMPEPSAnsatz(
            tile_basis=self.tile_basis,
            parameters=np.asarray(parameters, dtype=np.complex128),
            parameterization=self.parameterization,
            metadata=self.metadata,
        )

    def to_quimb_tensor_network(
        self,
        *,
        n_tiles_x: int,
        n_tiles_y: int,
        tags: str | Sequence[str] | None = ("QLINKS", "UNIT_CELL"),
    ) -> object:
        """Build the periodic PEPS as a generic quimb tensor network.

        The bonds are named explicitly, so this representation also handles
        one- and two-tile periodic directions, where a pair of neighbouring
        tensors is connected by parallel bonds.
        """
        if n_tiles_x <= 0 or n_tiles_y <= 0:
            raise ValueError("n_tiles_x and n_tiles_y must be positive.")
        qtn = _require_quimb()
        if tags is None:
            global_tags: tuple[str, ...] = ()
        elif isinstance(tags, str):
            global_tags = (tags,)
        else:
            global_tags = tuple(str(tag) for tag in tags)
        data = self.tensor_data
        tensors = []
        for tile_y in range(int(n_tiles_y)):
            for tile_x in range(int(n_tiles_x)):
                up = f"bond_y_{tile_x}_{tile_y}"
                right = f"bond_x_{tile_x}_{tile_y}"
                down = f"bond_y_{tile_x}_{(tile_y - 1) % int(n_tiles_y)}"
                left = f"bond_x_{(tile_x - 1) % int(n_tiles_x)}_{tile_y}"
                physical = f"physical_{tile_x}_{tile_y}"
                site_tags = (
                    *global_tags,
                    f"I{tile_y},{tile_x}",
                    f"X{tile_y}",
                    f"Y{tile_x}",
                )
                tensors.append(
                    qtn.Tensor(
                        data=data.copy(),
                        inds=(up, right, down, left, physical),
                        tags=site_tags,
                    )
                )
        return qtn.TensorNetwork(tensors, virtual=True)

    def to_quimb_peps(
        self,
        *,
        n_tiles_x: int,
        n_tiles_y: int,
        tags: str | Sequence[str] | None = ("QLINKS", "UNIT_CELL"),
    ) -> object:
        """Build a structured periodic :class:`quimb.tensor.PEPS`.

        Quimb's structured constructor identifies bonds by the unordered pair
        of neighbouring coordinates.  Periodic directions of length one or two
        contain parallel bonds between the same coordinates, so use
        :meth:`to_quimb_tensor_network` for those short tori.
        """
        if n_tiles_x < 3 or n_tiles_y < 3:
            raise ValueError(
                "Structured periodic quimb PEPS requires at least three tiles in each "
                "direction; use to_quimb_tensor_network() for shorter tori."
            )
        qtn = _require_quimb()
        data = self.tensor_data
        arrays = tuple(
            tuple(data.copy() for _ in range(int(n_tiles_x))) for _ in range(int(n_tiles_y))
        )
        return qtn.PEPS(arrays, shape="urdlp", tags=tags)


@dataclass(frozen=True, slots=True)
class SquareQDMPEPSResidualReport:
    """Exact finite-cluster eigenstate diagnostic for a PEPS parameter vector."""

    norm: float
    energy: complex
    residual: float
    energy_variance: float
    nonzero_basis_amplitudes: int
    hilbert_dimension: int

    @property
    def is_nonzero(self) -> bool:
        return self.norm > 0.0


@dataclass(frozen=True, slots=True)
class SquareQDMPEPSOptimizationResult:
    """Result of an exact small-cluster PEPS optimization."""

    initial_parameters: npt.NDArray[np.float64]
    optimized_parameters: npt.NDArray[np.float64]
    loss_history: tuple[float, ...]
    initial_report: SquareQDMPEPSResidualReport
    final_report: SquareQDMPEPSResidualReport
    requested_steps: int
    optimizer: str
    autodiff_backend: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_parameters, dtype=np.float64).reshape(-1)
        optimized = np.asarray(self.optimized_parameters, dtype=np.float64).reshape(-1)
        if initial.shape != optimized.shape:
            raise ValueError("initial_parameters and optimized_parameters must align.")
        object.__setattr__(self, "initial_parameters", initial.copy())
        object.__setattr__(self, "optimized_parameters", optimized.copy())
        object.__setattr__(self, "loss_history", tuple(float(value) for value in self.loss_history))
        object.__setattr__(self, "requested_steps", int(self.requested_steps))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def initial_loss(self) -> float:
        return float(self.initial_report.energy_variance)

    @property
    def final_loss(self) -> float:
        return float(self.final_report.energy_variance)

    @property
    def improved(self) -> bool:
        return self.final_loss < self.initial_loss

    @property
    def reached_exact_state(self) -> bool:
        tolerance = float(self.metadata.get("exact_tolerance", 1.0e-10))
        return self.final_report.residual <= tolerance

    def to_ansatz(
        self,
        tile_basis: SquareQDMRectangularTileTensorBasis,
    ) -> SquareQDMPEPSAnsatz:
        """Return the optimized parameters as a reusable PEPS ansatz."""
        return SquareQDMPEPSAnsatz(
            tile_basis=tile_basis,
            parameters=self.optimized_parameters.astype(np.complex128),
            parameterization="entry",
            metadata={
                **dict(self.metadata),
                "source": "finite_cluster_optimization",
                "final_energy_variance": self.final_loss,
            },
        )


@dataclass(frozen=True, slots=True)
class SquareQDMPEPSFiniteClusterProblem:
    """Exact small-torus objective for one translational PEPS unit tensor."""

    model: object
    tile_basis: SquareQDMRectangularTileTensorBasis
    basis_states: npt.NDArray[np.int8]
    hamiltonian: scipy_sparse.csr_array
    tile_coordinates: tuple[tuple[int, int], ...]
    entry_parameter_indices: npt.NDArray[np.int64]
    tensor_coordinates: npt.NDArray[np.int64]
    tolerance: float = 1.0e-10
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        states = np.asarray(self.basis_states, dtype=np.int8)
        hamiltonian = scipy_sparse.csr_array(self.hamiltonian, dtype=np.complex128)
        parameter_indices = np.asarray(self.entry_parameter_indices, dtype=np.int64)
        coordinates = np.asarray(self.tensor_coordinates, dtype=np.int64)
        if states.ndim != 2:
            raise ValueError("basis_states must be two-dimensional.")
        if hamiltonian.shape != (states.shape[0], states.shape[0]):
            raise ValueError("hamiltonian shape must match basis_states.")
        expected_shape = (states.shape[0], len(self.tile_coordinates))
        if parameter_indices.shape != expected_shape:
            raise ValueError("entry_parameter_indices has the wrong shape.")
        if coordinates.shape != (*expected_shape, 5):
            raise ValueError("tensor_coordinates has the wrong shape.")
        if np.any(parameter_indices < 0) or np.any(parameter_indices >= self.tile_basis.n_entries):
            raise ValueError("entry_parameter_indices contains an invalid entry.")
        object.__setattr__(self, "basis_states", states.copy())
        object.__setattr__(self, "hamiltonian", hamiltonian)
        object.__setattr__(self, "entry_parameter_indices", parameter_indices.copy())
        object.__setattr__(self, "tensor_coordinates", coordinates.copy())
        object.__setattr__(self, "tolerance", float(self.tolerance))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def hilbert_dimension(self) -> int:
        return int(self.basis_states.shape[0])

    @property
    def n_tiles(self) -> int:
        return len(self.tile_coordinates)

    @property
    def n_tiles_x(self) -> int:
        return int(self.metadata["n_tiles_x"])

    @property
    def n_tiles_y(self) -> int:
        return int(self.metadata["n_tiles_y"])

    def state_vector(
        self,
        parameters: npt.ArrayLike,
        *,
        normalize: bool = True,
    ) -> npt.NDArray[np.complex128]:
        """Evaluate the PEPS amplitudes on the exact constrained basis.

        The virtual indices are fixed uniquely by a global link configuration,
        so evaluating a basis amplitude reduces to multiplying the selected
        unit-tensor entries across tiles.
        """
        values = np.asarray(parameters, dtype=np.complex128).reshape(-1)
        if values.size != self.tile_basis.n_entries:
            raise ValueError(f"parameters must have size {self.tile_basis.n_entries}.")
        selected = values[self.entry_parameter_indices]
        state = np.prod(selected, axis=1, dtype=np.complex128)
        if normalize:
            norm = float(np.linalg.norm(state))
            if norm == 0.0:
                raise ValueError("The PEPS parameter vector produces the zero state.")
            state = state / norm
        return np.asarray(state, dtype=np.complex128)

    def diagnose(
        self,
        parameters: npt.ArrayLike,
        *,
        energy: complex | None = None,
    ) -> SquareQDMPEPSResidualReport:
        raw_state = self.state_vector(parameters, normalize=False)
        norm = float(np.linalg.norm(raw_state))
        if norm == 0.0:
            return SquareQDMPEPSResidualReport(
                norm=0.0,
                energy=0.0,
                residual=float("inf"),
                energy_variance=float("inf"),
                nonzero_basis_amplitudes=0,
                hilbert_dimension=self.hilbert_dimension,
            )
        state = raw_state / norm
        h_state = np.asarray(self.hamiltonian @ state, dtype=np.complex128)
        inferred_energy = complex(np.vdot(state, h_state))
        target_energy = inferred_energy if energy is None else complex(energy)
        residual_vector = h_state - target_energy * state
        residual = float(np.linalg.norm(residual_vector))
        h2 = float(np.vdot(h_state, h_state).real)
        variance = max(0.0, h2 - float(abs(inferred_energy) ** 2))
        return SquareQDMPEPSResidualReport(
            norm=norm,
            energy=inferred_energy,
            residual=residual,
            energy_variance=variance,
            nonzero_basis_amplitudes=int(np.count_nonzero(np.abs(raw_state) > self.tolerance)),
            hilbert_dimension=self.hilbert_dimension,
        )

    def loss(self, parameters: npt.ArrayLike) -> float:
        report = self.diagnose(parameters)
        return float(report.residual**2)

    def perturb_parameters(
        self,
        parameters: npt.ArrayLike,
        *,
        scale: float = 1.0e-2,
        seed: int | None = 0,
        normalize: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Add a reproducible real perturbation to escape sparse stationary points.

        The exact singlet-core tensor has only two nonzero entries and is a
        stationary point of the variance within the enlarged PEPS manifold.
        A small perturbation activates the boundary-compatible halo entries and
        produces a useful Autograd search direction.
        """
        values = np.asarray(parameters, dtype=np.complex128).reshape(-1)
        if values.size != self.tile_basis.n_entries:
            raise ValueError(f"parameters must have size {self.tile_basis.n_entries}.")
        if np.max(np.abs(values.imag), initial=0.0) > self.tolerance:
            raise ValueError("The Autograd prototype currently optimizes real parameters only.")
        if scale < 0.0:
            raise ValueError("scale must be non-negative.")
        real_values = np.asarray(values.real, dtype=np.float64)
        if scale > 0.0:
            rng = np.random.default_rng(seed)
            real_values = real_values + float(scale) * rng.normal(size=real_values.size)
        if normalize:
            norm = float(np.linalg.norm(real_values))
            if norm == 0.0:
                raise ValueError("Cannot normalize the zero parameter vector.")
            real_values = real_values / norm
        return np.asarray(real_values, dtype=np.float64)

    def loss_and_gradient_autograd(
        self,
        parameters: npt.ArrayLike,
    ) -> tuple[float, npt.NDArray[np.float64]]:
        """Evaluate the exact variance and its gradient using Autograd."""
        if not autograd_available():
            raise ImportError(
                "autograd is not installed. Install qlinks with the 'tn' extra, "
                "for example `pip install 'qlinks[tn]'`."
            )
        import autograd.numpy as anp  # type: ignore[import-not-found]
        from autograd import value_and_grad  # type: ignore[import-not-found]

        values = np.asarray(parameters, dtype=np.complex128).reshape(-1)
        if values.size != self.tile_basis.n_entries:
            raise ValueError(f"parameters must have size {self.tile_basis.n_entries}.")
        if np.max(np.abs(values.imag), initial=0.0) > self.tolerance:
            raise ValueError("The Autograd prototype currently optimizes real parameters only.")
        parameter_indices = np.asarray(self.entry_parameter_indices, dtype=np.int64)
        hamiltonian = anp.asarray(self.hamiltonian.toarray().real)

        def objective(compact_parameters: Any) -> Any:
            state = anp.prod(compact_parameters[parameter_indices], axis=1)
            norm_squared = anp.sum(state * state) + 1.0e-30
            h_state = anp.dot(hamiltonian, state)
            energy = anp.sum(state * h_state) / norm_squared
            residual = h_state - energy * state
            return anp.sum(residual * residual) / norm_squared

        loss, gradient = value_and_grad(objective)(np.asarray(values.real, dtype=np.float64))
        return float(loss), np.asarray(gradient, dtype=np.float64)

    def make_quimb_optimizer(
        self,
        parameters: npt.ArrayLike,
        *,
        autodiff_backend: str = "autograd",
        optimizer: str = "L-BFGS-B",
        progbar: bool = True,
        loss_target: float | None = None,
        **backend_options: object,
    ) -> object:
        """Create a compact quimb ``TNOptimizer`` for the exact variance.

        Only the locally allowed tensor entries are variational.  Quimb's
        :class:`~quimb.tensor.PTensor` embeds the compact vector into the dense
        ``urdlp`` tensor, reducing the optimization dimension from the full
        tensor size to ``tile_basis.n_entries`` (108 for the 3-by-2 tile).

        The current Autograd path is real-valued.  Complex tensors can later be
        handled through a JAX or PyTorch backend with an explicit real/imaginary
        parameter split.
        """
        qtn = _require_quimb()
        if autodiff_backend.lower() == "autograd" and not autograd_available():
            raise ImportError(
                "autograd is not installed. Install qlinks with the 'tn' extra, "
                "for example `pip install 'qlinks[tn]'`."
            )
        values = np.asarray(parameters, dtype=np.complex128).reshape(-1)
        if values.size != self.tile_basis.n_entries:
            raise ValueError(f"parameters must have size {self.tile_basis.n_entries}.")
        if np.max(np.abs(values.imag), initial=0.0) > self.tolerance:
            raise ValueError("The current compact optimizer accepts real parameters only.")

        tensor_shape = self.tile_basis.tensor_shape
        coordinates = self.tile_basis.entry_coordinates
        flat_coordinates = np.ravel_multi_index(
            tuple(coordinates[:, axis] for axis in range(5)),
            tensor_shape,
        )
        entry_index_map = np.zeros(int(np.prod(tensor_shape)), dtype=np.int64)
        structural_mask = np.zeros(int(np.prod(tensor_shape)), dtype=np.float64)
        entry_index_map[flat_coordinates] = np.arange(self.tile_basis.n_entries)
        structural_mask[flat_coordinates] = 1.0
        entry_index_map = entry_index_map.reshape(tensor_shape)
        structural_mask = structural_mask.reshape(tensor_shape)

        def compact_to_tensor(compact_parameters: Any) -> Any:
            return compact_parameters[entry_index_map] * structural_mask

        tensor = qtn.PTensor(
            compact_to_tensor,
            np.asarray(values.real, dtype=np.float64),
            inds=("up", "right", "down", "left", "physical"),
            tags={"UNIT_CELL"},
        )
        network = qtn.TensorNetwork([tensor])
        dense_hamiltonian = np.asarray(self.hamiltonian.toarray().real, dtype=np.float64)
        return qtn.TNOptimizer(
            network,
            _quimb_exact_cluster_loss,
            loss_constants={
                "tensor_coordinates": self.tensor_coordinates,
                "hamiltonian": dense_hamiltonian,
            },
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            loss_target=loss_target,
            **backend_options,
        )

    def optimize_with_quimb(
        self,
        parameters: npt.ArrayLike,
        *,
        max_steps: int = 20,
        noise_scale: float = 1.0e-2,
        seed: int | None = 0,
        autodiff_backend: str = "autograd",
        optimizer: str = "L-BFGS-B",
        progbar: bool = False,
        loss_target: float | None = None,
        exact_tolerance: float = 1.0e-10,
        **backend_options: object,
    ) -> SquareQDMPEPSOptimizationResult:
        """Optimize the shared unit tensor against the exact cluster variance."""
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        initial_parameters = self.perturb_parameters(
            parameters,
            scale=noise_scale,
            seed=seed,
            normalize=True,
        )
        initial_report = self.diagnose(initial_parameters)
        optimizer_object = self.make_quimb_optimizer(
            initial_parameters,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            loss_target=loss_target,
            **backend_options,
        )
        optimized_network = optimizer_object.optimize(int(max_steps))
        optimized_parameters = np.asarray(
            optimized_network["UNIT_CELL"].params,
            dtype=np.float64,
        ).reshape(-1)
        final_report = self.diagnose(optimized_parameters)
        return SquareQDMPEPSOptimizationResult(
            initial_parameters=initial_parameters,
            optimized_parameters=optimized_parameters,
            loss_history=tuple(float(value) for value in optimizer_object.losses),
            initial_report=initial_report,
            final_report=final_report,
            requested_steps=int(max_steps),
            optimizer=str(optimizer),
            autodiff_backend=str(autodiff_backend),
            metadata={
                "exact_tolerance": float(exact_tolerance),
                "noise_scale": float(noise_scale),
                "seed": seed,
                "optimizer_dimension": int(getattr(optimizer_object, "d", -1)),
                "n_function_evaluations": len(optimizer_object.losses),
            },
        )


def _quimb_exact_cluster_loss(
    tensor_network: object,
    *,
    tensor_coordinates: Any,
    hamiltonian: Any,
) -> Any:
    """Autodiff-compatible real variance used by the compact optimizer."""
    import autoray as ar

    unit_tensor = tensor_network["UNIT_CELL"].data
    n_basis = int(tensor_coordinates.shape[0])
    n_tiles = int(tensor_coordinates.shape[1])
    state = None
    for tile_index in range(n_tiles):
        coordinates = tensor_coordinates[:, tile_index, :]
        selected = unit_tensor[
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            coordinates[:, 3],
            coordinates[:, 4],
        ]
        state = selected if state is None else state * selected
    if state is None:
        state = ar.do("ones", (n_basis,), like=unit_tensor)
    norm_squared = ar.do("sum", state * state) + 1.0e-30
    h_state = hamiltonian @ state
    energy = ar.do("sum", state * h_state) / norm_squared
    residual = h_state - energy * state
    return ar.do("sum", residual * residual) / norm_squared


def build_square_qdm_rectangular_tile_tensor_basis(
    model: object,
    *,
    tile_shape: tuple[int, int],
    origin: tuple[int, int] = (0, 0),
    max_owned_links: int = 20,
) -> SquareQDMRectangularTileTensorBasis:
    """Enumerate the exact structural support of a rectangular QDM tile tensor."""
    lx, ly = _square_dimensions(model)
    tile_lx, tile_ly = (int(tile_shape[0]), int(tile_shape[1]))
    origin_x, origin_y = (int(origin[0]), int(origin[1]))
    if tile_lx <= 0 or tile_ly <= 0:
        raise ValueError("tile_shape entries must be positive.")
    if tile_lx > lx or tile_ly > ly:
        raise ValueError("tile_shape must fit inside the host lattice.")
    # Avoid using a host-periodic seam to define the abstract unit tensor.  The
    # resulting tensor can still be repeated periodically on arbitrary tori.
    if origin_x < 0 or origin_y < 0 or origin_x + tile_lx >= lx or origin_y + tile_ly >= ly:
        raise ValueError(
            "The host tile must be strictly interior so its local link orientation is unambiguous."
        )
    lattice = model.lattice
    required_count = int(getattr(model, "required_count", 1))
    if required_count != 1:
        raise ValueError("The current QDM vertex tensor implementation assumes one dimer per site.")

    owned_link_ids: list[int] = []
    owned_link_keys: list[tuple[int, int, str]] = []
    for relative_x in range(tile_lx):
        for relative_y in range(tile_ly):
            site = _site_id(lattice, origin_x + relative_x, origin_y + relative_y)
            for kind in ("x", "y"):
                owned_link_ids.append(_outgoing_link_id(lattice, site, kind))
                owned_link_keys.append((relative_x, relative_y, kind))
    if len(owned_link_ids) > int(max_owned_links):
        raise ValueError(
            f"Tile owns {len(owned_link_ids)} links, exceeding max_owned_links={max_owned_links}."
        )
    owned_position = {link_id: index for index, link_id in enumerate(owned_link_ids)}

    right_positions = [
        owned_position[
            _outgoing_link_id(
                lattice,
                _site_id(lattice, origin_x + tile_lx - 1, origin_y + relative_y),
                "x",
            )
        ]
        for relative_y in range(tile_ly)
    ]
    up_positions = [
        owned_position[
            _outgoing_link_id(
                lattice,
                _site_id(lattice, origin_x + relative_x, origin_y + tile_ly - 1),
                "y",
            )
        ]
        for relative_x in range(tile_lx)
    ]

    physical_configurations: list[npt.NDArray[np.int8]] = []
    entry_coordinates: list[tuple[int, int, int, int, int]] = []
    for integer_config in range(1 << len(owned_link_ids)):
        config = np.asarray(
            [(integer_config >> position) & 1 for position in range(len(owned_link_ids))],
            dtype=np.int8,
        )
        fixed_left_bits = [0] * tile_ly
        fixed_down_bits = [0] * tile_lx
        corner_deficit: int | None = None
        valid = True
        for relative_x in range(tile_lx):
            for relative_y in range(tile_ly):
                site = _site_id(lattice, origin_x + relative_x, origin_y + relative_y)
                local_count = sum(
                    int(config[owned_position[int(link_id)]])
                    for link_id in lattice.incident_links(site)
                    if int(link_id) in owned_position
                )
                missing_inputs = int(relative_x == 0) + int(relative_y == 0)
                deficit = required_count - local_count
                if deficit < 0 or deficit > missing_inputs:
                    valid = False
                    break
                if relative_x == 0 and relative_y == 0:
                    corner_deficit = deficit
                elif relative_x == 0:
                    fixed_left_bits[relative_y] = deficit
                elif relative_y == 0:
                    fixed_down_bits[relative_x] = deficit
                elif deficit != 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid or corner_deficit is None:
            continue

        corner_choices = tuple(
            (left_bit, down_bit)
            for left_bit in (0, 1)
            for down_bit in (0, 1)
            if left_bit + down_bit == corner_deficit
        )
        if not corner_choices:
            continue
        physical_index = len(physical_configurations)
        physical_configurations.append(config)
        right_pattern = _pattern_from_bits([config[position] for position in right_positions])
        up_pattern = _pattern_from_bits([config[position] for position in up_positions])
        for corner_left, corner_down in corner_choices:
            left_bits = list(fixed_left_bits)
            down_bits = list(fixed_down_bits)
            left_bits[0] = corner_left
            down_bits[0] = corner_down
            entry_coordinates.append(
                (
                    up_pattern,
                    right_pattern,
                    _pattern_from_bits(down_bits),
                    _pattern_from_bits(left_bits),
                    physical_index,
                )
            )

    return SquareQDMRectangularTileTensorBasis(
        tile_shape=(tile_lx, tile_ly),
        origin=(origin_x, origin_y),
        owned_link_ids=np.asarray(owned_link_ids, dtype=np.int64),
        owned_link_keys=tuple(owned_link_keys),
        physical_configurations=np.asarray(physical_configurations, dtype=np.int8),
        entry_coordinates=np.asarray(entry_coordinates, dtype=np.int64),
        required_count=required_count,
        metadata={
            "host_shape": (lx, ly),
            "full_owned_configuration_count": 1 << len(owned_link_ids),
        },
    )


def build_square_qdm_singlet_peps_ansatz(
    model: object,
    singlet: SquareQDMTwoPlaquetteSingletBlock,
    *,
    origin: tuple[int, int] | None = None,
    normalize_parameters: bool = True,
) -> SquareQDMPEPSAnsatz:
    """Construct the singlet-core starting tensor in a rectangular PEPS tile."""
    anchors = tuple(tuple(int(value) for value in cell) for cell in singlet.anchor_cells)
    if singlet.direction == "x":
        tile_shape = (3, 2)
    elif singlet.direction == "y":
        tile_shape = (2, 3)
    else:  # pragma: no cover - guarded by the wrapper type
        raise ValueError("Unsupported singlet direction.")
    if origin is None:
        origin = (
            min(cell[0] for cell in anchors),
            min(cell[1] for cell in anchors),
        )
    tile_basis = build_square_qdm_rectangular_tile_tensor_basis(
        model,
        tile_shape=tile_shape,
        origin=origin,
    )
    owned_position = {
        int(link_id): position for position, link_id in enumerate(tile_basis.owned_link_ids)
    }
    try:
        core_positions = np.asarray(
            [owned_position[int(link_id)] for link_id in singlet.block.link_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            "The selected singlet core is not contained in the rectangular tile."
        ) from error

    physical_sector = np.full(tile_basis.physical_dimension, -1, dtype=np.int64)
    for sector_index, core_config in enumerate(singlet.block.support_configs):
        matches = np.all(
            tile_basis.physical_configurations[:, core_positions]
            == np.asarray(core_config, dtype=np.int8),
            axis=1,
        )
        physical_sector[matches] = int(sector_index)
    amplitudes = np.asarray(singlet.block.amplitudes, dtype=np.complex128)
    parameters = np.zeros(tile_basis.n_entries, dtype=np.complex128)
    entry_physical = tile_basis.entry_coordinates[:, 4]
    valid_entries = physical_sector[entry_physical] >= 0
    parameters[valid_entries] = amplitudes[physical_sector[entry_physical[valid_entries]]]
    if normalize_parameters:
        norm = float(np.linalg.norm(parameters))
        if norm == 0.0:
            raise ValueError("The singlet core has no compatible rectangular-tile entries.")
        parameters = parameters / norm
    return SquareQDMPEPSAnsatz(
        tile_basis=tile_basis,
        parameters=parameters,
        parameterization="entry",
        metadata={
            "kind": "two_plaquette_singlet_core",
            "singlet_block_id": singlet.block_id,
            "singlet_direction": singlet.direction,
            "core_sector_by_physical_index": physical_sector,
            "n_core_compatible_entries": int(np.count_nonzero(valid_entries)),
        },
    )


def build_square_qdm_peps_finite_cluster_problem(
    model: object,
    tile_basis: SquareQDMRectangularTileTensorBasis,
    *,
    basis_solver: str = "dfs",
    builder: str = "sparse",
    tolerance: float = 1.0e-10,
) -> SquareQDMPEPSFiniteClusterProblem:
    """Map a translational tile tensor to an exact finite periodic QDM basis."""
    lx, ly = _square_dimensions(model)
    if lx % tile_basis.tile_lx != 0 or ly % tile_basis.tile_ly != 0:
        raise ValueError("The model dimensions must be integer multiples of the tile dimensions.")
    n_tiles_x = lx // tile_basis.tile_lx
    n_tiles_y = ly // tile_basis.tile_ly
    build_result = model.build(
        basis_solver=basis_solver,
        builder=builder,
    )
    basis_states = np.asarray(build_result.basis.states, dtype=np.int8)
    hamiltonian = scipy_sparse.csr_array(build_result.hamiltonian, dtype=np.complex128)
    lattice = model.lattice
    physical_lookup = tile_basis.physical_configuration_lookup()
    coordinate_lookup = tile_basis.entry_coordinate_lookup()

    tile_coordinates = tuple(
        (tile_x, tile_y) for tile_y in range(n_tiles_y) for tile_x in range(n_tiles_x)
    )
    entry_indices = np.empty((basis_states.shape[0], len(tile_coordinates)), dtype=np.int64)
    tensor_coordinates = np.empty(
        (basis_states.shape[0], len(tile_coordinates), 5),
        dtype=np.int64,
    )

    for tile_index, (tile_x, tile_y) in enumerate(tile_coordinates):
        origin_x = tile_x * tile_basis.tile_lx
        origin_y = tile_y * tile_basis.tile_ly
        owned_global_links = [
            _outgoing_link_id(
                lattice,
                _site_id(lattice, origin_x + relative_x, origin_y + relative_y),
                kind,
            )
            for relative_x, relative_y, kind in tile_basis.owned_link_keys
        ]
        left_global_links = [
            _incoming_link_id(
                lattice,
                _site_id(lattice, origin_x, origin_y + relative_y),
                "x",
            )
            for relative_y in range(tile_basis.tile_ly)
        ]
        down_global_links = [
            _incoming_link_id(
                lattice,
                _site_id(lattice, origin_x + relative_x, origin_y),
                "y",
            )
            for relative_x in range(tile_basis.tile_lx)
        ]
        right_global_positions = [
            tile_basis.owned_link_keys.index((tile_basis.tile_lx - 1, relative_y, "x"))
            for relative_y in range(tile_basis.tile_ly)
        ]
        up_global_positions = [
            tile_basis.owned_link_keys.index((relative_x, tile_basis.tile_ly - 1, "y"))
            for relative_x in range(tile_basis.tile_lx)
        ]

        for basis_index, global_config in enumerate(basis_states):
            local_config = np.asarray(global_config[owned_global_links], dtype=np.int8)
            try:
                physical_index = physical_lookup[tuple(int(bit) for bit in local_config)]
            except KeyError as error:
                raise ValueError(
                    "A globally valid dimer configuration is absent from the local tile basis."
                ) from error
            coordinate = (
                _pattern_from_bits([local_config[position] for position in up_global_positions]),
                _pattern_from_bits([local_config[position] for position in right_global_positions]),
                _pattern_from_bits(global_config[down_global_links]),
                _pattern_from_bits(global_config[left_global_links]),
                physical_index,
            )
            try:
                parameter_index = coordinate_lookup[coordinate]
            except KeyError as error:
                raise ValueError(
                    f"Global basis state produces a forbidden tensor coordinate {coordinate}."
                ) from error
            entry_indices[basis_index, tile_index] = parameter_index
            tensor_coordinates[basis_index, tile_index] = coordinate

    return SquareQDMPEPSFiniteClusterProblem(
        model=model,
        tile_basis=tile_basis,
        basis_states=basis_states,
        hamiltonian=hamiltonian,
        tile_coordinates=tile_coordinates,
        entry_parameter_indices=entry_indices,
        tensor_coordinates=tensor_coordinates,
        tolerance=tolerance,
        metadata={
            "n_tiles_x": n_tiles_x,
            "n_tiles_y": n_tiles_y,
            "model_shape": (lx, ly),
            "basis_solver": basis_solver,
            "builder": builder,
        },
    )


# ---------------------------------------------------------------------------
# Type-1 chiral PEPS specialization
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SquareQDMChiralParityRule:
    """Linear ``Z2`` representation of the Fock-space chiral operator.

    The rule assigns a parity

    ``q(config) = offset + link_coefficients @ config (mod 2)``.

    Every nonzero kinetic matrix element must connect configurations with
    opposite parity.  The corresponding chiral sign is ``(-1)**q``.  A linear
    rule is especially useful for tensor networks because the physical links
    are partitioned between tiles, so the global parity becomes a sum of local
    physical charges.
    """

    link_coefficients: npt.NDArray[np.int8]
    offset: int = 0
    n_edge_equations: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.link_coefficients, dtype=np.int8).reshape(-1) % 2
        object.__setattr__(self, "link_coefficients", coefficients.copy())
        object.__setattr__(self, "offset", int(self.offset) % 2)
        object.__setattr__(self, "n_edge_equations", int(self.n_edge_equations))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def labels(self, basis_states: npt.ArrayLike) -> npt.NDArray[np.int8]:
        states = np.asarray(basis_states, dtype=np.int8)
        if states.ndim != 2 or states.shape[1] != self.link_coefficients.size:
            raise ValueError("basis_states must align with the chiral link coefficients.")
        values = (states @ self.link_coefficients.astype(np.int64) + self.offset) % 2
        return np.asarray(values, dtype=np.int8)

    def signs(self, basis_states: npt.ArrayLike) -> npt.NDArray[np.int8]:
        return np.asarray(1 - 2 * self.labels(basis_states), dtype=np.int8)

    def validate_kinetic_matrix(
        self,
        basis_states: npt.ArrayLike,
        kinetic_matrix: object,
        *,
        tolerance: float = 1.0e-12,
    ) -> bool:
        labels = self.labels(basis_states)
        kinetic = scipy_sparse.coo_array(kinetic_matrix)
        active = np.abs(kinetic.data) > float(tolerance)
        rows = kinetic.row[active]
        cols = kinetic.col[active]
        off_diagonal = rows != cols
        if not np.any(off_diagonal):
            return True
        return bool(np.all(labels[rows[off_diagonal]] != labels[cols[off_diagonal]]))

    def tile_physical_charges(
        self,
        model: object,
        tile_basis: SquareQDMRectangularTileTensorBasis,
    ) -> npt.NDArray[np.int8]:
        """Return local physical ``Z2`` charges when the rule is tile-periodic.

        Raises:
            ValueError: If equivalent owned links in translated tiles carry
                different chiral coefficients.  In that case a larger or
                multi-tensor unit cell is required for a native symmetric PEPS.
        """
        lx, ly = _square_dimensions(model)
        if lx % tile_basis.tile_lx or ly % tile_basis.tile_ly:
            raise ValueError("Model dimensions must be multiples of the tile dimensions.")
        lattice = model.lattice
        reference: list[int] | None = None
        for tile_y in range(ly // tile_basis.tile_ly):
            for tile_x in range(lx // tile_basis.tile_lx):
                coefficients = []
                origin_x = tile_x * tile_basis.tile_lx
                origin_y = tile_y * tile_basis.tile_ly
                for relative_x, relative_y, kind in tile_basis.owned_link_keys:
                    site = _site_id(lattice, origin_x + relative_x, origin_y + relative_y)
                    link_id = _outgoing_link_id(lattice, site, kind)
                    coefficients.append(int(self.link_coefficients[link_id]))
                if reference is None:
                    reference = coefficients
                elif coefficients != reference:
                    raise ValueError(
                        "The inferred chiral rule is not periodic under the selected tile; "
                        "use a larger or multi-tensor unit cell."
                    )
        assert reference is not None
        local_coefficients = np.asarray(reference, dtype=np.int8)
        return np.asarray(
            (tile_basis.physical_configurations @ local_coefficients.astype(np.int64)) % 2,
            dtype=np.int8,
        )


def _solve_gf2_linear_system(
    matrix: npt.NDArray[np.int8],
    rhs: npt.NDArray[np.int8],
) -> npt.NDArray[np.int8]:
    """Return one solution of ``matrix @ x = rhs`` over GF(2)."""
    a = np.asarray(matrix, dtype=np.int8).copy() % 2
    b = np.asarray(rhs, dtype=np.int8).reshape(-1, 1).copy() % 2
    if a.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("GF(2) matrix and right-hand side do not align.")
    augmented = np.concatenate((a, b), axis=1)
    n_rows, n_columns = a.shape
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(n_columns):
        candidates = np.flatnonzero(augmented[pivot_row:, column])
        if candidates.size == 0:
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            augmented[[pivot_row, selected]] = augmented[[selected, pivot_row]]
        for row in range(n_rows):
            if row != pivot_row and augmented[row, column]:
                augmented[row] ^= augmented[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == n_rows:
            break
    inconsistent = np.all(augmented[:, :n_columns] == 0, axis=1) & (augmented[:, n_columns] != 0)
    if np.any(inconsistent):
        raise ValueError(
            "No linear occupation-parity representation of the chiral operator exists."
        )
    solution = np.zeros(n_columns, dtype=np.int8)
    for row, column in enumerate(pivot_columns):
        solution[column] = augmented[row, n_columns]
    return solution


def infer_square_qdm_chiral_parity_rule(
    basis_states: npt.ArrayLike,
    kinetic_matrix: object,
    *,
    reference_labels: npt.ArrayLike | None = None,
    reference_state_indices: Sequence[int] = (),
    reference_label: int = 0,
    tolerance: float = 1.0e-12,
) -> SquareQDMChiralParityRule:
    """Infer a linear link-occupation representation of the chiral operator.

    Each distinct kinetic transition contributes the GF(2) equation
    ``a @ (config_i xor config_j) = 1``.  Free coefficients are fixed to zero,
    yielding one deterministic solution.  The global offset is then aligned to
    either ``reference_state_indices`` or supplied graph bipartition labels.
    """
    states = np.asarray(basis_states, dtype=np.int8)
    if states.ndim != 2:
        raise ValueError("basis_states must be two-dimensional.")
    kinetic = scipy_sparse.coo_array(kinetic_matrix)
    active = (np.abs(kinetic.data) > float(tolerance)) & (kinetic.row < kinetic.col)
    deltas = {
        np.packbits(
            np.asarray(states[row] != states[column], dtype=np.uint8)
        ).tobytes(): np.asarray(states[row] != states[column], dtype=np.int8)
        for row, column in zip(kinetic.row[active], kinetic.col[active], strict=False)
    }
    if not deltas:
        raise ValueError("The kinetic graph has no nonzero transitions.")
    equation_matrix = np.stack(tuple(deltas.values()), axis=0)
    coefficients = _solve_gf2_linear_system(
        equation_matrix,
        np.ones(equation_matrix.shape[0], dtype=np.int8),
    )
    raw_labels = np.asarray((states @ coefficients.astype(np.int64)) % 2, dtype=np.int8)
    offset = 0
    if reference_state_indices:
        indices = np.asarray(tuple(int(index) for index in reference_state_indices), dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= states.shape[0]):
            raise IndexError("reference_state_indices contains an invalid basis index.")
        required_offsets = raw_labels[indices] ^ (int(reference_label) % 2)
        if np.unique(required_offsets).size != 1:
            raise ValueError("Reference states do not lie in one inferred chiral sector.")
        offset = int(required_offsets[0])
    elif reference_labels is not None:
        graph_labels = np.asarray(reference_labels, dtype=np.int8).reshape(-1) % 2
        if graph_labels.size != states.shape[0]:
            raise ValueError("reference_labels must align with basis_states.")
        kinetic_csr = scipy_sparse.csr_array(kinetic_matrix)
        active_vertices = np.diff(kinetic_csr.indptr) > 0
        offsets = raw_labels[active_vertices] ^ graph_labels[active_vertices]
        if offsets.size and np.unique(offsets).size == 1:
            offset = int(offsets[0])
    rule = SquareQDMChiralParityRule(
        link_coefficients=coefficients,
        offset=offset,
        n_edge_equations=equation_matrix.shape[0],
        metadata={"n_basis_states": states.shape[0]},
    )
    if not rule.validate_kinetic_matrix(states, kinetic_matrix, tolerance=tolerance):
        raise RuntimeError("The inferred parity rule does not anticommute with the kinetic matrix.")
    return rule


@dataclass(frozen=True, slots=True)
class SquareQDMType1PEPSResidualReport:
    """Separated type-1 cage diagnostics for one finite-cluster PEPS state."""

    norm_before_projection: float
    norm_after_projection: float
    retained_chiral_weight: float
    discarded_chiral_weight: float
    target_chiral_label: int
    kinetic_interference_norm: float
    kinetic_interference_density: float
    potential_mean: float
    potential_variance: float
    potential_variance_density: float
    total_variance: float
    objective: float
    max_interference_residual: float
    n_nonzero_interference_targets: int
    nonzero_projected_amplitudes: int
    hilbert_dimension: int
    target_potential_value: float | None = None
    target_potential_residual: float | None = None

    @property
    def is_nonzero(self) -> bool:
        return self.norm_after_projection > 0.0

    @property
    def satisfies_type1(self) -> bool:
        return self.kinetic_interference_norm <= 1.0e-10 and self.potential_variance <= 1.0e-10


@dataclass(frozen=True, slots=True)
class SquareQDMType1PEPSOptimizationResult:
    """Result of a quimb optimization of the type-1 separated objective."""

    initial_parameters: npt.NDArray[np.float64]
    optimized_parameters: npt.NDArray[np.float64]
    loss_history: tuple[float, ...]
    initial_report: SquareQDMType1PEPSResidualReport
    final_report: SquareQDMType1PEPSResidualReport
    requested_steps: int
    optimizer: str
    autodiff_backend: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_parameters, dtype=np.float64).reshape(-1)
        optimized = np.asarray(self.optimized_parameters, dtype=np.float64).reshape(-1)
        if initial.shape != optimized.shape:
            raise ValueError("initial_parameters and optimized_parameters must align.")
        object.__setattr__(self, "initial_parameters", initial.copy())
        object.__setattr__(self, "optimized_parameters", optimized.copy())
        object.__setattr__(self, "loss_history", tuple(float(value) for value in self.loss_history))
        object.__setattr__(self, "requested_steps", int(self.requested_steps))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def improved(self) -> bool:
        return self.final_report.objective < self.initial_report.objective


@dataclass(frozen=True, slots=True)
class SquareQDMType1PEPSFiniteClusterProblem:
    """Type-1 specialization of the finite-cluster constrained PEPS problem.

    The underlying PEPS generates amplitudes on the complete constrained basis.
    Before any objective is evaluated, the state is projected exactly onto one
    bipartite subset of the kinetic graph.  The loss then contains only the two
    defining type-1 conditions: kinetic destructive interference and uniform
    diagonal potential on the retained support.
    """

    base_problem: SquareQDMPEPSFiniteClusterProblem
    kinetic_matrix: scipy_sparse.csr_array
    potential_values: npt.NDArray[np.float64]
    chiral_labels: npt.NDArray[np.int8]
    target_chiral_label: int
    parity_rule: SquareQDMChiralParityRule | None = None
    potential_weight: float = 1.0
    target_potential_value: float | None = None
    tolerance: float = 1.0e-10
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kinetic = scipy_sparse.csr_array(self.kinetic_matrix, dtype=np.complex128)
        potential = np.asarray(self.potential_values, dtype=np.float64).reshape(-1)
        labels = np.asarray(self.chiral_labels, dtype=np.int8).reshape(-1) % 2
        dimension = self.base_problem.hilbert_dimension
        if kinetic.shape != (dimension, dimension):
            raise ValueError("kinetic_matrix must match the base PEPS problem.")
        if potential.size != dimension or labels.size != dimension:
            raise ValueError("potential_values and chiral_labels must match the basis.")
        if self.potential_weight < 0.0:
            raise ValueError("potential_weight must be non-negative.")
        object.__setattr__(self, "kinetic_matrix", kinetic)
        object.__setattr__(self, "potential_values", potential.copy())
        object.__setattr__(self, "chiral_labels", labels.copy())
        object.__setattr__(self, "target_chiral_label", int(self.target_chiral_label) % 2)
        object.__setattr__(self, "potential_weight", float(self.potential_weight))
        object.__setattr__(self, "tolerance", float(self.tolerance))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def tile_basis(self) -> SquareQDMRectangularTileTensorBasis:
        return self.base_problem.tile_basis

    @property
    def n_plaquettes(self) -> int:
        return int(self.metadata["n_plaquettes"])

    @property
    def chiral_mask(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.chiral_labels == self.target_chiral_label, dtype=np.float64)

    @property
    def target_basis_indices(self) -> npt.NDArray[np.int64]:
        """Indices of configurations in the occupied type-1 chiral subset."""
        return np.asarray(
            np.flatnonzero(self.chiral_labels == self.target_chiral_label),
            dtype=np.int64,
        )

    @property
    def opposite_basis_indices(self) -> npt.NDArray[np.int64]:
        """Indices of configurations on the empty opposite chiral subset."""
        return np.asarray(
            np.flatnonzero(self.chiral_labels != self.target_chiral_label),
            dtype=np.int64,
        )

    @property
    def target_entry_parameter_indices(self) -> npt.NDArray[np.int64]:
        """Compact tensor entries needed only for the occupied chiral block."""
        return np.asarray(
            self.base_problem.entry_parameter_indices[self.target_basis_indices],
            dtype=np.int64,
        )

    @property
    def target_tensor_coordinates(self) -> npt.NDArray[np.int64]:
        """Dense tensor coordinates needed only for the occupied chiral block."""
        return np.asarray(
            self.base_problem.tensor_coordinates[self.target_basis_indices],
            dtype=np.int64,
        )

    @property
    def target_potential_values(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.potential_values[self.target_basis_indices], dtype=np.float64)

    @property
    def kinetic_interference_matrix(self) -> scipy_sparse.csr_array:
        """Rectangular type-1 map ``B: H_target -> H_opposite``.

        This is the block that must annihilate the occupied-subset amplitude
        vector.  Storing and applying this rectangular block avoids forming a
        full projected wavefunction or multiplying by the complete kinetic
        matrix during optimization.
        """
        return scipy_sparse.csr_array(
            self.kinetic_matrix[
                self.opposite_basis_indices[:, None],
                self.target_basis_indices,
            ],
            dtype=np.complex128,
        )

    def native_state_vector(
        self,
        parameters: npt.ArrayLike,
        *,
        normalize: bool = True,
    ) -> npt.NDArray[np.complex128]:
        """Evaluate amplitudes directly in the occupied chiral block.

        Unlike :meth:`projected_state_vector`, this never constructs amplitudes
        on the opposite Fock-space subset and then discards them.  It is the
        finite-cluster counterpart of contracting the native ``Z2``-symmetric
        PEPS in a fixed total-charge sector.
        """
        values = np.asarray(parameters, dtype=np.complex128).reshape(-1)
        if values.size != self.tile_basis.n_entries:
            raise ValueError(f"parameters must have size {self.tile_basis.n_entries}.")
        state = np.prod(values[self.target_entry_parameter_indices], axis=1)
        if normalize:
            norm = float(np.linalg.norm(state))
            if norm == 0.0:
                raise ValueError("The PEPS is zero in the selected chiral sector.")
            state = state / norm
        return np.asarray(state, dtype=np.complex128)

    def projected_state_vector(
        self,
        parameters: npt.ArrayLike,
        *,
        normalize: bool = True,
    ) -> npt.NDArray[np.complex128]:
        raw = self.base_problem.state_vector(parameters, normalize=False)
        state = raw * self.chiral_mask
        if normalize:
            norm = float(np.linalg.norm(state))
            if norm == 0.0:
                raise ValueError("The PEPS has zero weight in the selected chiral sector.")
            state = state / norm
        return np.asarray(state, dtype=np.complex128)

    def diagnose_native(
        self,
        parameters: npt.ArrayLike,
    ) -> SquareQDMType1PEPSResidualReport:
        """Diagnose the native occupied-block state without post-projection."""
        state = self.native_state_vector(parameters, normalize=False)
        state_norm = float(np.linalg.norm(state))
        if state_norm == 0.0:
            return SquareQDMType1PEPSResidualReport(
                norm_before_projection=0.0,
                norm_after_projection=0.0,
                retained_chiral_weight=1.0,
                discarded_chiral_weight=0.0,
                target_chiral_label=self.target_chiral_label,
                kinetic_interference_norm=float("inf"),
                kinetic_interference_density=float("inf"),
                potential_mean=0.0,
                potential_variance=float("inf"),
                potential_variance_density=float("inf"),
                total_variance=float("inf"),
                objective=float("inf"),
                max_interference_residual=float("inf"),
                n_nonzero_interference_targets=0,
                nonzero_projected_amplitudes=0,
                hilbert_dimension=self.base_problem.hilbert_dimension,
                target_potential_value=self.target_potential_value,
                target_potential_residual=None,
            )
        state = state / state_norm
        interference = np.asarray(
            self.kinetic_interference_matrix @ state,
            dtype=np.complex128,
        )
        kinetic_norm_squared = float(np.vdot(interference, interference).real)
        probabilities = np.abs(state) ** 2
        potential = self.target_potential_values
        potential_mean = float(np.sum(probabilities * potential))
        potential_variance = float(np.sum(probabilities * (potential - potential_mean) ** 2))
        objective = (
            kinetic_norm_squared / self.n_plaquettes
            + self.potential_weight * potential_variance / self.n_plaquettes
        )
        target_residual = None
        if self.target_potential_value is not None:
            target_residual = float(
                np.sum(probabilities * (potential - float(self.target_potential_value)) ** 2)
            )
        return SquareQDMType1PEPSResidualReport(
            norm_before_projection=state_norm,
            norm_after_projection=state_norm,
            retained_chiral_weight=1.0,
            discarded_chiral_weight=0.0,
            target_chiral_label=self.target_chiral_label,
            kinetic_interference_norm=kinetic_norm_squared,
            kinetic_interference_density=kinetic_norm_squared / self.n_plaquettes,
            potential_mean=potential_mean,
            potential_variance=potential_variance,
            potential_variance_density=potential_variance / self.n_plaquettes,
            total_variance=kinetic_norm_squared + potential_variance,
            objective=objective,
            max_interference_residual=float(np.max(np.abs(interference), initial=0.0)),
            n_nonzero_interference_targets=int(
                np.count_nonzero(np.abs(interference) > self.tolerance)
            ),
            nonzero_projected_amplitudes=int(np.count_nonzero(np.abs(state) > self.tolerance)),
            hilbert_dimension=self.base_problem.hilbert_dimension,
            target_potential_value=self.target_potential_value,
            target_potential_residual=target_residual,
        )

    def diagnose(self, parameters: npt.ArrayLike) -> SquareQDMType1PEPSResidualReport:
        raw = self.base_problem.state_vector(parameters, normalize=False)
        raw_norm = float(np.linalg.norm(raw))
        projected = raw * self.chiral_mask
        projected_norm = float(np.linalg.norm(projected))
        if raw_norm == 0.0 or projected_norm == 0.0:
            return SquareQDMType1PEPSResidualReport(
                norm_before_projection=raw_norm,
                norm_after_projection=projected_norm,
                retained_chiral_weight=0.0,
                discarded_chiral_weight=1.0,
                target_chiral_label=self.target_chiral_label,
                kinetic_interference_norm=float("inf"),
                kinetic_interference_density=float("inf"),
                potential_mean=0.0,
                potential_variance=float("inf"),
                potential_variance_density=float("inf"),
                total_variance=float("inf"),
                objective=float("inf"),
                max_interference_residual=float("inf"),
                n_nonzero_interference_targets=0,
                nonzero_projected_amplitudes=0,
                hilbert_dimension=self.base_problem.hilbert_dimension,
                target_potential_value=self.target_potential_value,
                target_potential_residual=None,
            )
        retained_weight = float((projected_norm / raw_norm) ** 2)
        state = projected / projected_norm
        kinetic_state = np.asarray(self.kinetic_matrix @ state, dtype=np.complex128)
        kinetic_norm_squared = float(np.vdot(kinetic_state, kinetic_state).real)
        probabilities = np.abs(state) ** 2
        potential_mean = float(np.sum(probabilities * self.potential_values))
        potential_variance = float(
            np.sum(probabilities * (self.potential_values - potential_mean) ** 2)
        )
        h_state = kinetic_state + self.potential_values * state
        energy = complex(np.vdot(state, h_state))
        total_variance = max(
            0.0,
            float(np.vdot(h_state, h_state).real - abs(energy) ** 2),
        )
        objective = (
            kinetic_norm_squared / self.n_plaquettes
            + self.potential_weight * potential_variance / self.n_plaquettes
        )
        target_residual = None
        if self.target_potential_value is not None:
            target_residual = float(
                np.sum(
                    probabilities
                    * (self.potential_values - float(self.target_potential_value)) ** 2
                )
            )
        return SquareQDMType1PEPSResidualReport(
            norm_before_projection=raw_norm,
            norm_after_projection=projected_norm,
            retained_chiral_weight=retained_weight,
            discarded_chiral_weight=max(0.0, 1.0 - retained_weight),
            target_chiral_label=self.target_chiral_label,
            kinetic_interference_norm=kinetic_norm_squared,
            kinetic_interference_density=kinetic_norm_squared / self.n_plaquettes,
            potential_mean=potential_mean,
            potential_variance=potential_variance,
            potential_variance_density=potential_variance / self.n_plaquettes,
            total_variance=total_variance,
            objective=objective,
            max_interference_residual=float(np.max(np.abs(kinetic_state), initial=0.0)),
            n_nonzero_interference_targets=int(
                np.count_nonzero(np.abs(kinetic_state) > self.tolerance)
            ),
            nonzero_projected_amplitudes=int(np.count_nonzero(np.abs(projected) > self.tolerance)),
            hilbert_dimension=self.base_problem.hilbert_dimension,
            target_potential_value=self.target_potential_value,
            target_potential_residual=target_residual,
        )

    def loss(self, parameters: npt.ArrayLike) -> float:
        return float(self.diagnose_native(parameters).objective)

    def loss_and_gradient_autograd(
        self,
        parameters: npt.ArrayLike,
    ) -> tuple[float, npt.NDArray[np.float64]]:
        if not autograd_available():
            raise ImportError("autograd is required; install qlinks with the 'tn' extra.")
        import autograd.numpy as anp  # type: ignore[import-not-found]
        from autograd import value_and_grad  # type: ignore[import-not-found]

        values = np.asarray(parameters, dtype=np.complex128).reshape(-1)
        if values.size != self.tile_basis.n_entries:
            raise ValueError(f"parameters must have size {self.tile_basis.n_entries}.")
        if np.max(np.abs(values.imag), initial=0.0) > self.tolerance:
            raise ValueError("The Autograd prototype currently optimizes real parameters only.")
        parameter_indices = np.asarray(self.target_entry_parameter_indices, dtype=np.int64)
        interference = anp.asarray(self.kinetic_interference_matrix.toarray().real)
        potential = anp.asarray(self.target_potential_values)
        n_plaquettes = float(self.n_plaquettes)
        potential_weight = float(self.potential_weight)

        def objective(compact_parameters: Any) -> Any:
            state = anp.prod(compact_parameters[parameter_indices], axis=1)
            norm_squared = anp.sum(state * state) + 1.0e-30
            kinetic_state = anp.dot(interference, state)
            kinetic_loss = anp.sum(kinetic_state * kinetic_state) / norm_squared
            probabilities = state * state / norm_squared
            potential_mean = anp.sum(probabilities * potential)
            potential_variance = anp.sum(
                probabilities * (potential - potential_mean) * (potential - potential_mean)
            )
            return (kinetic_loss + potential_weight * potential_variance) / n_plaquettes

        loss, gradient = value_and_grad(objective)(np.asarray(values.real, dtype=np.float64))
        return float(loss), np.asarray(gradient, dtype=np.float64)

    def make_quimb_optimizer(
        self,
        parameters: npt.ArrayLike,
        *,
        autodiff_backend: str = "autograd",
        optimizer: str = "L-BFGS-B",
        progbar: bool = True,
        loss_target: float | None = None,
        **backend_options: object,
    ) -> object:
        qtn = _require_quimb()
        values = np.asarray(parameters, dtype=np.complex128).reshape(-1)
        if values.size != self.tile_basis.n_entries:
            raise ValueError(f"parameters must have size {self.tile_basis.n_entries}.")
        if np.max(np.abs(values.imag), initial=0.0) > self.tolerance:
            raise ValueError("The current compact optimizer accepts real parameters only.")
        tensor_shape = self.tile_basis.tensor_shape
        coordinates = self.tile_basis.entry_coordinates
        flat_coordinates = np.ravel_multi_index(
            tuple(coordinates[:, axis] for axis in range(5)), tensor_shape
        )
        entry_index_map = np.zeros(int(np.prod(tensor_shape)), dtype=np.int64)
        structural_mask = np.zeros(int(np.prod(tensor_shape)), dtype=np.float64)
        entry_index_map[flat_coordinates] = np.arange(self.tile_basis.n_entries)
        structural_mask[flat_coordinates] = 1.0
        entry_index_map = entry_index_map.reshape(tensor_shape)
        structural_mask = structural_mask.reshape(tensor_shape)

        def compact_to_tensor(compact_parameters: Any) -> Any:
            return compact_parameters[entry_index_map] * structural_mask

        tensor = qtn.PTensor(
            compact_to_tensor,
            np.asarray(values.real, dtype=np.float64),
            inds=("up", "right", "down", "left", "physical"),
            tags={"UNIT_CELL"},
        )
        network = qtn.TensorNetwork([tensor])
        return qtn.TNOptimizer(
            network,
            _quimb_type1_cluster_loss,
            loss_constants={
                "tensor_coordinates": self.target_tensor_coordinates,
                "kinetic": np.asarray(
                    self.kinetic_interference_matrix.toarray().real,
                    dtype=np.float64,
                ),
                "potential": np.asarray(self.target_potential_values, dtype=np.float64),
                "n_plaquettes": float(self.n_plaquettes),
                "potential_weight": float(self.potential_weight),
            },
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            loss_target=loss_target,
            **backend_options,
        )

    def optimize_with_quimb(
        self,
        parameters: npt.ArrayLike,
        *,
        max_steps: int = 20,
        noise_scale: float = 1.0e-2,
        seed: int | None = 0,
        autodiff_backend: str = "autograd",
        optimizer: str = "L-BFGS-B",
        progbar: bool = False,
        loss_target: float | None = None,
        **backend_options: object,
    ) -> SquareQDMType1PEPSOptimizationResult:
        initial_parameters = self.base_problem.perturb_parameters(
            parameters, scale=noise_scale, seed=seed, normalize=True
        )
        initial_report = self.diagnose_native(initial_parameters)
        optimizer_object = self.make_quimb_optimizer(
            initial_parameters,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            loss_target=loss_target,
            **backend_options,
        )
        optimized_network = optimizer_object.optimize(int(max_steps))
        optimized_parameters = np.asarray(
            optimized_network["UNIT_CELL"].params, dtype=np.float64
        ).reshape(-1)
        return SquareQDMType1PEPSOptimizationResult(
            initial_parameters=initial_parameters,
            optimized_parameters=optimized_parameters,
            loss_history=tuple(float(value) for value in optimizer_object.losses),
            initial_report=initial_report,
            final_report=self.diagnose_native(optimized_parameters),
            requested_steps=int(max_steps),
            optimizer=str(optimizer),
            autodiff_backend=str(autodiff_backend),
            metadata={"noise_scale": float(noise_scale), "seed": seed},
        )


def _quimb_type1_cluster_loss(
    tensor_network: object,
    *,
    tensor_coordinates: Any,
    kinetic: Any,
    potential: Any,
    n_plaquettes: float,
    potential_weight: float,
) -> Any:
    """Autodiff-compatible type-1 kinetic-plus-potential objective."""
    import autoray as ar

    unit_tensor = tensor_network["UNIT_CELL"].data
    n_basis = int(tensor_coordinates.shape[0])
    n_tiles = int(tensor_coordinates.shape[1])
    state = None
    for tile_index in range(n_tiles):
        coordinates = tensor_coordinates[:, tile_index, :]
        selected = unit_tensor[
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            coordinates[:, 3],
            coordinates[:, 4],
        ]
        state = selected if state is None else state * selected
    if state is None:
        state = ar.do("ones", (n_basis,), like=unit_tensor)
    norm_squared = ar.do("sum", state * state) + 1.0e-30
    kinetic_state = kinetic @ state
    kinetic_loss = ar.do("sum", kinetic_state * kinetic_state) / norm_squared
    probabilities = state * state / norm_squared
    potential_mean = ar.do("sum", probabilities * potential)
    potential_delta = potential - potential_mean
    potential_variance = ar.do("sum", probabilities * potential_delta * potential_delta)
    return (kinetic_loss + potential_weight * potential_variance) / n_plaquettes


@dataclass(frozen=True, slots=True)
class SquareQDMType1ClusterValidationRecord:
    """Type-1 diagnostics for one member of a shared-tensor cluster family."""

    label: str
    n_plaquettes: int
    report: SquareQDMType1PEPSResidualReport


@dataclass(frozen=True, slots=True)
class SquareQDMType1ClusterValidationReport:
    """Cross-cluster type-1 diagnostics for one common PEPS tensor."""

    records: tuple[SquareQDMType1ClusterValidationRecord, ...]
    aggregation_power: float = 4.0
    potential_weight: float = 1.0
    cluster_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("At least one cluster validation record is required.")
        if self.aggregation_power < 1.0:
            raise ValueError("aggregation_power must be at least one.")
        if self.potential_weight < 0.0:
            raise ValueError("potential_weight must be non-negative.")
        labels = tuple(record.label for record in self.records)
        if len(set(labels)) != len(labels):
            raise ValueError("Cluster validation labels must be unique.")
        weights = {label: float(self.cluster_weights.get(label, 1.0)) for label in labels}
        if any(weight <= 0.0 for weight in weights.values()):
            raise ValueError("All cluster weights must be positive.")
        object.__setattr__(self, "aggregation_power", float(self.aggregation_power))
        object.__setattr__(self, "potential_weight", float(self.potential_weight))
        object.__setattr__(self, "cluster_weights", weights)

    @property
    def by_label(self) -> dict[str, SquareQDMType1ClusterValidationRecord]:
        return {record.label: record for record in self.records}

    def _aggregate(self, attribute: str) -> float:
        power = self.aggregation_power
        weighted = 0.0
        total_weight = 0.0
        for record in self.records:
            weight = self.cluster_weights[record.label]
            value = float(getattr(record.report, attribute))
            weighted += weight * max(value, 0.0) ** power
            total_weight += weight
        return float((weighted / total_weight) ** (1.0 / power))

    @property
    def kinetic_aggregate(self) -> float:
        return self._aggregate("kinetic_interference_density")

    @property
    def potential_aggregate(self) -> float:
        return self._aggregate("potential_variance_density")

    @property
    def objective(self) -> float:
        return self.kinetic_aggregate + self.potential_weight * self.potential_aggregate

    @property
    def max_kinetic_interference_density(self) -> float:
        return max(record.report.kinetic_interference_density for record in self.records)

    @property
    def max_potential_variance_density(self) -> float:
        return max(record.report.potential_variance_density for record in self.records)

    @property
    def worst_cluster_label(self) -> str:
        return max(
            self.records,
            key=lambda record: (
                record.report.kinetic_interference_density
                + self.potential_weight * record.report.potential_variance_density
            ),
        ).label

    @property
    def satisfies_type1_on_all_clusters(self) -> bool:
        return all(record.report.satisfies_type1 for record in self.records)


def validate_square_qdm_type1_peps_on_clusters(
    parameters: npt.ArrayLike,
    problems: Mapping[str, SquareQDMType1PEPSFiniteClusterProblem],
    *,
    aggregation_power: float = 4.0,
    potential_weight: float = 1.0,
    cluster_weights: Mapping[str, float] | None = None,
) -> SquareQDMType1ClusterValidationReport:
    """Evaluate one shared tensor with native type-1 blocks on many tori."""
    if not problems:
        raise ValueError("At least one type-1 cluster problem is required.")
    records = tuple(
        SquareQDMType1ClusterValidationRecord(
            label=str(label),
            n_plaquettes=problem.n_plaquettes,
            report=problem.diagnose_native(parameters),
        )
        for label, problem in problems.items()
    )
    return SquareQDMType1ClusterValidationReport(
        records=records,
        aggregation_power=aggregation_power,
        potential_weight=potential_weight,
        cluster_weights={} if cluster_weights is None else cluster_weights,
    )


@dataclass(frozen=True, slots=True)
class SquareQDMType1PEPSJointOptimizationResult:
    """Joint optimization result for separated type-1 losses across clusters."""

    initial_parameters: npt.NDArray[np.float64]
    optimized_parameters: npt.NDArray[np.float64]
    loss_history: tuple[float, ...]
    initial_validation: SquareQDMType1ClusterValidationReport
    final_validation: SquareQDMType1ClusterValidationReport
    requested_steps: int
    optimizer: str
    autodiff_backend: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_parameters, dtype=np.float64).reshape(-1)
        optimized = np.asarray(self.optimized_parameters, dtype=np.float64).reshape(-1)
        if initial.shape != optimized.shape:
            raise ValueError("initial_parameters and optimized_parameters must align.")
        object.__setattr__(self, "initial_parameters", initial.copy())
        object.__setattr__(self, "optimized_parameters", optimized.copy())
        object.__setattr__(self, "loss_history", tuple(float(value) for value in self.loss_history))
        object.__setattr__(self, "requested_steps", int(self.requested_steps))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def improved(self) -> bool:
        return self.final_validation.objective < self.initial_validation.objective


@dataclass(frozen=True, slots=True)
class SquareQDMType1PEPSJointClusterProblem:
    """One native type-1 tensor optimized on several finite clusters at once.

    Kinetic interference and potential nonuniformity are aggregated separately
    across clusters and combined only after each component has been converted
    to a smooth ``p``-norm.  This prevents a low-potential-error cluster from
    masking a large kinetic leakage, or vice versa.
    """

    problems: Mapping[str, SquareQDMType1PEPSFiniteClusterProblem]
    aggregation_power: float = 4.0
    potential_weight: float = 1.0
    cluster_weights: Mapping[str, float] = field(default_factory=dict)
    tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        problems = {str(label): problem for label, problem in self.problems.items()}
        if not problems:
            raise ValueError("At least one type-1 cluster problem is required.")
        if self.aggregation_power < 1.0:
            raise ValueError("aggregation_power must be at least one.")
        if self.potential_weight < 0.0:
            raise ValueError("potential_weight must be non-negative.")
        first = next(iter(problems.values()))
        reference_coordinates = first.tile_basis.entry_coordinates
        reference_charges = None
        if first.parity_rule is not None:
            reference_charges = first.parity_rule.tile_physical_charges(
                first.base_problem.model,
                first.tile_basis,
            )
        for label, problem in problems.items():
            if problem.tile_basis.n_entries != first.tile_basis.n_entries or not np.array_equal(
                problem.tile_basis.entry_coordinates,
                reference_coordinates,
            ):
                raise ValueError(
                    f"Cluster {label!r} does not use the same compact PEPS tensor basis."
                )
            if reference_charges is not None and problem.parity_rule is not None:
                charges = problem.parity_rule.tile_physical_charges(
                    problem.base_problem.model,
                    problem.tile_basis,
                )
                if not np.array_equal(charges, reference_charges):
                    raise ValueError(
                        f"Cluster {label!r} uses a different tile-local chiral charge rule."
                    )
        weights = {label: float(self.cluster_weights.get(label, 1.0)) for label in problems}
        if any(weight <= 0.0 for weight in weights.values()):
            raise ValueError("All cluster weights must be positive.")
        object.__setattr__(self, "problems", problems)
        object.__setattr__(self, "aggregation_power", float(self.aggregation_power))
        object.__setattr__(self, "potential_weight", float(self.potential_weight))
        object.__setattr__(self, "cluster_weights", weights)
        object.__setattr__(self, "tolerance", float(self.tolerance))

    @property
    def tile_basis(self) -> SquareQDMRectangularTileTensorBasis:
        return next(iter(self.problems.values())).tile_basis

    def diagnose(self, parameters: npt.ArrayLike) -> SquareQDMType1ClusterValidationReport:
        return validate_square_qdm_type1_peps_on_clusters(
            parameters,
            self.problems,
            aggregation_power=self.aggregation_power,
            potential_weight=self.potential_weight,
            cluster_weights=self.cluster_weights,
        )

    def loss(self, parameters: npt.ArrayLike) -> float:
        return float(self.diagnose(parameters).objective)

    def loss_and_gradient_autograd(
        self,
        parameters: npt.ArrayLike,
    ) -> tuple[float, npt.NDArray[np.float64]]:
        if not autograd_available():
            raise ImportError("autograd is required; install qlinks with the 'tn' extra.")
        import autograd.numpy as anp  # type: ignore[import-not-found]
        from autograd import value_and_grad  # type: ignore[import-not-found]

        values = np.asarray(parameters, dtype=np.complex128).reshape(-1)
        if values.size != self.tile_basis.n_entries:
            raise ValueError(f"parameters must have size {self.tile_basis.n_entries}.")
        if np.max(np.abs(values.imag), initial=0.0) > self.tolerance:
            raise ValueError("The Autograd prototype currently optimizes real parameters only.")
        cluster_data = tuple(
            (
                anp.asarray(problem.target_entry_parameter_indices),
                anp.asarray(problem.kinetic_interference_matrix.toarray().real),
                anp.asarray(problem.target_potential_values),
                float(problem.n_plaquettes),
                float(self.cluster_weights[label]),
            )
            for label, problem in self.problems.items()
        )
        power = float(self.aggregation_power)
        potential_weight = float(self.potential_weight)
        total_weight = float(sum(self.cluster_weights.values()))
        aggregate_epsilon = 1.0e-32

        def objective(compact_parameters: Any) -> Any:
            kinetic_terms = []
            potential_terms = []
            weights = []
            for parameter_indices, interference, potential, n_plaquettes, weight in cluster_data:
                state = anp.prod(compact_parameters[parameter_indices], axis=1)
                norm_squared = anp.sum(state * state) + 1.0e-30
                residual = anp.dot(interference, state)
                kinetic_density = anp.sum(residual * residual) / norm_squared / n_plaquettes
                probabilities = state * state / norm_squared
                potential_mean = anp.sum(probabilities * potential)
                delta = potential - potential_mean
                potential_density = anp.sum(probabilities * delta * delta) / n_plaquettes
                kinetic_terms.append(kinetic_density)
                potential_terms.append(potential_density)
                weights.append(weight)
            weight_array = anp.asarray(weights)
            kinetic_array = anp.stack(kinetic_terms)
            potential_array = anp.stack(potential_terms)
            kinetic_aggregate = (
                anp.sum(weight_array * kinetic_array**power) / total_weight + aggregate_epsilon
            ) ** (1.0 / power) - aggregate_epsilon ** (1.0 / power)
            potential_aggregate = (
                anp.sum(weight_array * potential_array**power) / total_weight + aggregate_epsilon
            ) ** (1.0 / power) - aggregate_epsilon ** (1.0 / power)
            return kinetic_aggregate + potential_weight * potential_aggregate

        loss, gradient = value_and_grad(objective)(np.asarray(values.real, dtype=np.float64))
        return float(loss), np.asarray(gradient, dtype=np.float64)

    def make_quimb_optimizer(
        self,
        parameters: npt.ArrayLike,
        *,
        autodiff_backend: str = "autograd",
        optimizer: str = "L-BFGS-B",
        progbar: bool = True,
        loss_target: float | None = None,
        **backend_options: object,
    ) -> object:
        qtn = _require_quimb()
        values = np.asarray(parameters, dtype=np.complex128).reshape(-1)
        if values.size != self.tile_basis.n_entries:
            raise ValueError(f"parameters must have size {self.tile_basis.n_entries}.")
        if np.max(np.abs(values.imag), initial=0.0) > self.tolerance:
            raise ValueError("The current compact optimizer accepts real parameters only.")
        tensor_shape = self.tile_basis.tensor_shape
        coordinates = self.tile_basis.entry_coordinates
        flat_coordinates = np.ravel_multi_index(
            tuple(coordinates[:, axis] for axis in range(5)), tensor_shape
        )
        entry_index_map = np.zeros(int(np.prod(tensor_shape)), dtype=np.int64)
        structural_mask = np.zeros(int(np.prod(tensor_shape)), dtype=np.float64)
        entry_index_map[flat_coordinates] = np.arange(self.tile_basis.n_entries)
        structural_mask[flat_coordinates] = 1.0
        entry_index_map = entry_index_map.reshape(tensor_shape)
        structural_mask = structural_mask.reshape(tensor_shape)

        def compact_to_tensor(compact_parameters: Any) -> Any:
            return compact_parameters[entry_index_map] * structural_mask

        tensor = qtn.PTensor(
            compact_to_tensor,
            np.asarray(values.real, dtype=np.float64),
            inds=("up", "right", "down", "left", "physical"),
            tags={"UNIT_CELL"},
        )
        network = qtn.TensorNetwork([tensor])
        cluster_data = tuple(
            {
                "tensor_coordinates": problem.target_tensor_coordinates,
                "kinetic": np.asarray(
                    problem.kinetic_interference_matrix.toarray().real,
                    dtype=np.float64,
                ),
                "potential": np.asarray(problem.target_potential_values, dtype=np.float64),
                "n_plaquettes": float(problem.n_plaquettes),
                "weight": float(self.cluster_weights[label]),
            }
            for label, problem in self.problems.items()
        )
        return qtn.TNOptimizer(
            network,
            _quimb_type1_joint_cluster_loss,
            loss_constants={
                "cluster_data": cluster_data,
                "aggregation_power": float(self.aggregation_power),
                "potential_weight": float(self.potential_weight),
            },
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            loss_target=loss_target,
            **backend_options,
        )

    def optimize_with_quimb(
        self,
        parameters: npt.ArrayLike,
        *,
        max_steps: int = 20,
        noise_scale: float = 1.0e-2,
        seed: int | None = 0,
        autodiff_backend: str = "autograd",
        optimizer: str = "L-BFGS-B",
        progbar: bool = False,
        loss_target: float | None = None,
        **backend_options: object,
    ) -> SquareQDMType1PEPSJointOptimizationResult:
        first_problem = next(iter(self.problems.values()))
        initial_parameters = first_problem.base_problem.perturb_parameters(
            parameters,
            scale=noise_scale,
            seed=seed,
            normalize=True,
        )
        initial_validation = self.diagnose(initial_parameters)
        optimizer_object = self.make_quimb_optimizer(
            initial_parameters,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            loss_target=loss_target,
            **backend_options,
        )
        optimized_network = optimizer_object.optimize(int(max_steps))
        optimized_parameters = np.asarray(
            optimized_network["UNIT_CELL"].params,
            dtype=np.float64,
        ).reshape(-1)
        return SquareQDMType1PEPSJointOptimizationResult(
            initial_parameters=initial_parameters,
            optimized_parameters=optimized_parameters,
            loss_history=tuple(float(value) for value in optimizer_object.losses),
            initial_validation=initial_validation,
            final_validation=self.diagnose(optimized_parameters),
            requested_steps=int(max_steps),
            optimizer=str(optimizer),
            autodiff_backend=str(autodiff_backend),
            metadata={
                "noise_scale": float(noise_scale),
                "seed": seed,
                "cluster_labels": tuple(self.problems),
                "aggregation_power": self.aggregation_power,
            },
        )


def _quimb_type1_joint_cluster_loss(
    tensor_network: object,
    *,
    cluster_data: Sequence[Mapping[str, Any]],
    aggregation_power: float,
    potential_weight: float,
) -> Any:
    """Autodiff-compatible separated type-1 objective on many clusters."""
    import autoray as ar

    unit_tensor = tensor_network["UNIT_CELL"].data
    kinetic_sum = 0.0
    potential_sum = 0.0
    total_weight = 0.0
    power = float(aggregation_power)
    aggregate_epsilon = 1.0e-32
    for data in cluster_data:
        coordinates = data["tensor_coordinates"]
        n_basis = int(coordinates.shape[0])
        n_tiles = int(coordinates.shape[1])
        state = None
        for tile_index in range(n_tiles):
            coordinate = coordinates[:, tile_index, :]
            selected = unit_tensor[
                coordinate[:, 0],
                coordinate[:, 1],
                coordinate[:, 2],
                coordinate[:, 3],
                coordinate[:, 4],
            ]
            state = selected if state is None else state * selected
        if state is None:
            state = ar.do("ones", (n_basis,), like=unit_tensor)
        norm_squared = ar.do("sum", state * state) + 1.0e-30
        residual = data["kinetic"] @ state
        kinetic_density = (
            ar.do("sum", residual * residual) / norm_squared / float(data["n_plaquettes"])
        )
        probabilities = state * state / norm_squared
        potential_mean = ar.do("sum", probabilities * data["potential"])
        delta = data["potential"] - potential_mean
        potential_density = ar.do("sum", probabilities * delta * delta) / float(
            data["n_plaquettes"]
        )
        weight = float(data["weight"])
        kinetic_sum = kinetic_sum + weight * kinetic_density**power
        potential_sum = potential_sum + weight * potential_density**power
        total_weight += weight
    kinetic_aggregate = (kinetic_sum / total_weight + aggregate_epsilon) ** (
        1.0 / power
    ) - aggregate_epsilon ** (1.0 / power)
    potential_aggregate = (potential_sum / total_weight + aggregate_epsilon) ** (
        1.0 / power
    ) - aggregate_epsilon ** (1.0 / power)
    return kinetic_aggregate + float(potential_weight) * potential_aggregate


def build_square_qdm_type1_joint_cluster_problem(
    problems: Mapping[str, SquareQDMType1PEPSFiniteClusterProblem],
    *,
    aggregation_power: float = 4.0,
    potential_weight: float = 1.0,
    cluster_weights: Mapping[str, float] | None = None,
    tolerance: float = 1.0e-10,
) -> SquareQDMType1PEPSJointClusterProblem:
    """Build a shared-tensor type-1 objective over several finite clusters."""
    return SquareQDMType1PEPSJointClusterProblem(
        problems=problems,
        aggregation_power=aggregation_power,
        potential_weight=potential_weight,
        cluster_weights={} if cluster_weights is None else cluster_weights,
        tolerance=tolerance,
    )


def build_square_qdm_type1_peps_problem(
    model: object,
    tile_basis: SquareQDMRectangularTileTensorBasis,
    *,
    target_chiral_label: int | None = None,
    reference_parameters: npt.ArrayLike | None = None,
    cage_record: Any | None = None,
    potential_weight: float = 1.0,
    infer_parity_rule: bool = True,
    basis_solver: str = "dfs",
    builder: str = "sparse",
    tolerance: float = 1.0e-10,
) -> SquareQDMType1PEPSFiniteClusterProblem:
    """Build the finite-cluster type-1 PEPS objective.

    ``cage_record`` may be an existing type-1 :class:`CageRecord`; its support
    fixes the occupied chiral subset and records the exact potential value of
    the finite cage.  Otherwise, ``reference_parameters`` can select the sector
    carrying the larger PEPS norm.  If neither is supplied, sector zero is used.
    """
    from qlinks.caging.search import bipartition_labels

    base_problem = build_square_qdm_peps_finite_cluster_problem(
        model,
        tile_basis,
        basis_solver=basis_solver,
        builder=builder,
        tolerance=tolerance,
    )
    build_result = model.build(basis_solver=basis_solver, builder=builder)
    kinetic = scipy_sparse.csr_array(build_result.kinetic, dtype=np.complex128)
    if build_result.potential is None:
        potential_values = np.zeros(base_problem.hilbert_dimension, dtype=np.float64)
    else:
        potential_values = np.asarray(
            scipy_sparse.csr_array(build_result.potential).diagonal().real,
            dtype=np.float64,
        )
    labels = np.asarray(bipartition_labels(kinetic), dtype=np.int8)
    target_potential_value: float | None = None
    reference_indices: tuple[int, ...] = ()
    if cage_record is not None:
        kappa = int(cage_record.kappa)
        if kappa != 0:
            raise ValueError("A type-1 PEPS seed requires a cage record with kappa=0.")
        support = np.asarray(cage_record.support, dtype=np.int64).reshape(-1)
        if support.size == 0 or np.any(support < 0) or np.any(support >= labels.size):
            raise ValueError("The cage record support does not match the finite-cluster basis.")
        support_labels = np.unique(labels[support])
        if support_labels.size != 1:
            raise ValueError("The cage record does not lie in one chiral subset.")
        inferred_label = int(support_labels[0])
        if target_chiral_label is not None and int(target_chiral_label) % 2 != inferred_label:
            raise ValueError("target_chiral_label conflicts with the cage record support.")
        target_chiral_label = inferred_label
        support_potential = potential_values[support]
        if np.max(np.abs(support_potential - support_potential[0]), initial=0.0) > tolerance:
            raise ValueError("The cage record does not have uniform potential on its support.")
        target_potential_value = float(support_potential[0])
        reference_indices = tuple(int(index) for index in support)
    elif target_chiral_label is None and reference_parameters is not None:
        raw = base_problem.state_vector(reference_parameters, normalize=False)
        weights = (
            float(np.sum(np.abs(raw[labels == 0]) ** 2)),
            float(np.sum(np.abs(raw[labels == 1]) ** 2)),
        )
        target_chiral_label = int(weights[1] > weights[0])
    if target_chiral_label is None:
        target_chiral_label = 0

    parity_rule = None
    if infer_parity_rule:
        parity_reference_indices = reference_indices
        if not parity_reference_indices:
            matching_indices = np.flatnonzero(labels == int(target_chiral_label))
            if matching_indices.size == 0:
                raise ValueError("The requested chiral subset is empty.")
            parity_reference_indices = (int(matching_indices[0]),)
        try:
            parity_rule = infer_square_qdm_tile_chiral_parity_rule(
                model,
                base_problem.basis_states,
                kinetic,
                tile_basis,
                reference_labels=labels,
                reference_state_indices=parity_reference_indices,
                reference_label=0,
                tolerance=tolerance,
            )
        except ValueError:
            parity_rule = infer_square_qdm_chiral_parity_rule(
                base_problem.basis_states,
                kinetic,
                reference_labels=labels,
                reference_state_indices=parity_reference_indices,
                reference_label=0,
                tolerance=tolerance,
            )

    return SquareQDMType1PEPSFiniteClusterProblem(
        base_problem=base_problem,
        kinetic_matrix=kinetic,
        potential_values=potential_values,
        chiral_labels=labels,
        target_chiral_label=int(target_chiral_label),
        parity_rule=parity_rule,
        potential_weight=potential_weight,
        target_potential_value=target_potential_value,
        tolerance=tolerance,
        metadata={
            "n_plaquettes": len(model.plaquette_ids()),
            "model_shape": _square_dimensions(model),
            "source_cage_record": cage_record is not None,
        },
    )


def infer_square_qdm_tile_chiral_parity_rule(
    model: object,
    basis_states: npt.ArrayLike,
    kinetic_matrix: object,
    tile_basis: SquareQDMRectangularTileTensorBasis,
    *,
    reference_labels: npt.ArrayLike | None = None,
    reference_state_indices: Sequence[int] = (),
    reference_label: int = 0,
    tolerance: float = 1.0e-12,
) -> SquareQDMChiralParityRule:
    """Infer a chiral parity rule constrained to repeat with the PEPS tile.

    The unknown GF(2) coefficients live only on the tile's owned-link keys.
    Global links whose source coordinates differ by a tile translation share
    one coefficient.  Existence of a solution means the chiral operator can be
    encoded natively with the chosen one-tensor unit cell.
    """
    states = np.asarray(basis_states, dtype=np.int8)
    if states.ndim != 2:
        raise ValueError("basis_states must be two-dimensional.")
    lx, ly = _square_dimensions(model)
    if lx % tile_basis.tile_lx or ly % tile_basis.tile_ly:
        raise ValueError("Model dimensions must be multiples of the tile dimensions.")
    key_to_variable = {tuple(key): index for index, key in enumerate(tile_basis.owned_link_keys)}
    global_to_local = np.empty(states.shape[1], dtype=np.int64)
    lattice = model.lattice
    for link_id, link in enumerate(lattice.links):
        source = lattice.sites[int(link.source)]
        cell_x, cell_y = (int(source.cell[0]), int(source.cell[1]))
        key = (
            cell_x % tile_basis.tile_lx,
            cell_y % tile_basis.tile_ly,
            str(link.kind),
        )
        try:
            global_to_local[link_id] = key_to_variable[key]
        except KeyError as error:
            raise ValueError(
                f"Global link {link_id} has no matching tile-owned key {key}."
            ) from error

    kinetic = scipy_sparse.coo_array(kinetic_matrix)
    active = (np.abs(kinetic.data) > float(tolerance)) & (kinetic.row < kinetic.col)
    equation_rows: dict[bytes, npt.NDArray[np.int8]] = {}
    for row, column in zip(kinetic.row[active], kinetic.col[active], strict=False):
        local_row = np.zeros(tile_basis.owned_link_ids.size, dtype=np.int8)
        for link_id in np.flatnonzero(states[row] != states[column]):
            local_row[global_to_local[int(link_id)]] ^= 1
        equation_rows[local_row.tobytes()] = local_row
    if not equation_rows:
        raise ValueError("The kinetic graph has no nonzero transitions.")
    equation_matrix = np.stack(tuple(equation_rows.values()), axis=0)
    local_coefficients = _solve_gf2_linear_system(
        equation_matrix,
        np.ones(equation_matrix.shape[0], dtype=np.int8),
    )
    global_coefficients = local_coefficients[global_to_local]
    raw_labels = np.asarray(
        (states @ global_coefficients.astype(np.int64)) % 2,
        dtype=np.int8,
    )
    offset = 0
    if reference_state_indices:
        indices = np.asarray(tuple(int(index) for index in reference_state_indices), dtype=np.int64)
        required_offsets = raw_labels[indices] ^ (int(reference_label) % 2)
        if np.unique(required_offsets).size != 1:
            raise ValueError("Reference states do not lie in one tile-periodic chiral sector.")
        offset = int(required_offsets[0])
    elif reference_labels is not None:
        graph_labels = np.asarray(reference_labels, dtype=np.int8).reshape(-1) % 2
        kinetic_csr = scipy_sparse.csr_array(kinetic_matrix)
        active_vertices = np.diff(kinetic_csr.indptr) > 0
        offsets = raw_labels[active_vertices] ^ graph_labels[active_vertices]
        if offsets.size and np.unique(offsets).size == 1:
            offset = int(offsets[0])
    rule = SquareQDMChiralParityRule(
        link_coefficients=global_coefficients,
        offset=offset,
        n_edge_equations=equation_matrix.shape[0],
        metadata={
            "n_basis_states": states.shape[0],
            "tile_shape": tile_basis.tile_shape,
            "tile_periodic": True,
            "local_link_coefficients": tuple(int(value) for value in local_coefficients),
        },
    )
    if not rule.validate_kinetic_matrix(states, kinetic_matrix, tolerance=tolerance):
        raise RuntimeError("The inferred tile-periodic parity rule is invalid.")
    return rule


@dataclass(frozen=True, slots=True)
class SquareQDMChiralPEPSAnsatz:
    """Native ``Z2``-symmetric PEPS for one selected type-1 chiral subset.

    Each original virtual index is augmented by one parity bit.  For every
    locally allowed QDM tensor entry, the four parity bits are summed modulo
    two and constrained to equal the tile physical chiral charge.  On a closed
    torus all virtual charges cancel pairwise, so the PEPS has support only in
    the globally even sector of the inferred chiral operator.  The parity-rule
    offset is chosen by :func:`build_square_qdm_type1_peps_problem` so that this
    even sector is precisely the selected type-1 subset.

    The additional charge configurations do not introduce new variational
    parameters: all eight parity-compatible copies of an allowed structural
    entry share the same compact amplitude.
    """

    tile_basis: SquareQDMRectangularTileTensorBasis
    parameters: npt.NDArray[np.complex128]
    physical_charges: npt.NDArray[np.int8]
    global_charge_sector: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parameters = np.asarray(self.parameters, dtype=np.complex128).reshape(-1)
        charges = np.asarray(self.physical_charges, dtype=np.int8).reshape(-1) % 2
        if parameters.size != self.tile_basis.n_entries:
            raise ValueError(f"parameters must have size {self.tile_basis.n_entries}.")
        if charges.size != self.tile_basis.physical_dimension:
            raise ValueError(
                f"physical_charges must have size {self.tile_basis.physical_dimension}."
            )
        object.__setattr__(self, "parameters", parameters.copy())
        object.__setattr__(self, "physical_charges", charges.copy())
        object.__setattr__(self, "global_charge_sector", int(self.global_charge_sector) % 2)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_type1_problem(
        cls,
        problem: SquareQDMType1PEPSFiniteClusterProblem,
        parameters: npt.ArrayLike,
    ) -> SquareQDMChiralPEPSAnsatz:
        """Build the native chiral tensor associated with a finite problem."""
        if problem.parity_rule is None:
            raise ValueError("The type-1 problem has no inferred chiral parity rule.")
        charges = problem.parity_rule.tile_physical_charges(
            problem.base_problem.model,
            problem.tile_basis,
        )
        return cls(
            tile_basis=problem.tile_basis,
            parameters=np.asarray(parameters, dtype=np.complex128),
            physical_charges=charges,
            global_charge_sector=int(problem.parity_rule.offset),
            metadata={
                "source": "type1_problem",
                "target_graph_chiral_label": problem.target_chiral_label,
                "parity_rule_offset": problem.parity_rule.offset,
            },
        )

    @property
    def n_parameters(self) -> int:
        return int(self.parameters.size)

    @property
    def tensor_shape(self) -> tuple[int, int, int, int, int]:
        base = self.tile_basis.tensor_shape
        return (2 * base[0], 2 * base[1], 2 * base[2], 2 * base[3], base[4])

    @property
    def n_nonzero_tensor_entries(self) -> int:
        return 8 * self.tile_basis.n_entries

    def tensor_data(
        self,
        *,
        charge_shift: int = 0,
    ) -> npt.NDArray[np.complex128]:
        """Return the dense charge-augmented ``urdlp`` tensor.

        ``charge_shift`` inserts a single global ``Z2`` flux.  On a periodic
        connected network, placing this shift on one tensor selects odd total
        physical charge while leaving every variational amplitude unchanged.
        """
        charge_shift = int(charge_shift) % 2
        data = np.zeros(self.tensor_shape, dtype=np.complex128)
        for entry_index, coordinate in enumerate(self.tile_basis.entry_coordinates):
            up, right, down, left, physical = (int(value) for value in coordinate)
            physical_charge = int(self.physical_charges[physical]) ^ charge_shift
            for charge_pattern in range(16):
                charges = tuple((charge_pattern >> axis) & 1 for axis in range(4))
                if (sum(charges) % 2) != physical_charge:
                    continue
                charge_up, charge_right, charge_down, charge_left = charges
                data[
                    2 * up + charge_up,
                    2 * right + charge_right,
                    2 * down + charge_down,
                    2 * left + charge_left,
                    physical,
                ] = self.parameters[entry_index]
        return data

    def native_sector_mask(
        self,
        problem: SquareQDMPEPSFiniteClusterProblem,
    ) -> npt.NDArray[np.bool_]:
        """Return the exact total-charge mask induced by the native PEPS."""
        if problem.tile_basis.n_entries != self.tile_basis.n_entries or not np.array_equal(
            problem.tile_basis.entry_coordinates,
            self.tile_basis.entry_coordinates,
        ):
            raise ValueError("The finite-cluster problem uses a different tile tensor basis.")
        physical_indices = problem.tensor_coordinates[:, :, 4]
        total_charge = np.sum(self.physical_charges[physical_indices], axis=1) % 2
        return np.asarray(total_charge == self.global_charge_sector, dtype=np.bool_)

    def finite_cluster_state_vector(
        self,
        problem: SquareQDMPEPSFiniteClusterProblem,
        *,
        normalize: bool = True,
    ) -> npt.NDArray[np.complex128]:
        """Contract the native charge rule on an exact constrained basis.

        The virtual charge sum can be performed analytically: every compatible
        physical configuration has the same charge multiplicity, so normalized
        amplitudes are the compact PEPS products restricted by the native total
        ``Z2`` sector.  No post-hoc graph bipartition projector is used.
        """
        values = np.asarray(self.parameters, dtype=np.complex128).reshape(-1)
        state = np.prod(values[problem.entry_parameter_indices], axis=1)
        state = state * self.native_sector_mask(problem)
        if normalize:
            norm = float(np.linalg.norm(state))
            if norm == 0.0:
                raise ValueError("The native chiral PEPS is zero on this finite cluster.")
            state = state / norm
        return np.asarray(state, dtype=np.complex128)

    @staticmethod
    def charge_degeneracy(*, n_tiles_x: int, n_tiles_y: int) -> int:
        """Return the constant virtual-charge multiplicity on a periodic torus."""
        if n_tiles_x <= 0 or n_tiles_y <= 0:
            raise ValueError("n_tiles_x and n_tiles_y must be positive.")
        n_tensors = int(n_tiles_x) * int(n_tiles_y)
        return 1 << (n_tensors + 1)

    def to_quimb_tensor_network(
        self,
        *,
        n_tiles_x: int,
        n_tiles_y: int,
        tags: str | Sequence[str] | None = ("QLINKS", "TYPE1", "UNIT_CELL"),
    ) -> object:
        """Build a periodic quimb network with explicit chiral charge bonds."""
        if n_tiles_x <= 0 or n_tiles_y <= 0:
            raise ValueError("n_tiles_x and n_tiles_y must be positive.")
        qtn = _require_quimb()
        if tags is None:
            global_tags: tuple[str, ...] = ()
        elif isinstance(tags, str):
            global_tags = (tags,)
        else:
            global_tags = tuple(str(tag) for tag in tags)
        regular_data = self.tensor_data()
        shifted_data = self.tensor_data(charge_shift=self.global_charge_sector)
        tensors = []
        for tile_y in range(int(n_tiles_y)):
            for tile_x in range(int(n_tiles_x)):
                up = f"chiral_bond_y_{tile_x}_{tile_y}"
                right = f"chiral_bond_x_{tile_x}_{tile_y}"
                down = f"chiral_bond_y_{tile_x}_{(tile_y - 1) % int(n_tiles_y)}"
                left = f"chiral_bond_x_{(tile_x - 1) % int(n_tiles_x)}_{tile_y}"
                physical = f"physical_{tile_x}_{tile_y}"
                tensors.append(
                    qtn.Tensor(
                        data=(
                            shifted_data.copy()
                            if tile_x == 0 and tile_y == 0
                            else regular_data.copy()
                        ),
                        inds=(up, right, down, left, physical),
                        tags=(
                            *global_tags,
                            f"I{tile_y},{tile_x}",
                            f"X{tile_y}",
                            f"Y{tile_x}",
                        ),
                    )
                )
        return qtn.TensorNetwork(tensors, virtual=True)
