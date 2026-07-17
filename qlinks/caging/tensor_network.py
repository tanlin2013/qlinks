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
