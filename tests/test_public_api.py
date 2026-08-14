from __future__ import annotations

import importlib

PUBLIC_MODULES = (
    "qlinks",
    "qlinks.backends",
    "qlinks.basis",
    "qlinks.basis.solvers",
    "qlinks.builders",
    "qlinks.caging",
    "qlinks.caging.analysis",
    "qlinks.caging.local_search",
    "qlinks.caging.stability",
    "qlinks.constraints",
    "qlinks.conventions",
    "qlinks.encoded",
    "qlinks.exceptions",
    "qlinks.io",
    "qlinks.lattice",
    "qlinks.models",
    "qlinks.open_system",
    "qlinks.open_system.diagnostics",
    "qlinks.open_system.constructions",
    "qlinks.operators",
    "qlinks.variables",
    "qlinks.visualizer",
)


EXPECTED_EXPORTS = {
    "qlinks.caging.analysis": {
        "EnvironmentReductionReport",
        "EnvironmentRemovalMechanismLabel",
        "diagnose_cage_environment_reduction",
    },
    "qlinks.caging.local_search": {
        "LocalCageSearcher",
        "LocalQDMCageSearcher",
        "robust_qdm_local_cage_search",
    },
    "qlinks.caging.stability": {
        "CageStabilityDiagnostic",
        "diagnose_cage_stability",
        "diagnose_many_body_topological_localization",
    },
    "qlinks.basis": {
        "VariableOrderStrategy",
        "ValueOrderStrategy",
    },
    "qlinks.open_system.diagnostics": {
        "DarkManifoldDiagnostics",
        "diagnose_dark_manifold",
        "target_manifold_weight_series",
    },
    "qlinks.open_system.constructions": {
        "CageLindbladDesignProblem",
        "CageLindbladDetectorOperators",
        "CageLindbladDesignResult",
        "CageLindbladExportResult",
        "CageLindbladWorkflowReport",
        "build_cage_lindblad_detector_operators",
        "build_cage_lindblad_problem",
        "export_cage_lindblad_design",
    },
    "qlinks.models": {
        "DirectedPlaquetteCoupling",
        "DirectedPlaquetteCouplingLike",
        "peierls_plaquette_coupling",
    },
    "qlinks.visualizer": {
        "GraphBackend",
        "PlaquetteSymbolMode",
    },
}


def test_public_api_exports_are_sorted_and_bound() -> None:
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        public_names = module.__all__

        assert public_names == sorted(public_names), module_name
        assert len(public_names) == len(set(public_names)), module_name

        missing = [name for name in public_names if not hasattr(module, name)]
        assert missing == [], module_name


def test_expected_public_api_symbols_are_exposed() -> None:
    for module_name, expected_names in EXPECTED_EXPORTS.items():
        module = importlib.import_module(module_name)
        public_names = set(module.__all__)

        assert expected_names <= public_names
