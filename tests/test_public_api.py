from __future__ import annotations

import importlib

PUBLIC_MODULES = (
    "qlinks",
    "qlinks.backends",
    "qlinks.basis",
    "qlinks.basis.solvers",
    "qlinks.builders",
    "qlinks.caging",
    "qlinks.constraints",
    "qlinks.conventions",
    "qlinks.encoded",
    "qlinks.exceptions",
    "qlinks.io",
    "qlinks.lattice",
    "qlinks.models",
    "qlinks.open_system",
    "qlinks.operators",
    "qlinks.variables",
    "qlinks.visualizer",
)


EXPECTED_EXPORTS = {
    "qlinks.basis": {
        "VariableOrderStrategy",
        "ValueOrderStrategy",
    },
    "qlinks.caging": {
        "JumpOperatorDesign",
        "MonitorSource",
        "ReducedIZMonitorContent",
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
