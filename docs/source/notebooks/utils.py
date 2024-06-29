import zipfile
from collections import defaultdict
from typing import List

import igraph
import numpy as np

from qlinks.model.quantum_link_model import QuantumLinkModel
from qlinks.symmetry.gauss_law import GaussLaw


def format_custom_index(index):
    return [f"({i}) {idx}" for i, idx in enumerate(index)]


def group_indices_by_value(dictionary):
    index_groups = defaultdict(set)
    for index, value in dictionary.items():
        index_groups[value].add(index)
    return list(index_groups.values())


def setup_model(model_name, lattice_shape, coup_j, coup_rk, momenta=None):
    gauss_law = {
        "qlm": GaussLaw.from_zero_charge_distri,
        "qdm": GaussLaw.from_staggered_charge_distri,
    }[model_name](*lattice_shape, flux_sector=(0, 0))
    basis = gauss_law.solve()
    model = QuantumLinkModel(coup_j, coup_rk, lattice_shape, basis, momenta=momenta)
    return basis, model


def setup_igraph(nx_g, highlight: List[int] = None, highlight_color: List[str] = None):
    ig = igraph.Graph.from_networkx(nx_g)
    ig.vs["label"] = [str(i) for i in range(ig.vcount())]
    color = ["whitesmoke" for _ in range(ig.vcount())]
    if highlight is not None:
        for i, nodes in enumerate(highlight):
            for node in nodes:
                color[node] = highlight_color[i]
        ig.vs["color"] = color
    return ig


def load_from_npz(path, name):
    """
    Argument `mmap_mode` of np.load not work with npz file, use workaround
    Reference: https://github.com/numpy/numpy/issues/5976
    """
    # figure out offset of .npy in .npz
    zf = zipfile.ZipFile(path)
    info = zf.NameToInfo[name + ".npy"]
    assert info.compress_type == 0
    offset = zf.open(name + ".npy")._orig_compress_start

    fp = open(path, "rb")
    fp.seek(offset)
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fp)
    else:
        raise ImportError(f"Unknown version {version}")
    data_offset = fp.tell()  # file position will be left at beginning of data
    return np.memmap(
        path,
        dtype=dtype,
        shape=shape,
        order="F" if fortran_order else "C",
        mode="r",
        offset=data_offset,
    )
