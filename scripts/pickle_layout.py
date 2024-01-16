import pickle

import igraph
import networkx as nx
from ed import setup_link_model

if __name__ == "__main__":
    coup_j, coup_rk = (1, -0.7)  # dfs 3 mins 33 secs
    basis, model = setup_link_model(
        lattice_shape=(6, 4), n_solution=32810, coup_j=coup_j, coup_rk=coup_rk
    )

    g = nx.from_numpy_array(-model.kinetic_term)
    ig = igraph.Graph.from_networkx(g)
    ig.vs["label"] = [str(i) for i in range(ig.vcount())]
    layout = ig.layout_kamada_kawai()

    with open(f'qlm_6x4_coup_j_{coup_j}_coup_rk_{coup_rk}_kin.pickle', 'wb') as f:
        pickle.dump(layout, f)
