from matplotlib import pyplot as plt

from qlinks.model import QuantumLinkModel


class TestQuantumLinkModel:
    def test_hamiltonian(self):
        model = QuantumLinkModel(2, 2, 1, 1)
        ham = model.hamiltonian
        plt.matshow(ham)
        plt.show()
