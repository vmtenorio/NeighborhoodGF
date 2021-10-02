# NeighborhoodGF

This repository contains the companion code for the paper "A Robust Alternative for Graph Convolutional Neural Networks via Graph Neighborhood Filters", by Victor M. Tenorio, Samuel Rey, Fernando Gama, Santiago Segarra and Antonio G. Marques. Abstract:

    Graph convolutional neural networks (GCNNs) are popular deep learning architectures that, upon replacing regular convolutions with graph filters (GFs), generalize CNNs to irregular domains. However, classical GFs are prone to numerical errors since they consist of high-order polynomials. This problem is aggravated when several filters are applied in cascade, limiting the practical depth of GCNNs. To tackle this issue, we present the neighborhood graph filters (NGFs), a family of GFs that replaces the powers of the graph shift operator with *k*-hop neighborhood adjacency matrices. NGFs help to alleviate the numerical issues of traditional GFs, allow for the design of deeper GCNNs, and enhance the robustness to errors in the topology of the graph. To illustrate the advantage over traditional GFs in practical applications, we use NGFs in the design of deep neighborhood GCNNs to solve graph signal denoising and node classification problems over both synthetic and real-world data.

In the paper, we tackle the problem of the numerical issues that classic Graph Filters (defined as polynomials of the GSO) present. The powers of the adjacency matrix contain the number of paths of a certain length between two nodes of the graph. As we raise the matrix to a higher value, the number of paths grows exponentially, and therefore the values of the matrix could explode. Also, if we have errors in the adjacency matrix, these errors are amplified.

As an alternative, we propose **Neighborhood Graph Filters** (NGF), that are defined from the *k*-hop adjacency matrices. We define these matrices as follows: the entry *i,j* of the *k*-hop adjacency matrix has a value of 1 if and only if the shortest path between nodes *i* and *j* of the graph is of *k* hops. NGFs are created as weighted linear combinations of these matrices.

We also define a new type of Graph Convolutional Neural Network that uses NGFs: **Neighborhood Graph Convolutional Neural Networks** (NGCNN). In each layer, NGCNN perform the following operation:
$$
\mathbf{X}^{(\ell)} = \sigma \big( \mathbf{H}_{\mathcal{N}}^{(\ell)} \mathbf{X}^{(\ell-1)} \mathbf{\Theta}^{(\ell)} \big)
$$

Where $\mathbf{H}_{\mathcal{N}}^{(\ell)}$ is an NGF.

In the paper, we demonstrate, theoretically and via numerical experiments, that these type of filters (and the networks built from them) are 1) More stable as we increase the order of the filters used (i.e. the neighborhood size considered in each layer) and 2) More robust to graph perturbations.

## Structure of the repository

The files used for the experiments presented in the paper are:
* `neigh_gf_src` folder contains the library with all the utilities used in the experiments:
    * `graphutils.py`: contains several helper functions related with graphs: creation of new graphs following a given random graph model, perturbation of the adjacency matrix of a graph, creation of filters, etc.
    * `datasets.py`: classes used to synthetically generate data.
    * `arch.py`: contains the Modules with the architectures defined in the paper and also the SOTA architectures definition.
    * `model.py`: contains helper classes to train and test the models.
    * `layers.py`: contains the modules to implement the different types of layers.
* `denoising-Experiments.ipynb`: jupyter notebook with the code used to execute the denoising experiments described in section V.A of the paper.
* `real_world.py` and `real_world_pertK.py`: experiments with real-world datasets whose results are presented in Section V.B of the paper.
