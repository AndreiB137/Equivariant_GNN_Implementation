# Equivariant_GNN_Implementation

## Background
We have seen in the ["GraphNGD"](https://github.com/AndreiB137/Graph-Convolutions-and-NGD-Optimization) repository how we can fully describe the CNN by imposing equivariant transformations on the data space. Similarly, we can ask what transformations to specify on graphs. You could go over all of them, one by one, and build a model with it, but it is more enriching when put together with a real-life example, i.e molecules. We can model the bounding between atoms with edges and nodes. Also, molecules have a geometry that can be seen by imagining this arrangement. If we think about water, or $H_{2}O$, we can picture this structure as a red oxygen "ball" connected to two blue hydrogen "balls". Then, you could ask why do the molecules form a "V" shape? Well, it was an arbitrary choice, but how? It is maybe intuitive to ask as well, why the "lengths" of the edges to the hydrogen atoms are identical? Thinking about the bounds between atoms as springs that can stretch or contract depending on the energy of the system, there is natural to see how a longer or shorter edge could be a link to the concept of energy. Hence, we hope that if the relative distances between atoms is also given as input to a graph neural network, this can infer various thermodinamical quantities. Probably there is no surprise, but only the links in the graph with the additional distance information is enough to produce very close results to the ground truth.

<p align="center">
  <img src="https://github.com/AndreiB137/Equivariant_GNN_Implementation/blob/main/FiguresTables/Screenshot%202024-10-31%20at%2011.18.32.jpeg" width="300" height="300">
</p>
<p align="center">
  Figure 1
</p>

## Implementation

This follows the details in the ["EGNN"](https://arxiv.org/pdf/2102.09844) paper. Since the concepts and model are very intuitive, I will focus on the implementation. Firstly, in comparison to a graph convolution network where the edges weights (or features) were the Laplacian multiplied by the nodes features, here we are talking in a general setting with an MLP (or neural network) instead of the precise description of the Laplacian multiplying the nodes features. Now, the edges features $m_{i,j}$ are just the output of an MLP $\phi(h_{i}^l,h_{j}^l,||x_{i}-x_{j}||^2)$ with input $h_{i}^l$, $h_{j}^l$ and $||x_{i}-x_{j}||^2$. This is a standard notation in pytorch_geometric, where the $i$ labels the node to aggregate information to and $j$ labels the neighbors of $i$. Then, the nodes features $h_{i}^{l+1}$ are the output of an MLP $\psi(h_{i}^l, m_{i}^l)$ with input $h_{i}^l$ and $m_{i}^l = \sum_{j\in(i,j)}{}m_{i,j}^l$. $l$ in the superscript labels the output after this operation has been applied $l$ times. Also, $x$ labels the positions of the atoms in 3D space (vectors with three components).

As long as $\phi$ depends on the relative distance, then this architecture is invariant under rotations, translations and permutation equivariant. The latter is a consequence of the graph prescription, while the first two are a conclusion of our discussion before. It is relevant the "length", but not the individual orientations of atoms or if the molecule is translated in space. 

The implementation starts with a toy dataset, the QM9 which contains roughly 130K graphs representating different configurations of molecules formed out of H,C,O,N,F atoms. Each graph has 19 thermodynamical properties out of which I used only the first 12 to compare with the paper. Then, I normalized these values by the mean and and absolute error across all examples. To perform the layer, I created a custom message class that inheritates MessagePassing, where update() and message() functions need override. In message() I get the edge features by the MLP $\phi$ just described, then updating nodes features by the other MLP $\psi$. The global_add_pool is a useful function in pytorch_geometric to perform addition over all nodes in a graph batch. This works well in combination with the DataLoader which creates a batch tensor (a tensor with every node associated its graph index inside the batch). Why you can't directly add nodes by yourself is because DataLoader creates a big graph of all the graphs in the batch, but it also gives a batch tensor for every batch which can be used to do global_add_pool. 

In the figure below there are the results compared with the paper after 10 epochs, which seems to be the maximum value before the model breaks. This is because my training was done with constant learning rate $10^{-3}$ and no learning scheduler. From the paper looks like it is good to start with a small learning rate of $10^{-5}$, $10^{-6}$ in the first epochs and stop at $10^{-3}$. Also, the paper suggest a weight decay of 10^{-16}, which might indicate the sensitivity of the model to exploding gradients. In the figure, some of the properties are by a relatively small margin close to the results in the paper, although others, especially Homo, Lumo and $\Delta\epsilon$ are very far away, giving a reason why they decided to train separately for those. I will come with updates to this repository after training. The model is quite light, so that it runs with 40s per epoch on M1 pro CPU. The dataset is split in 80/10/10 train, validation and test. Aditionally, the hyperparameters might need a change from the paper since in my example I use a random shuffle of the dataset, instead of the pre-loaded split in the paper.

<p align="center">
<img src="https://github.com/AndreiB137/Equivariant_GNN_Implementation/blob/main/FiguresTables/Screenshot%202024-10-31%20at%2010.59.36.png">
Table 1
</p>

## Acknowledgements

* [EGNN](https://github.com/AndreiB137/Graph-Convolutions-and-NGD-Optimization)

## Citation

If you find this repository useful, please cite the following:

```
@misc{Bodnar2024EGNN,
  author = {Bodnar, Andrei},
  title = {Equivariant_GNN_Implementation},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AndreiB137/Equivariant_GNN_Implementation}},
}
```

### Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/AndreiB137/Equivariant_GNN_Implementation/blob/main/LICENSE)
