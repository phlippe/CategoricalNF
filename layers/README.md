## Implementation of flows and network layers

This folder summarizes the implementation of:
* Normalizing flow layers (folder [flows](flows/)) including:
   * Logistic Mixture Coupling Layers
   * Activation Normalization
   * Invertible 1x1 Convolutions
* Network architectures (folder [networks](networks/)) such as graph neural network (RGCN, Edge-GNN) and autoregressive layers
* Encoding of categorical variables into continuous latent space (folder [categorical_encoding](categorical_encoding/)), including:
   * Mixture of logistics
   * Linear flows
   * Variational dequantization
