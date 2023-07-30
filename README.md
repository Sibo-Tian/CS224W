## Colab 2

2 tasks.

The 1st one is to perform semi-supervised learning on one graph about node prediction.

The input is the whole graph in each epoch.


The 2nd one is to perform supervised learning on several graphs about graph prediction.

In order to take multiple graphs as input at the same time, torch_geometric has Dataloader class, which implemented the mini-batch method.

This feature is used in training with a large number of graphs.


Review basics for classification:

softmax + log + nll_loss = crossEtropyLoss

nll stands for Negative Logits Likelihood, which punishes the unconfidence about the ground_truth class
