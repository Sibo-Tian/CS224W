## Colab 2

2 tasks, based on GCN provided by torch_geometric.

The 1st one is to perform semi-supervised learning on one graph about node prediction.

The input is the whole graph in each epoch.

The 2nd one is to perform supervised learning on several graphs about graph prediction.

In order to take multiple graphs as input at the same time, torch_geometric has Dataloader class, which implemented the mini-batch method.

This feature is used in training with a large number of graphs.

Review basics for classification:

softmax + log + nll_loss = crossEtropyLoss

nll stands for Negative Logits Likelihood, which punishes the unconfidence about the ground_truth class

## Colab 3

Implement GraphSage based on `torch_geometric.nn.conv.MessagePassing`

This class is similar to a GCN layer, the only things need to do is implement functions like `forward`, `message`, `aggregate`.

The attribute `propgate` of MessagePassing takes node_features and edge_idx as input, do the following things:

- Use 	`message` to generate messages of neigorhood nodes
- Use `aggregate` to perform convolution among neighborhood for each central node (i.e. weighted-sum)

Then in a MessagePassing class, the whole convolution operator is implemented by `propgate` and `skip-connection`.

Take aways:

- When designing pipelines, it's convinient to set the input of `train` like `train(args, dataset)`, the model and optimizer (and scheduler) can be initialized in the train function. Return loss / metric value & best_model / checkpoint in the end.

## Colab 5

Implement a heterogeneous GNN. 

Data perspective: feature, lable, type are the most important elements for a node / edge / graph.

1. Sample the graph by edge type, do GraphSage on the subgraph
2. Take the mean or attention-based weighted sum of the embeddings calculated by different subgraph for each node

Network Desigh Architecture

1. HeteroGNNConv(pyg_nn.MessagePassing): Do convolution on a subgraph
2. HeteroGNNWrapperConv(deepsnap.heterp_gnn.HeteroConv): Collect embeddings from different subgraph, and aggregate them to a single embedding
3. HeteroGNN(torch.nn.Module): just stack all modules of GNN (GNN layer, batchNorm1d, relu)

For classification tasks, use F1 score macro(mean of F1 of groups)/micro as criterion.
