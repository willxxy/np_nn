# np_nn

Simple implementation of feed forward and backward passes of a neural network with numpy and pytorch on MNIST dataset.

Inspired by https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb. 

If you are trying to understand what goes on "under the hood" in neural networks, please refer to `mlp_np.py`.

## Performances

### Pytorch

![Accuracy](./accuracy_plot_torch.png)

![Loss](./loss_plot_torch.png)

|Comparions| Test Accuracy |
|-----------|----------|
|[geohot](https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb)| 0.9288    | 
|**Ours**| 0.9633    | 




### Numpy

![Accuracy](./accuracy_plot_np.png)

![Loss](./loss_plot_np.png)


|Comparions| Test Accuracy |
|-----------|----------|
|[geohot](https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb)| 0.9635    | 
|**Ours**| 0.9805    | 

