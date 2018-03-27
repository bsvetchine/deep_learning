# A feed forward neuronal network implementation in python (without the use of external libraries) for learning purposes

-> insert ff image

The feed_forward python file implements a neuronal network (FF class). You can set up the number of inputs and the number of neurons in the hidden layer. You can load a dataset (and targets) into the network and then train the network. Finally, you can draw the network and watch the current accuracy.

## Instantiate a network
```python
ff = FF(nb_features=2, hidden_layer_size=5)
```

## Load a dataset
```python
ff.load_dataset(dataset=dataset, targets=targets)
```

## Train the network
```python
ff.train(epochs=100, learning_rate=0.1)
```

## Draw the network
```python
ff.draw()
```

Have a look a the examples shown in the examples file !