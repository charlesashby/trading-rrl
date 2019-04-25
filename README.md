# Automated FOREX trading using recurrent reinforcement learning

## Download requirements

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

`tf_rrl.py` contains the code for running the recurrent reinforcement learning algorithm using a deep neural network, an LSTM, or a simple RNN. You can also pick the optimizer of your choice (SGD or Adam), set hyper-parameters s.a. transaction costs, learning rate, number of epochs, etc.

`tradingrrl.py` contains the code for running the layered system i.e. the RRL algorithm using a simple RNN trained using gradient descent with the risk management and the dynamic optimization layers.

`utils` contains multiple files mostly to clean and download the dataset.
