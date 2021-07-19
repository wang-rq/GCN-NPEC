# GCN-NPEC in PyTorch

PyTorch implementation of **Efficiently Solving the Practical Vehicle Routing Problem: A
Novel Joint Learning Approach (KDD'20)**


## Dependencies

* Python >= 3.6
* PyTorch = 1.5
* tqdm
* scipy
* numpy
* plotly (only for plotting)
* matplotlib (only for plotting)


## Usage

Generate the pickle file containing hyper-parameter values by running the following command.

```
python config.py
```

You would see the pickle file in `Pkl` dir. now you can start training the model.

```
python train.py -p Pkl/***.pkl
```

Plot prediction of the trained model
(in this example, batch size is 128, number of customer nodes is 50)

```
python plot.py -p Weights/***.pt(or ***.h5) -b 128 -n 50
```

## Reference
* https://github.com/Rintarooo/VRP_DRL_MHA
