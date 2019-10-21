# Anchor Loss in PyTorch
PyTorch implementation of [Anchor Loss: Modulating loss scale based on prediction difficulty](https://arxiv.org/abs/1909.11155), Serim Ryou, Seong-Gyun Jeong, Pietro Perona, ICCV 2019


## Anchor Loss
![anchorloss](https://github.com/slryou41/slryou41.github.io/blob/master/images/overview.png?raw=true)
This code provides anchor loss on image classification. To train the model with anchor loss, include `anchor_loss.py` and call the `AnchorLoss()` function. 

```python
from anchor_loss import AnchorLoss

gamma = 0.5
slack = 0.05
anchor = 'neg'
warm_up = False

criterion = AnchorLoss(gamma, slack, anchor, warm_up)
```

The default parameter settings are shown above. Details about the parameters are explained in the `anchor_loss.py`.

If you use this code, please cite it:
```

```