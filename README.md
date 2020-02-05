# edge-popup

**Unofficial** reproduction of the paper [What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/abs/1911.13299).

Status: in development.

## Results

| Size (px) | Arch | Epochs | Commit | Accuracy | # Runs | edge-popup |
|--|--|--|--|--|--|
|128|`xresnet50`|20|[xxx](xxx)|86.44%|5, mean|yes|
|128|`xresnet50`|20|[xxx](xxx)|**89.53%**|5, mean|no|

## ToDo

- [ ] init
    - [x] signed kaiming constant
    - [ ] kaiming normal
- [ ] layers
    - [x] `Linear`
    - [x] `Conv2d`
    - [ ] `LSTM`
- [ ] tests
    - [ ] initializations / variance

## Contributing

Any contributions are welcome.

Feel free to file an `issue` or send a `pull request`.
