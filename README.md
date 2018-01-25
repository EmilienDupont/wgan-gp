# Wasserstein GAN with Gradient penalty

Pytorch implementation of [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) by Gulrajani et al.

## Examples

### MNIST

Parameters used were `lr=1e-4`, `betas=(.9, .99)`, `dim=16`, `latent_dim=100`. Note that the images were resized from (28, 28) to (32, 32).

#### Training (200 epochs)
![mnist_gif](https://github.com/EmilienDupont/wgan-gp/raw/master/gifs/mnist_200_epochs.gif)

#### Samples
![mnist_samples](https://github.com/EmilienDupont/wgan-gp/raw/master/imgs/mnist_samples.png)


### Fashion MNIST

#### Training (200 epochs)
![fashion_mnist_gif](https://github.com/EmilienDupont/wgan-gp/raw/master/gifs/training_200_epochs_fashion_mnist.gif)

#### Samples
![fashion_mnist_samples](https://github.com/EmilienDupont/wgan-gp/raw/master/imgs/fashion_mnist_samples.png)

### LSUN Bedrooms

Gif [work in progress]

Samples [work in progress]

## Usage

Set up a generator and discriminator model

```python
from models import Generator, Discriminator
generator = Generator(img_size=(32, 32, 1), latent_dim=100, dim=16)
discriminator = Discriminator(img_size=(32, 32, 1), dim=16)
```

The generator and discriminator are built to automatically scale with image sizes, so you can easily use images from your own dataset.

Train the generator and discriminator with the WGAN-GP loss

```python
import torch
# Initialize optimizers
G_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(.9, .99))
D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(.9, .99))

# Set up trainer
from training import Trainer
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())

# Train model for 200 epochs
trainer.train(data_loader, epochs=200, save_training_gif=True)
```

This will train the models and generate a gif of the training progress.

Note that WGAN-GPs take a *long* time to converge. Even on MNIST it takes about 50 epochs to start seeing decent results. For more information and a full example on MNIST, check out `main.py`.

## Sources and inspiration

* https://github.com/caogang/wgan-gp
* https://github.com/kuc2477/pytorch-wgan-gp
