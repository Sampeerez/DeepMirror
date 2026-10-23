"""Voice model training: dataset preparation, feature extraction, index and GAN training.

The training procedure follows RVC v2 (VITS-style GAN fine-tuned from the
official pretrained generator/discriminator); the networks are shared with
``backend.inference`` so a trained voice loads exactly like a downloaded one.
"""
