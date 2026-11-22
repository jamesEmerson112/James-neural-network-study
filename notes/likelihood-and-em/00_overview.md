# Likelihood and EM

> See also: [The logarithm](../exponential-and-logarithm/02_the_logarithm.md), [Why the Gaussian has exp](../exponential-and-logarithm/04_why_the_gaussian_has_exp.md), [Softmax](../22_softmax.md), [GANs and VAEs](../33_gan_and_vae.md).

This folder covers how a model is fit to data by maximizing its log-likelihood. It looks at the expectation-maximization (EM) algorithm for models with hidden variables, and at the gradient training that neural networks use instead.

- `01_logs_em_and_gradient_descent_video.mp4` is a Manim explainer, 7 minutes 34 seconds long at 1080p30. It shows why training works with the log of the likelihood, then fits a two-curve Gaussian mixture with EM's E-step and M-step. Next it shows why EM's log-likelihood never decreases, and then fits the same mixture with gradient descent and autograd. It ends with EM's lower bound surviving as the ELBO of variational autoencoders. Its companion note, with every formula, number and source, is [Logs, EM and Gradient Descent](01_logs_em_and_gradient_descent.md).
- `01_logs_em_and_gradient_descent_video.py` renders it. No LaTeX is needed, because every formula is drawn with Pango markup.
