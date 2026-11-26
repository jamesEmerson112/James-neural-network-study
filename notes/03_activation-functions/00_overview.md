# Activation Functions

> See also: [Vanishing gradient and tanh](../13_vanishing_gradient_and_tanh.md), [Softmax](../22_softmax.md).

This folder holds a series of Manim videos on neural-network activation functions, each with a script that renders it.

- `01_sigmoid_to_geglu_video.mp4` runs from sigmoid through tanh, ReLU, Leaky ReLU, GELU and SiLU to the gated feed-forward blocks GLU, SwiGLU and GeGLU. This page describes it.
- `02_gelu_silu_and_recent_activations_video.mp4`, about five minutes long at 1080p30, looks again at GELU and SiLU through the Swish-$\beta$ family, then covers the activations that released LLMs adopted from 2024 to 2026: squared ReLU, gpt-oss's clamped SwiGLU, xIELU, and a sigmoid gate on the attention output. Its companion note is [GELU, SiLU and Recent LLM Activations](02_gelu_silu_and_recent_activations.md).

## Video 1: from sigmoid to GeGLU

`01_sigmoid_to_geglu_video.mp4` is a Manim explainer, about five and a half minutes long at 1080p30, that walks the activation functions in the order they were adopted and ends at the gated feed-forward blocks that modern LLMs use. It opens with why a network needs a nonlinearity at all: two linear layers multiply out to one matrix, $W_2(W_1x) = (W_2W_1)x$. It then plots sigmoid and its derivative, shows how a slope of at most $0.25$ per layer makes the gradient vanish, and moves through tanh, ReLU and its dead-neuron problem, Leaky ReLU, GELU and SiLU. The last third builds GLU, SwiGLU and GeGLU as diagrams and works out why SwiGLU's hidden width is $\tfrac{2}{3}\cdot 4d$.

Render it from this folder (Manim Community; no LaTeX needed, every formula is drawn with `Text`):

```
manim --resolution 1920,1080 --frame_rate 30 01_sigmoid_to_geglu_video.py Activations
```

On a machine without Manim installed, `uvx --from manim manim ...` runs the same command.

### The functions

| Function | Formula | History |
|---|---|---|
| Sigmoid | $\sigma(x) = \dfrac{1}{1+e^{-x}}$ | Verhulst's logistic curve from population studies; it was the standard unit when Rumelhart, Hinton and Williams popularised backpropagation in 1986. |
| Tanh | $\tanh(x) = \dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ | LeCun, Bottou, Orr and Müller's "Efficient BackProp" (1998) recommended it over the logistic because its outputs are centred on zero. |
| ReLU | $\max(0, x)$ | Fukushima used rectifying units in a 1969 network for visual feature extraction. ReLU took off in deep learning four decades later. Nair and Hinton (2010) showed that rectified linear units improve restricted Boltzmann machines, and they credit Jarrett et al. (2009) with finding that rectifying nonlinearities help convolutional nets. Glorot, Bordes and Bengio (2011) showed that deep rectifier networks train well without unsupervised pre-training. AlexNet (Krizhevsky, Sutskever and Hinton, 2012) then made ReLU the default for deep networks. |
| Leaky ReLU | $\max(\alpha x, x)$, $\alpha = 0.01$ | Maas, Hannun and Ng (2013) kept a small negative slope so units below zero still receive gradient. |
| GELU | $x\,\Phi(x)$ | Hendrycks and Gimpel (2016) weighted the input by the Gaussian CDF; BERT, GPT-2 and GPT-3 use it. |
| SiLU / Swish | $x\,\sigma(x)$ | The GELU paper named it the Sigmoid Linear Unit; Elfwing, Uchibe and Doya (2017) used it for reinforcement learning, and Ramachandran, Zoph and Le (2017) rediscovered it by automated search and named it Swish. |
| GLU | $(Wx) \odot \sigma(Vx)$ | Dauphin, Fan, Auli and Grangier (2017) gated the layers of a convolutional language model. |
| SwiGLU | $W_2\big(\mathrm{SiLU}(W_1x) \odot W_3x\big)$ | Shazeer's "GLU Variants Improve Transformer" (2020) introduced it; PaLM, LLaMA, Mistral, Qwen and DeepSeek use it. |
| GeGLU | $W_2\big(\mathrm{GELU}(W_1x) \odot W_3x\big)$ | It comes from the same Shazeer (2020) paper, and Google's Gemma models use it. |

### Why SwiGLU's hidden width is $\tfrac{2}{3}\cdot 4d$

A standard transformer feed-forward block has two matrices of size $d \times 4d$, so $8d^2$ parameters. SwiGLU has three matrices of size $d \times h$, so $3dh$. Setting $3dh = 8d^2$ gives $h = \tfrac{8d}{3} = \tfrac{2}{3}\cdot 4d$. LLaMA-7B has $d = 4096$, which gives about $10923$, and the model rounds that up to $11008$, a multiple of $256$.

## Sources

These sources back the ReLU row of the table.

- K. Fukushima, "Visual Feature Extraction by a Multilayered Network of Analog Threshold Elements," *IEEE Transactions on Systems Science and Cybernetics* 5(4), 322–333, 1969. [doi:10.1109/TSSC.1969.300225](https://doi.org/10.1109/TSSC.1969.300225). The paper is paywalled. Schmidhuber's history, listed next, credits it with rectified linear units.
- J. Schmidhuber, "Annotated History of Modern AI and Deep Learning," 2022. [people.idsia.ch](https://people.idsia.ch/~juergen/deep-learning-history.html)
- K. Jarrett, K. Kavukcuoglu, M. Ranzato and Y. LeCun, "What is the Best Multi-Stage Architecture for Object Recognition?," ICCV 2009, 2146–2153. [doi:10.1109/ICCV.2009.5459469](https://doi.org/10.1109/ICCV.2009.5459469)
- V. Nair and G. E. Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines," ICML 2010. [PDF](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)
- X. Glorot, A. Bordes and Y. Bengio, "Deep Sparse Rectifier Neural Networks," AISTATS 2011. [PDF](https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)
- A. Krizhevsky, I. Sutskever and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," NeurIPS 2012. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
