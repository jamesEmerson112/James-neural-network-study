# Activation Functions: From Sigmoid to GeGLU

> See also: [Vanishing gradient and tanh](../13_vanishing_gradient_and_tanh.md), [Softmax](../22_softmax.md).

`01_sigmoid_to_geglu_video.mp4` is a Manim explainer, about five minutes long, that walks the activation functions in the order they were adopted and ends at the gated feed-forward blocks that modern LLMs use. It opens with why a network needs a nonlinearity at all: two linear layers multiply out to one matrix, $W_2(W_1x) = (W_2W_1)x$. It then plots sigmoid and its derivative, shows how a slope of at most $0.25$ per layer makes the gradient vanish, and moves through tanh, ReLU and its dead-neuron problem, Leaky ReLU, GELU and SiLU. The last third builds GLU, SwiGLU and GeGLU as diagrams and works out why SwiGLU's hidden width is $\tfrac{2}{3}\cdot 4d$.

Render it from this folder (Manim Community; no LaTeX needed, every formula is drawn with `Text`):

```
manim --resolution 1920,1080 --frame_rate 30 01_sigmoid_to_geglu_video.py Activations
```

On a machine without Manim installed, `uvx --from manim manim ...` runs the same command.

## The functions

| Function | Formula | History |
|---|---|---|
| Sigmoid | $\sigma(x) = \dfrac{1}{1+e^{-x}}$ | Verhulst's logistic curve from population studies; it was the standard unit when Rumelhart, Hinton and Williams popularised backpropagation in 1986. |
| Tanh | $\tanh(x) = \dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ | LeCun, Bottou, Orr and Müller's "Efficient BackProp" (1998) recommended it over the logistic because its outputs are centred on zero. |
| ReLU | $\max(0, x)$ | Nair and Hinton (2010) used it in restricted Boltzmann machines, and AlexNet (Krizhevsky, Sutskever and Hinton, 2012) made it the default for deep networks. |
| Leaky ReLU | $\max(\alpha x, x)$, $\alpha = 0.01$ | Maas, Hannun and Ng (2013) kept a small negative slope so units below zero still receive gradient. |
| GELU | $x\,\Phi(x)$ | Hendrycks and Gimpel (2016) weighted the input by the Gaussian CDF; BERT, GPT-2 and GPT-3 use it. |
| SiLU / Swish | $x\,\sigma(x)$ | Elfwing, Uchibe and Doya (2017) proposed it for reinforcement learning, and Ramachandran, Zoph and Le (2017) rediscovered it by automated search and named it Swish. |
| GLU | $(Wx) \odot \sigma(Vx)$ | Dauphin, Fan, Auli and Grangier (2017) gated the layers of a convolutional language model. |
| SwiGLU | $W_2\big(\mathrm{SiLU}(W_1x) \odot W_3x\big)$ | Shazeer's "GLU Variants Improve Transformer" (2020) introduced it; PaLM, LLaMA, Mistral, Qwen and DeepSeek use it. |
| GeGLU | $W_2\big(\mathrm{GELU}(W_1x) \odot W_3x\big)$ | It comes from the same Shazeer (2020) paper, and Google's Gemma models use it. |

## Why SwiGLU's hidden width is $\tfrac{2}{3}\cdot 4d$

A standard transformer feed-forward block has two matrices of size $d \times 4d$, so $8d^2$ parameters. SwiGLU has three matrices of size $d \times h$, so $3dh$. Setting $3dh = 8d^2$ gives $h = \tfrac{8d}{3} = \tfrac{2}{3}\cdot 4d$. LLaMA-7B has $d = 4096$, which gives about $10923$, and the model rounds that up to $11008$, a multiple of $256$.
