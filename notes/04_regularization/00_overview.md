# Regularization

> See also: [Logs, EM and gradient descent](../02_likelihood-and-em/01_logs_em_and_gradient_descent.md), [Model complexity and error tradeoffs](../17b_model_complexity_and_error_tradeoffs.md), [Activation functions](../03_activation-functions/00_overview.md).

This folder covers how training keeps a model from fitting the noise in its training data. It looks at the L1 and L2 penalties on the weights, at weight decay, and at why weight decay is a different knob from the learning rate. It is step 4 of the numbered path through `notes/`, after the log-likelihood (step 2) and activation functions (step 3).

- `01_l1_l2_regularization_and_weight_decay_video.mp4` is a Manim explainer, 9 minutes 19 seconds long at 1080p30. A synthetic drone dataset shows least squares overfitting. Ridge then shrinks every weight, and the lasso sets the useless weights to exactly zero. The video explains the zeros with one weight and with two, separates weight decay from the learning rate, and shows why Adam with an L2 penalty differs from AdamW. Its companion note, with every formula, number and source, is [L1 and L2 Regularization, and Weight Decay](01_l1_l2_regularization_and_weight_decay.md).
- `01_l1_l2_regularization_and_weight_decay_video.py` renders it. No LaTeX is needed, because every formula is drawn with Pango markup.
