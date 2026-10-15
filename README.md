# Pure Mathematical Deep Learning (From First Principles)

A deep learning model built with **no autograd**. Every forward pass, backward pass, and optimizer step is written by hand in NumPy, using matrix calculus, batch normalization's exact total derivative, and a continuous-time Neural ODE with a Pontryagin adjoint solver.

The whole project lives in one notebook: [`pure_math_deep_learning.ipynb`](pure_math_deep_learning.ipynb).

## What's inside

| Section | Content |
|---|---|
| 0. Math overview | Network definition, operator notation, backward Fréchet adjoints |
| 1. Data | 3-class Archimedes spiral in ℝ² (200 points per class, Gaussian angular noise) |
| 2. Calculus engine | `PureMathDeepNetwork`: affine → BatchNorm → Leaky-ReLU blocks, softmax + cross-entropy head, exact BatchNorm backward pass |
| 3. Gradient check | Centered finite differences against every analytic gradient |
| 4. Training | Adam implemented from its update equations; decision-region plot |
| 5. Neural ODE | RK4 forward integration of `h' = tanh(hW + b)` and a backward Pontryagin adjoint solve |

## Requirements

- Python 3 (tested with 3.11)
- NumPy
- Matplotlib
- Jupyter (to open the notebook)

Install them with:

```bash
pip install numpy matplotlib jupyter
```

## Running it

```bash
jupyter notebook pure_math_deep_learning.ipynb
```

Then choose **Run → Run All Cells**. Random seeds are fixed (`np.random.seed(42)`), so results repeat across runs.

## Expected results

From a full run on NumPy 2.1.3 and Matplotlib 3.9.2:

- **Gradient check** (network `[2, 8, 8, 3]`, 16 samples): relative error is about 1e-10 for `W`, `gamma`, and `beta` in every layer.
- **Training** (network `[2, 64, 64, 3]`, Adam, 300 iterations): final loss ≈ 0.0069, training accuracy 100%.
- **Neural ODE**: integrates 16 starting points over `T = 2` with 40 RK4 steps and produces a flow plot.

### Known caveat: bias gradient check

The check prints a relative error of about **1.0** for the biases `b1` and `b2`. This is not a bug in the gradient. Those biases come before BatchNorm, which subtracts the batch mean and so removes any bias effect. The true gradient is zero. The analytic and numeric gradients are both about 1e-10 of floating-point noise, and the error metric divides by that near-zero norm.

The notebook's text says all biases pass the ≤ 1e-9 threshold, and the printed numbers do not show that for `b1` and `b2`. Either the metric should compare absolute norms when both gradients are near zero, or the text should note the exception.

## Notes on the method

- **BatchNorm backward** uses the exact total derivative through the batch mean and variance, not the common simplified form.
- **Adam** is implemented from its update equations with bias correction. There is no optimizer library.
- **Neural ODE adjoint** integrates the adjoint state backward in time with the same step size as the forward pass, and accumulates the parameter gradients over the trajectory.

## Figures

**Figure 1: Training data.** The 3-class spiral in ℝ², 200 points per class, with Gaussian noise on the angle.

<img width="531" height="431" alt="image" src="https://github.com/user-attachments/assets/e6035939-69cf-4f44-b5fa-589ca25ddc76" />


**Figure 2: Training curve and learned decision regions.** Left: regularized cross-entropy loss over 300 Adam iterations. Right: the partition of ℝ² that the trained network learns, with the training points overlaid.

<img width="1198" height="448" alt="image" src="https://github.com/user-attachments/assets/e4b9f0d1-d9a1-41d4-a670-a709f2f770c0" />

**Figure 3: Neural ODE flow.** Sixteen starting states (red) integrated with RK4 over `T = 2` to their final states (blue), following the vector field `tanh(hW + b)`.

<img width="716" height="452" alt="image" src="https://github.com/user-attachments/assets/10dc9c7b-96f1-44e4-bdff-0283cd6d2bf3" />

## Files

- `pure_math_deep_learning.ipynb`: the notebook (6 code cells, 3 figures)
- `README.md`: this file
- `figures/`: PNG exports of the three notebook figures, used by this README
