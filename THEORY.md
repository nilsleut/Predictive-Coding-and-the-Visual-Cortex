# Why Does Predictive Coding Learn V1-like Representations?

**Nils Leutenegger · 2026**

---

*This post derives, from first principles, why a Predictive Coding network initialized on ResNet features should produce representations that align more strongly with early visual cortex (V1/V2) than feedforward models — and connects this to the empirical finding that PC r0 outperforms ViT-B/16 and CLIP at V1 in the THINGS-fMRI experiment.*

---

## 1. The Empirical Puzzle

In the [Predictive Coding and the Visual Cortex](https://github.com/nilsleut/Predictive-Coding-and-the-Visual-Cortex) project, something unexpected happened. A Predictive Coding (PC) network — trained only on 100 images, using ResNet-50 features as input — outperformed ViT-B/16 and CLIP at predicting V1 activity, as measured by Representational Similarity Analysis (RSA) on 7T fMRI data.

This is surprising. ViT-B/16 and CLIP were trained on hundreds of millions of images. The PC network was not trained on raw images at all. So why does it produce better V1-like representations?

The answer lies in what the PC objective *forces* the network to compute, and why that computation happens to be aligned with what V1 does.

---

## 2. The PC Energy Function

Following Rao & Ballard (1999) and Friston (2005), a hierarchical PC network with layers $r_0, r_1, \ldots, r_L$ minimizes a **free energy** (total prediction error):

$$\mathcal{F} = \sum_{i=0}^{L-1} \|\varepsilon_i\|^2 = \sum_{i=0}^{L-1} \|r_i - W_i r_{i+1}\|^2$$

where $W_i$ is a weight matrix mapping the higher-level representation $r_{i+1}$ down to the level of $r_i$, and $\varepsilon_i = r_i - W_i r_{i+1}$ is the **prediction error** at layer $i$.

Each term in the sum penalizes the failure of the higher layer to predict the lower layer. The network minimizes $\mathcal{F}$ jointly over all representations $r_i$ and weights $W_i$.

In our implementation, the input $r_L$ is fixed to the ResNet layer4 features of the current image. The representations $r_0, \ldots, r_{L-1}$ are updated by gradient descent on $\mathcal{F}$:

$$r_i \leftarrow r_i - \eta_r \frac{\partial \mathcal{F}}{\partial r_i}$$

and the weights are updated by a Hebbian-style rule:

$$W_i \leftarrow W_i + \eta_w \, \varepsilon_i \, r_{i+1}^\top$$

---

## 3. The Equilibrium Condition

At the fixed point of the inference dynamics (when $\frac{\partial \mathcal{F}}{\partial r_i} = 0$ for all $i$), the representations satisfy:

$$\frac{\partial \mathcal{F}}{\partial r_i} = 2\varepsilon_i - 2W_{i-1}^\top \varepsilon_{i-1} = 0$$

which gives the equilibrium condition:

$$\varepsilon_i = W_{i-1}^\top \varepsilon_{i-1}$$

In words: at equilibrium, the prediction error at layer $i$ equals the *backprojected* error from the layer below. This is a fixed-point condition where errors propagate upward through transposed weight matrices — a structure that has been compared to the transpose of backpropagation, but running on *representations*, not on a loss gradient.

For the lowest layer $r_0$, the equilibrium condition simplifies. There is no layer below $r_0$, so the only gradient contribution is from the term $\|r_0 - W_0 r_1\|^2$:

$$\frac{\partial \mathcal{F}}{\partial r_0} = 2(r_0 - W_0 r_1) = 2\varepsilon_0 = 0$$

At equilibrium: $r_0^* = W_0 r_1$.

This is the key result: **the lowest-level representation converges to the best linear prediction of $r_0$ given $r_1$, i.e., the linear projection of the higher-level representation onto the lower-level space.**

---

## 4. What This Means for V1

Now consider what $r_0$ is initialized to in our experiment: ResNet layer1 features. Layer1 of ResNet-50 is a $7 \times 7$ convolution with 64 filters applied to the input image, followed by batch normalization and ReLU. These features capture low-frequency spatial patterns — roughly analogous to simple cell responses in V1.

At equilibrium, $r_0^*$ is not simply equal to the ResNet layer1 features. It is the layer that minimizes the reconstruction error of $r_1$ (layer2 features) via $W_0$, while simultaneously being close to its prior (the initialization).

Formally, after training, $W_0$ learns to map $r_1 \to r_0$ in a way that minimizes total prediction error across the training set. The learned $W_0$ performs a kind of **linear inverse** of the ResNet layer1→layer2 transformation. So $r_0^*$ is the representation that, when projected up by the learned $W_0^{-1}$, best reconstructs $r_1$.

This is a *reconstruction* objective at the lowest layer, not a classification objective. And reconstruction of low-level spatial structure is precisely what V1 computes. V1 neurons act as a bank of linear filters over small spatial patches — they represent images in terms of local edge energy, orientation, and spatial frequency. The PC reconstruction objective at $r_0$ forces exactly this kind of representation.

---

## 5. Why Feedforward Models Don't Have This

Why doesn't ResNet layer1 itself have V1-like representations to the same degree?

ResNet layer1 is trained with a classification loss $\mathcal{L}_{CE}$ that propagates gradients from the final softmax layer down through all 50 layers. The gradient at layer1 is:

$$\frac{\partial \mathcal{L}_{CE}}{\partial r_0} = \frac{\partial \mathcal{L}_{CE}}{\partial r_L} \cdot \prod_{i=1}^{L} \frac{\partial r_i}{\partial r_{i-1}}$$

This gradient is dominated by the classification signal — what is useful for distinguishing 1000 ImageNet categories. The layer1 features are shaped by what the later layers need for classification, not by what is needed to reconstruct the input.

ViT and CLIP face an analogous problem. ViT block3 (the early layer used in our RSA comparison) is trained via backpropagation from a classification or contrastive loss. CLIP's early layers are shaped by the need to align image and text embeddings — a semantic objective, not a reconstruction objective.

The PC network, by contrast, imposes a *local* objective at each layer: minimize the prediction error to the layer below. This local objective is reconstruction-like at the lowest layer, and it is this reconstruction pressure that produces V1-aligned representations.

---

## 6. Connection to Sparse Coding

There is a deeper connection here. Olshausen & Field (1996) showed that requiring a linear generative model to *sparsely* reconstruct natural images produces basis functions that resemble V1 simple cell receptive fields: oriented Gabor-like filters at multiple scales.

The PC energy function is closely related to sparse coding. If we add a sparsity prior $\lambda \|r_i\|_1$ to the free energy:

$$\mathcal{F}_{\text{sparse}} = \sum_i \|r_i - W_i r_{i+1}\|^2 + \lambda \|r_i\|_1$$

the equilibrium representations become sparse codes of the input. This is precisely the formulation studied by Olshausen & Field.

Our implementation does not use explicit sparsity, but the reconstruction pressure in $\mathcal{F}$ still drives $r_0$ toward representations that efficiently encode the local structure of layer1 features — which are, in turn, derived from local spatial patches of the image. The result is a representation geometry that is more aligned with V1 than a classification-driven feedforward representation.

---

## 7. Why Higher PC Layers Are Different

The argument above applies specifically to $r_0$. For $r_L$ (the highest layer, fixed to ResNet layer4 in our experiment), there is no reconstruction pressure — it is the *input* to the PC network, not a free variable. Its representations are exactly ResNet layer4 features, which are classification-driven.

For intermediate layers $r_1, r_2$, the equilibrium condition is:

$$r_i^* = W_i r_{i+1} + W_{i-1}^\top \varepsilon_{i-1}$$

These representations are pulled in two directions: toward the top-down prediction from $r_{i+1}$, and toward the bottom-up error correction from $r_{i-1}$. At equilibrium, intermediate layers represent a compromise between the two.

This explains the **gradient in RSA performance**: as we move from $r_0$ to $r_3$, the reconstruction pressure at each layer decreases, and the representations become increasingly dominated by the ResNet classification features that initialized the network. This is exactly the crossing pattern we observe — $r_0$ aligns with V1, $r_3$ aligns more with LOC/IT (where ResNet features are strong).

---

## 8. Formal Summary

Let me state the core argument compactly.

**Claim:** In a PC network trained to minimize $\mathcal{F} = \sum_i \|r_i - W_i r_{i+1}\|^2$, the lowest-level representation $r_0$ converges (at inference equilibrium) to a reconstruction of the input from higher-level features. This reconstruction objective produces representations that are geometrically similar to V1, which also performs a local linear reconstruction of the retinal input via a bank of oriented filters.

**Formal statement:** At the inference fixed point, $r_0^* = W_0 r_1^*$. After learning, $W_0$ approximates the pseudo-inverse of the ResNet layer1→layer2 mapping. Therefore $r_0^*$ is the pre-image of $r_1^*$ under the ResNet forward pass — a representation in the layer1 feature space that maximally explains $r_1^*$. Since layer1 features are local-patch representations, $r_0^*$ encodes images in terms of local spatial structure. V1 encodes images in terms of local oriented edges. These two representations have similar geometry, which is why $r_0^*$ correlates with V1 in RSA.

**Prediction:** A PC network initialized from features at different abstraction levels would show the same V1 advantage at $r_0$ regardless of the specific backbone, as long as the backbone produces hierarchically organized features. This prediction is testable.

---

## 9. What Remains Open

This analysis has two significant gaps.

First, it assumes that the inference dynamics actually converge to the fixed point. With $T=30$ inference steps and a learning rate of $\eta_r = 0.01$, convergence is not guaranteed, especially for high-dimensional representations. In practice, the free energy oscillates after Epoch 20, suggesting the network never fully converges within the training budget.

Second, the argument explains why $r_0$ should align *better* with V1 than a feedforward model, but not why it aligns as well as it does in absolute terms. The noise ceiling at V1 is 0.519, and our best PC model achieves 0.308 — 59% of the theoretical maximum. Understanding the gap requires a better model of what V1 actually computes, which is an open problem in computational neuroscience.

---

## References

- Rao, R.P.N. & Ballard, D.H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79–87.
- Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B*, 360(1456), 815–836.
- Olshausen, B.A. & Field, D.J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381, 607–609.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11, 127–138.
- Hebart, M.N. et al. (2023). THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior. *eLife*, 12, e82580.
- Kriegeskorte, N., Mur, M. & Bandettini, P. (2008). Representational similarity analysis. *Frontiers in Systems Neuroscience*, 2, 4.
