# A Convex Approach to Two-Layer Convolutional Neural Networks

## Overview

In this project, our team investigated whether we could apply convex optimization (mathematical problems with guaranteed optimal solutions) to convolutional neural networks (CNNs - the type of AI models great at recognizing images) for classifying pictures. Instead of using the typical training methods that can get stuck finding "good enough" solutions, we reformulated the entire problem to find the absolute minima every time with convex optimization.


## What We Built

### The Core Magic
- **Convex Optimization**: We transformed the usual messy, non-convex convolutional neural network training into a convex problem.
- **Smart Activation Functions**: Instead of the standard ReLU activation, we used polynomial functions (fancy curves) that behaved similarly but played nicely with convex math
- **No More Local Minima**: Unlike regular neural networks that might get stuck in "pretty good" solutions, our convex approach was guaranteed to find the globally best solution

<br>


### The Math Behind It
We reformulated two-layer CNNs using something called semidefinite programming - essentially a way to solve optimization problems involving matrices with special properties. For binary classification (is this a car or plane?), we needed one type of formulation, but for multi-class problems (what's in this image out of 10 options?), we had to get creative with vector outputs and separate parameter matrices for each class.

### Implementation Details
- **Binary Task**: Used CVXPY (a Python library for convex problems) to classify planes vs cars from CIFAR-2
- **Multi-class Challenge**: Built a custom PyTorch implementation to handle all 10 CIFAR-10 categories
- **Polynomial Activation**: Used σ(u) = 0.09u² + 0.5u + 0.47 - a quadratic curve that mimics ReLU behavior
- **Constraint Wrestling**: Had to enforce tricky mathematical constraints to keep my matrices well-behaved during training

<br>

### Tools Used

- **Python** - My go-to for implementation
- **CVXPY** - Specialized library for convex optimization (made the binary case much easier)
- **PyTorch** - Had to get creative with custom modules for the multi-class problem
- **NumPy** - Essential for all the matrix math
- **CIFAR-10/CIFAR-2** - Standard image datasets for testin

## What We Achieved

### Performance Numbers
- **Binary Classification**: Hit 84.05% test accuracy (compared to 88.90% with traditional methods)
- **Multi-class Classification**: Reached 28.5% test accuracy (vs 36.50% baseline)

<br>


### Key Challenges
The main limitation was computational efficiency - our custom PyTorch implementation lacked the optimizations built into standard deep learning frameworks, making training significantly slower. We also had to make some compromises on mathematical constraints due to computational limitations.



### Moving Forward
While the convex approach didn't outperform standard methods, it remains a valuable task to show the feasibility of global optimization for neural networks and provided insights into the mathematical structure underlying deep learning. The project highlighted both the theoretical promise and practical challenges of applying convex optimization to computer vision tasks.

---

*This was a group project completed as part of CS 231N (Deep Learning for Computer Vision) at Stanford University, working under the guidance of EE PhD student Burak Bartan, whose prior research on convex neural networks provided the theoretical foundation for our implementation.*
