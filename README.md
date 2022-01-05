# MothPruning

## Scientific Overview 
Originally inspired by biological nervous systems, deep neural networks (DNNs) are powerful computational tools for modeling complex systems. DNNs are used in a diversity of domains and have helped solve some of the most intractable problems in physics, biology, and computer science. Despite their prevalence, the use of DNNs as a modeling tool comes with some major downsides. DNNs are highly overparameterized, which often results in them being difficult to generalize and interpret, as well as being incredibly computationally expensive. Unlike DNNs, which are often trained until they reach the highest accuracy possible, biological networks have to balance performance with robustness to a noisy and dynamic environment. Biological neural systems use a variety of mechanisms to promote specialized and efficient pathways capable of performing complex tasks in the presence of noise. One such mechanism, synaptic pruning, plays a significant role in refining task-specific behaviors. Synaptic pruning results in a more sparsely connected network that can still perform complex cognitive and motor tasks. Here, we draw inspiration from biology and use DNNs and the method of neural network pruning to find a sparse computational model for controlling a biological motor task. 

In this work, we use the inertial dynamics model in [[2]](#2) to simulate examples of *M. sexta* hovering flight. These data are used to train a DNN to learn the controllers for hovering. Drawing inspiration from pruning in biological neural systems, we sparsify the network using neural network pruning. Here, we prune weights based simply on their magnitudes, removing those weights closest to zero. Insects must maneuver through high noise environments to accomplish controlled flight. It is often assumed that there is a trade-off between perfect flight control and robustness to noise and that the sensory data may be limited by the signal-to-noise ratio. Thus the network need not train for the most accurate model since in practice noise prevents high-fidelity models from exhibiting their underlying accuracy. Rather, we seek to find the sparsest model capable of performing the task given the noisy environment. We employed two methods for neural network pruning: either through manually setting weights to zero or by utilizing binary masking layers. Furthermore, the DNN is pruned sequentially, meaning groups of weights are removed slowly from the network, with retraining in-between successive prunes, until a target sparsity is reached. Monte Carlo simulations are also used to quantify the statistical distribution of network weights during pruning given random initialization of network weights.

For more information, please see our [paper](https://link-url-here.org) [[1]](#1). 

## Project Description

The deep, fully-connected neural network was constructed with ten input variables and seven output variables. The initial and final state space conditions are the inputs to the network: <img src="https://render.githubusercontent.com/render/math?math=\dot{x}"><sub>*i*</sub>, <img src="https://render.githubusercontent.com/render/math?math=\dot{y}"><sub>*i*</sub>, 
<img src="https://render.githubusercontent.com/render/math?math=\phi"><sub>*i*</sub>, <img src="https://render.githubusercontent.com/render/math?math=\theta"><sub>*i*</sub>, <img src="https://render.githubusercontent.com/render/math?math=\dot{\phi}"><sub>*i*</sub>, <img src="https://render.githubusercontent.com/render/math?math=\dot{\theta}"><sub>*i*</sub>, 
<img src="https://render.githubusercontent.com/render/math?math=x"><sub>*f*</sub>,
<img src="https://render.githubusercontent.com/render/math?math=y"><sub>*f*</sub>, <img src="https://render.githubusercontent.com/render/math?math=\phi"><sub>*f*</sub>, and <img src="https://render.githubusercontent.com/render/math?math=\theta"><sub>*f*</sub>. The network predicts the control variables and the final derivatives of the state space in its output layer: <img src="https://render.githubusercontent.com/render/math?math=F"><sub>*x*</sub>, <img src="https://render.githubusercontent.com/render/math?math=F"><sub>*y*</sub>, <img src="https://render.githubusercontent.com/render/math?math=\tau">, <img src="https://render.githubusercontent.com/render/math?math=\dot{x}"><sub>*f*</sub>, <img src="https://render.githubusercontent.com/render/math?math=\dot{y}"><sub>*f*</sub>, <img src="https://render.githubusercontent.com/render/math?math=\dot{\phi}"><sub>*f*</sub>, and <img src="https://render.githubusercontent.com/render/math?math=\dot{\theta}"><sub>*f*</sub>.

After the fully-connected network is trained to a minimum error, we used the method of neural network pruning to promote sparsity between the network layers. In this work, a target sparsity (percentage of pruned network weights) is specified and those weights are forced to zero. The network is then retrained until a minimum error is reached. This process is repeated until most of the weights have been pruned from the network.

To ensure weights remain pruned during retraining, we implemented the pruning functionality of a TensorFlow built toolkit called the Model Optimization Toolkit \cite{tensorflow}. The toolkit contains functions for pruning deep neural networks. In the Model Optimization Toolkit, pruning is achieved through the use of binary masking layers that are multiplied element-wise to each weight matrix in the network. A four-layer neural network can be mathematically described the following way. 

To be able to train and analyze many neural networks, the training and pruning protocols were parallelized in the Jax framework \cite{jax2018github}. Rather than requiring data to be in the form of tensors (such as in TensorFlow), Jax is capable of performing transformations on NumPy \cite{harris2020array} structures. Jax however does not come with a toolkit for pruning, therefore pruning by way of the binary masking matrices was coded into the training loop.

The training and pruning protocols were developed using Keras with the TensorFlow backend. To scale up training for the statistical analysis of many networks, the training and pruning protocols were parallelized using the Jax framework.

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;\hat{y}&space;=&space;\mathbf{M_4}&space;\circ&space;\sigma_4(\mathbf{A_4}...(\mathbf{M_1}&space;\circ&space;(\sigma_1(\mathbf{A_1}x))))

## Installation

## How to use

### Step 1: Train networks 

### Step 2: Evaluate at prunes

### Step 3: Pre-process networks 

### Step 4: Find sparse networks 

## References
<a id="1">[1]</a> 
Zahn, O., Bustamante, Jr J., Switzer, C., Daniel, T., and Kutz, J. N. (2022). 
Pruning deep neural networks generates a sparse, bio-inspired nonlinear controller for insect flight. 

<a id="2">[2]</a> 
Bustamante, Jr J., Ahmed, M., Deora, T., Fabien, B., and Daniel, T. (2021). 
Abdominal movements in insect flight reshape the role of non-aerodynamic structures for flight maneuverability. 
J. Integrative and Comparative Biology. *In revision.*
