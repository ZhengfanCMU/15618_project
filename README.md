# TSUC: TNN simulator using CUDA

# Summary
We are going to build a Temporal Neural Network (TNN) simulator on GHC’s CUDA GPUs and run MNIST on it. The model will be based on 18743 (Neuromorphic Computer Architecture & Processor Design) lab assignment 1’s pytorch model. And it will be parameterized in 3 or 4 levels of abstractions: 1. the number of synapses per neuron, 2. the number of neurons per macro-column, 3. the number of macro-columns per layer, 4. possibly the number of layers in the entire TNN.

# Background
Temporal Neural Networks is a biomorphic neural network architecture that uses the difference in the arrival time of the signals to encode the magnitude of the signal. The relative arrival time of a signal (called ‘spike’ in this context) from a synapse is combined with the synapse weight and then contributes to the body potential of the neuron which the synapse belongs to. And when a neuron’s body potential reaches a threshold, the neuron will fire a spike at that time. The neuron’s body potential will reset once a while (for a few clock cycles), and this is called a ‘gamma cycle’.
The output function of a synapse in our model follows the Ramp-No-Leak (RNL) function. It does not immediately add the whole weight to the body potential of the neuron, but instead adds 1 to the body potential every clock cycle and stops after weight cycles.
The weight on a synapse is trained according to the spike-timing dependent plasticity model (STDP). This means that if the synapse receives a spike before the neuron fires a spike in a gamma cycle, the weight is strengthened, otherwise weakened.
Multiple neurons that have their synapses connected to the same set of inputs form a macro-column. In each macro-column, a winner-takes-all operation is performed over all the neurons: once a neuron fires a spike, the macro-column will pass down that spike and inhibit other neurons from firing in that gamma cycle.
Multiple macro-columns form a layer. The TNN consists of multiple layers. In our model, we will start from 3 layers.

# Workload
In terms of memory access pattern, in simulating each macro-column, we need to find out the min firing time among the neurons to perform WTA operation. This certainly has a communication between threads. Hopefully, we can limit those in a single block using shared memory.
And at the layer level, spiking times will be passed between the layers. And since we are not sure how the layers are simulated, we cannot make much assumptions on its memory access characteristics. But we are at least sure that this data should be put in some place where we can have quick access to (and possibly discarded right after use for space).
Moreover, the synapse weights should certainly stay in some fast access memory on the GPU to be easily checked and updated.
In terms of divergent execution, we are certain that WTA will cause a lot of divergence because only the winning neuron in each macro-column passes out its spike, and this seems to be an obvious divergence. We need to think more about the model and start working on it to learn more about where divergence can happen.

# Constraints
Since the size (parameters) of our model should be generalizable, we should consider how the change of size affects the workload mapping. On a CUDA platform, the size of shared memory in each block is limited, and we should consider how to make good use of these. Moreover, there is also the problem that during weight update, different synapses in different neurons may update in different ways, causing a lot of divergence.
Also, since not every clock cycle has an input spike, how to skip the cycles without input spikes is also a question to think about. We don’t want to waste computation time on simulating the clock cycles when nothing happens.

# Resources
We have the GHC machines almost always ready for us to use. This is a great benefit. Also, we have the base-line pytorch model from 18743 to use as a reference. We are also using MNIST, which is a well-studied dataset for testing purposes.
We are not sure if there are past implementations of such simulators in CUDA.

# Goals and Deliverables
In the end, we should have a TNN simulator that allows setting a few parameters and can then be trained over an input dataset and perform inference. It should have no less than 5x speedup over the PyTorch model when running on the same GPUs. Since PyTorch already supports CUDA acceleration, we don’t expect a lot of speedup. The speedup should probably come mainly from fusing different potential kernel launches in the PyTorch model together.
It will be good if we can have a better speedup (like 10x), but we think our model can be more ‘cycle-accurate’ than the PyTorch model. That’s probably the most important part. We can accept that if our model has a lower speedup, but it should not slow down.
And in terms of TNN training time and inference accuracy, it should not be too different from the PyTorch model.

# Platform Choice
We choose to use GHC machines’ CUDA GPUs to accelerate our model.
The main reason is that we think the SPMD model is very suitable for simulating the large number of parallel synapses in the neuron macro-columns: In each macro-column, the neurons and their synapses are performing calculations in sync with the ‘gamma wave’. And we think the way threads within a block executes in lock-steps is intuitively very similar to the way the neurons work in a macro-column. This similarity makes us think using CUDA GPUs to simulate our TNN model is a good choice.
A minor reason is that GHC machines’ CUDA GPUs are platforms easily available to us and we have enough experience programming the model. This could save a lot of burdens in trying to understand a new platform.

# Tentative Schedule
Assuming 5 weeks left:
Week 1: Theoretical designs. Basic single neuron implementation should be supported. It should have a parameterized number of synapses.
Week 2: Macro-column implementation should be supported. It should have a parameterized number of neurons.
Week 3: Layers should be implemented and a basic 3 layer network should be working.
Week 4: Testing and optimization. Making our model more generalizable and also improving the speedup.
Week 5: Wrapping up, demo, and writing report.

