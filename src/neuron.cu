#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

/**
 * @brief Simulates a synapse
 * 1 layer contains x*y columns (number of rfs after convolution with a window). Determines the number of output wire groups.
 * 1 neuron column contains numNeuronPerColumn neurons, output 1 spike on numNeuronPerColumn lines
 * each neuron has numSynapsePerNeuron synapses
 * @param[inout] weight initial synapse weight, and final updated weight array
 * @param[in]    spike_time_in input spike times (numSynapsePerNeuron * numNeuronPerColumn)
 * @param[out]   spike_time_out output spike time (numNeuronPerColumn)
 */
__global__ void column_kernel(int* weight, int* spike_time_in, int* spike_time_out) {
    // Let each 2d thread block represent a neuron column:
    // Let each row (same threadIdx.y?) of threads represent the synapses of a neuron
    // Within each row, different threads after updating the rnl function, 1 thread update the body potential?
    // Or use parallel reduce.
    // gamma cycle time, neuron and column dimensions in global constants
    
    // TODO implement STDP
    // TODO implement multiple gamma cycles in 1 kernel launch
    
    //!< increment indicator for each synapse of each neuron
    //!< if true, synapse contributes to body potential this cycle
    __shared__ bool synapseRNL[blockDim.x * blockDim.y];
    // shared body potential for each neuron in a column
    __shared__ int neuronBodyPot[blockDim.y];

    // 0 0 1 1 1 0 0 0
    // 0 0 0 1 1 1 1 0
    // 0 1 1 0 0 0 0 0

    // 0 1 3 5 

    // local incStart and incEnd to compare to cycle time
    int incStart = spike_time_in[/* index here*/];
    int incEnd = spike_time_in[/* index here*/] + weight[/* index here*/];
    
    synapseRNL[threadIdx.x + blockDim.x*threadIdx.y] = false;
    if (threadIdx.x == 0) {
        neuronBodyPot[threadIdx.y] = 0;
    }
    // TODO sync
    // end iteration when body potential reaches threshold before gamma cycle time (spikes)
    // end iteration after cycle time reaches gamma cycle time and no spike
    for (int tickCycles = 0; tickCycles < gamma_length; ++tickCycles) {
        // TODO for each synapse, check the spike_time_in to see if start spiking
        if (tickCycles >= incStart && tickCycles < incEnd) {
            synapseRNL[threadIdx.x + blockDim.x*threadIdx.y] = true;
        }
        else {
            synapseRNL[threadIdx.x + blockDim.x*threadIdx.y] = false;
        }
        // TODO sync
        // TODO after all RNLs are updated, 1 thread for each neuron updates the body potential
        if (threadIdx.x == 0) {
            // TODO sum the synapseRNLs for each neuronBodyPot[threadIdx.y]
            for (int synpaseIdx = 0; synpaseIdx < blockDim.x; ++synpaseIdx) {
                neuronBodyPot[threadIdx.y] += synapseRNL[threadIdx.x + blockDim.x*threadIdx.y];
            }
        }
        // TODO sync
        if (neuronBodyPot[threadIdx.y] >= spikeThreshold) {
            if (threadIdx.x == 0) {
                spike_time_out[/* index here*/] = tickCycles;
            }
            break;
        }
    }

    // TODO: WTA spike time out
    
    // TODO: STDP and update weight
}

void launch_column() {
    // neuron count determined by number of classes at current layer
    int numNeuronPerColumn = 32;
    // synapse count determined by receptive field size
    int numSynapsePerNeuron = 32;
    column_kernel<<<dim3(rows, cols), dim3(numNeuronPerColumn, numSynapsePerNeuron)>>>();

}