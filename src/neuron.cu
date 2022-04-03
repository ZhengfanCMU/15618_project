#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

/**
 * @brief Simulates a synapse
 * 1 layer contains x*y columns (number of rfs after convolution with a kernel). Determines the number of output wire groups.
 * 1 neuron column contains numNeuronPerColumn neurons, output 1 spike on numNeuronPerColumn lines
 * each neuron has numSynapsePerNeuron synapses
 * @param[inout] weight initial synapse weight, and final updated weight array
 * @param[in]    spike_time_in input spike times (numSynapsePerNeuron * numColumns * dataLength)
 *               x: spike time for each synapse
 *               y: column ID
 *               concatenate additional input data samples in the y dimension (additional column IDs)
 * @param[out]   spike_time_out output spike time (numNeuronPerColumn * numColumns * dataLength)
 *               rows*cols columns
                 numNeuronPerColumn spike lines
                 x: spike time for each neuron (connected to synapse of next layer) (after 1-WTA)
                 TODO: maybe change to k-WTA
                 y: column ID (rows*cols)
                 concatenate additional output data samples in the y dimension
 */
__global__ void column_kernel(int* weight, int* spike_time_in, int* spike_time_out) {
    // rows * columns of filtered input image
    const int numColumns = gridDim.x * gridDim.y;
    const int columnID = blockIdx.y * gridDim.x + blockIdx.x;
    
    const int numSynapsePerNeuron = blockDim.x;
    const int numNeuronPerColumn = blockDim.y;
    const int synapseID = threadIdx.x; //!< within a neuron
    const int neuronID = threadIdx.y; //!< within a column
    
    // Let each 2d thread block represent a neuron column:
    // Let each row (same threadIdx.y?) of threads represent the synapses of a neuron
    // Within each row, different threads after updating the rnl function, 1 thread update the body potential?
    // Or use parallel reduce.
    // gamma cycle time, neuron and column dimensions in global constants
    
    // TODO implement STDP
    // TODO implement multiple gamma cycles in 1 kernel launch
    
    //!< increment indicator for each synapse of each neuron
    //!< if true, synapse contributes to body potential this cycle
    __shared__ bool synapseRNL[numSynapsePerNeuron * numNeuronPerColumn];
    // shared body potential for each neuron in a column
    __shared__ int neuronBodyPot[numNeuronPerColumn];

    // shared output spiketime for each neuron in a column
    // for STDP learning
    __shared__ int spikeTimeOutNoInhibit[numNeuronPerColumn];
    // record the earliest spike in the column
    int earliestSpikingNeuron;
    int earliestSpikingTime;

    // 0 0 1 1 1 0 0 0
    // 0 0 0 1 1 1 1 0
    // 0 1 1 0 0 0 0 0

    // 0 1 3 5 


    for (int dataIdx = 0; dataIdx < dataLength; ++dataLength) {
        // local incStart and incEnd to compare to cycle time
        int incStart = spike_time_in[/* index here*/];
        int incEnd = spike_time_in[/* index here*/] + weight[/* index here*/];

        // reset increment indicator
        synapseRNL[synapseID + numSynapsePerNeuron * neuronID] = false;
        // reset earliest time
        earliestSpikingTime = gammaLength;
        
        if (synapseID == 0) {
            // reset body potential for this gama cycle (single dataIdx)
            neuronBodyPot[neuronID] = 0;
            // reset spike time to gamma (no spike)
            spike_time_out[/* index here*/] = gammaLength;
        }
        __syncthreads();
        
        // end iteration when body potential reaches threshold before gamma cycle time (spikes)
        // end iteration after cycle time reaches gamma cycle time and no spike
        for (int tickCycles = 0; tickCycles < gammaLength; ++tickCycles) {
            // for each synapse, check the spike_time_in to see if start spiking
            if (tickCycles >= incStart && tickCycles < incEnd) {
                synapseRNL[synapseID + numSynapsePerNeuron * neuronID] = true;
            }
            else {
                synapseRNL[synapseID + numSynapsePerNeuron * neuronID] = false;
            }
            __syncthreads();
            // after all RNLs are updated, 1 thread for each neuron updates the body potential
            if (synapseID == 0) {
                // sum the synapseRNLs for each neuron's body potential
                for (int synpaseIdx = 0; synpaseIdx < numSynapsePerNeuron; ++synpaseIdx) {
                    neuronBodyPot[neuronID] += synapseRNL[synapseID + numSynapsePerNeuron * neuronID];
                }
            }
            __syncthreads();
            if (neuronBodyPot[neuronID] >= spikeThreshold) {
                // record the earliest spike in the column
                if (earliestSpikingTime > tickCycles) {
                    earliestSpikingNeuron = neuronID;
                    earliestSpikingTime = tickCycles;
                }
                if (synapseID == 0) {
                    
                    spikeTimeOutNoInhibit[columnID + neuronID] = tickCycles;
                }
                break;
            }
        }
        // TODO: WTA spike time out
        // reduce min for global spike time out
        // TODO: thread 0 write earliestSpikingTime to output at earliestSpikingNeuron
        // TODO: thread 0 inhibit all the later spikes
        // remember to include dataidx in y calculation
        
        
        // TODO: STDP and update weight
    }

}

void launch_column() {
    // neuron count determined by number of classes at current layer
    int numNeuronPerColumn = 32;
    // synapse count determined by receptive field size
    int numSynapsePerNeuron = 32;
    column_kernel<<<dim3(rows, cols), dim3(numSynapsePerNeuron, numNeuronPerColumn)>>>();

}