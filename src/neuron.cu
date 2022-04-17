#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <string>

#include "CycleTimer.h"
struct GlobalConstants {
    // Spike delay and weight resolution
    int wres;
    int rate_capture;
    int rate_backoff;
    int rate_search;
    int gammaLength;
    int spikeThreshold;
};
__constant__ GlobalConstants cuConstTNNParams;
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
                 (1-WTA, but still retaining the 0s so it can be connected to the next layer)
                 TODO: maybe change to k-WTA
                 y: column ID (rows*cols)
                 concatenate additional output data samples in the y dimension
 */
 //TODO: change int that are used for spike time to char
__global__ void column_kernel(int wres, int* weight, char* spike_time_in, char* spike_time_out, int dataLength) {
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
    
    // TODO implement multiple gamma cycles in 1 kernel launch
    
    // shared within each column, for each synapse of each neuron
    // synapse increment indicator
    // if true, synapse contributes to body potential of neuron this cycle
    __shared__ bool synapseRNL_shared[numSynapsePerNeuron * numNeuronPerColumn];
    // TODO optimize: load weight into shared mem to reduce mem access
    //__shared__ int weight_shared[numSynapsePerNeuron * numNeuronPerColumn];

    // ===========================
    // shared within each column, for each neuron in the column
    // body potential of neuron
    __shared__ int neuronBodyPot_shared[numNeuronPerColumn];
    // output spiketime of neuron, for STDP learning (not needed)
    //__shared__ int spikeTimeOutNoInhibit[numNeuronPerColumn];

    

    // record the earliest spike in the column
    __shared__ int earliestSpikingNeuron_shared;
    __shared__ int earliestSpikingTime_shared;

    // 0 0 1 1 1 0 0 0
    // 0 0 0 1 1 1 1 0
    // 0 1 1 0 0 0 0 0

    // 0 1 3 5 


    for (int dataIdx = 0; dataIdx < dataLength; ++dataLength) {
        const int dataYIdx = dataIdx * numColumns + columnID;

        // same synapseID share same input
        const int inputSynapseIdx = dataYIdx * numSynapsePerNeuron + synapseID;
        // each synapse in each neuron has its own weight, updated across data samples
        const int columnSynapseIdx = neuronID * numSynapsePerNeuron + synapseID;

        // local incStart and incEnd to compare to cycle time
        int incStart = spike_time_in[inputSynapseIdx];
        int incEnd = spike_time_in[inputSynapseIdx] + weight[columnSynapseIdx];
        
        // reset increment indicator
        synapseRNL_shared[columnSynapseIdx] = false;
        // reset earliest time
        if (neuronID == 0 && synapseID == 0) {
            earliestSpikingTime_shared = cuConstTNNParams.gammaLength;
        }

        // neuron corresponds to output wire
        const int outputNeuronIdx = dataYIdx * numNeuronPerColumn + neuronID;
        if (synapseID == 0) {
            // reset body potential for this gama cycle (single dataIdx)
            neuronBodyPot_shared[neuronID] = 0;
            // reset spike time to gamma (no spike)
            //spikeTimeOutNoInhibit[neuronID] = gammaLength;
            spike_time_out[outputNeuronIdx] = cuConstTNNParams.gammaLength;
        }
        __syncthreads();

        // end iteration when body potential reaches threshold before gamma cycle time (spikes)
        // end iteration after cycle time reaches gamma cycle time and no spike
        for (int tickCycles = 0; tickCycles < cuConstTNNParams.gammaLength; ++tickCycles) {
            // for each synapse, check the spike_time_in to see if start spiking
            if (tickCycles >= incStart && tickCycles < incEnd) {
                synapseRNL_shared[columnSynapseIdx] = true;
            }
            __syncthreads();
            // after all RNLs are updated, 1 thread for each neuron updates the body potential
            if (synapseID == 0) {
                // sum the synapseRNL_shareds for each neuron's body potential
                // TODO: maybe change to binary reduction add
                for (int synapseIdx = 0; synapseIdx < numSynapsePerNeuron; ++synapseIdx) {
                    neuronBodyPot_shared[neuronID] += synapseRNL_shared[neuronID * numSynapsePerNeuron + synapseIdx];
                }
                if (neuronBodyPot_shared[neuronID] >= cuConstTNNParams.spikeThreshold) {
                    // record the earliest spike in the column
                    // Only synapse 0 need to do this, but not adding if condition
                    // to avoid a conditional lane mask
                    
                    // NOTE: We assume when multiple threads writes the same address, HW will
                    //       sequentialize writes. This would slow down the program.
                    // NOTE: the assumption is that if multiple neuron spike at the same time, a race doesn't matter
                    //       but that's with the assumption that tickCycles is synced
                    // TODO: doublecheck if possible race condition among neurons affect correctness
                    //       e.g. if neurons are across multiple warps and they are at different tickCycles
                    if (earliestSpikingTime_shared > tickCycles) {
                        earliestSpikingNeuron_shared = neuronID;
                        earliestSpikingTime_shared = tickCycles;
                    }
                }
            }
            __syncthreads();
            // In case neuron reaches firing threshold in this cycle
            if (neuronBodyPot_shared[neuronID] >= cuConstTNNParams.spikeThreshold) {
                // used to test race conditions
                assert(earliestSpikingTime_shared <= tickCycles);
                break;
            }
        }
        // 1-WTA spike time out
        // reduce min for global spike time out
        // thread 0 write earliestSpikingTime_shared to output at earliestSpikingNeuron_shared
        // all the later spikes are inhibited by default
        if (synapseID == 0 && neuronID == earliestSpikingNeuron_shared) {
            spike_time_out[dataYIdx * numNeuronPerColumn + earliestSpikingNeuron_shared] = earliestSpikingTime_shared;
        }
        
        // STDP and update weight
        // TODO: define constant parameters: rate_capture, rate_backoff, rate_search
        // per synapse operation

        // weightOverHalf
        // inLeOut
        // inNotInfty
        // outNotInfty
        // inOutNotInfty
        bool isCausal = spike_time_in[inputSynapseIdx] <= spike_time_out[outputNeuronIdx];
        bool inHasSpike = spike_time_in[inputSynapseIdx] < cuConstTNNParams.gammaLength;
        bool outHasSpike = spike_time_out[outputNeuronIdx] < cuConstTNNParams.gammaLength;
        bool isCapture = inHasSpike && outHasSpike && isCausal;
        bool isBackoff = (inHasSpike && outHasSpike && !isCausal) ||
                         (!inHasSpike && outHasSpike);
        bool isSearch = inHasSpike && !outHasSpike;
        //  w_max = gammaLength - 1
        // since integral comparison, not doing -1 on gammaLength
        bool weightOverHalf = weight[columnSynapseIdx] >= cuConstTNNParams.gammaLength / 2;
        int stabUp = weightOverHalf ? cuConstTNNParams.rate_capture : cuConstTNNParams.rate_capture / 2;
        int stabDown = !weightOverHalf ? cuConstTNNParams.rate_backoff : cuConstTNNParams.rate_backoff / 2;
        if (isCapture) weight += cuConstTNNParams.rate_capture * stabUp;
        else if (isBackoff) weight += cuConstTNNParams.rate_backoff * stabDown;
        else if (isSearch) weight += cuConstTNNParams.rate_search;
    }
}

// unroll and convert MNIST data into spike_time_in array
/**
 * @brief unroll and convert MNIST data into spike_time_in array
 * 
 * @param[in] cuMNISTdata MNIST data file loaded into cuda memory
 * @param[out] spike_time_in input spike time array to TNN layer in cuda memory
 */
__global__ void MNIST_loader_kernel(char * cuMNISTdata, char * spike_time_in) {
    // sampleStart
    // numSamplesToConvert
    // rfsize
    // stride
    // channels/format determined by kernel itself
}

void launch_column(std::string input_file, std::string output_file) {
    // neuron count determined by number of classes at current layer
    int numNeuronPerColumn = 12;
    // synapse count determined by receptive field size
    //int numSynapsePerNeuron = rfsize * rfsize * chanleng;
    // rows and cols is the number of receptive fields
    //column_kernel<<<dim3(rows, cols), dim3(numSynapsePerNeuron, numNeuronPerColumn)>>>();

}