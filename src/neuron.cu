#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <string>

#include <cassert>

#include "CycleTimer.h"
#include "neuron.h"
#include <cuda_fp16.h>

struct GlobalConstants {
    // Spike delay and weights resolution
    int wres;
    float rate_capture;
    float rate_backoff;
    float rate_search;
    int spikeThreshold;
};
__constant__ GlobalConstants cuConstTNNParams;
// unroll and convert MNIST data into spike_time_in array

/**
 * @brief unroll and convert MNIST data into spike_time_in array, using pos-neg encoding.
 *        Pos: if pixel greater/brigher than threshold, spike at time 0,
 *             otherwise spike at time gammaLength.
 *        Neg: if pixel less/darker than threshold, spike at time 0,
 *             otherwise spike at time gammaLength.
 *        For MNIST, there are n imgs, each img is 28x28, we want to generate
 *        2 channels for each img. Each block is used to load 1 img, and each
 *        thread within a block is used to process 1 input pixel. So each block
 *        needs to have 28x28 threads, each thread generate 2 pixels.
 * @param[in] pnThreshold Threshold to determine if a pixel should be pos or neg.
 * @param[in] cuMNISTdata MNIST data file loaded into cuda memory
 * @param[out] spike_time_in input spike time array to TNN layer in cuda memory
 */
__global__ void MNIST_loader_kernel(const int pnThreshold, uint8_t * cuMNISTdata, uint8_t * spike_time_in) {
    // sampleStart
    // numSamplesToConvert
    // rfsize
    // stride
    // channels/format determined by kernel itself

    const int gammaLength = 1 << cuConstTNNParams.wres;
    const int imgIdx = blockIdx.x;
    const int imgDimX = blockDim.x;
    const int imgDimY = blockDim.y;
    const int pixelX = threadIdx.x;
    const int pixelY = threadIdx.y;
    const int dataIdx = imgDimX * imgDimY * imgIdx + imgDimX * pixelY + pixelX;
    const int posSpikeIdx = dataIdx * 2;
    const int negSpikeIdx = dataIdx * 2 + 1;

    spike_time_in[posSpikeIdx] = cuMNISTdata[dataIdx] < 128 ? 0 : gammaLength;
    spike_time_in[negSpikeIdx] = cuMNISTdata[dataIdx] >= 128 ? 0 : gammaLength;
}


void launch_load_MNIST(int nImgs, int nRows, int nCols, uint8_t* imgData, char* & spike_time_in) {
    // copy raw MNIST to device
    int nImageBytes = nImgs * nRows * nCols; // Each pixel is 1 byte.
    uint8_t * imgData_device = NULL;
    cudaError_t cuerr = cudaMalloc(&imgData_device, nImageBytes);
    assert(cuerr == cudaSuccess);
    cuerr = cudaMemcpy(imgData_device, imgData, nImageBytes, cudaMemcpyHostToDevice);
    assert(cuerr == cudaSuccess);
    int nChan = 2;
    cuerr = cudaMalloc(&spike_time_in, nImageBytes * nChan);
    assert(cuerr == cudaSuccess);
    // launch kernel to convert to spike time
    MNIST_loader_kernel<<<nImgs, dim3(nRows, nCols)>>>(UINT8_MAX/2, imgData_device, (uint8_t *)spike_time_in);
}
void copyLabelToDevice(int nLabels, uint8_t * labelData, char *& labels) {
    cudaError_t cuerr = cudaMalloc(&labels, nLabels);
    assert(cuerr == cudaSuccess);
    cuerr = cudaMemcpy(labels, labelData, nLabels, cudaMemcpyHostToDevice);
    assert(cuerr == cudaSuccess);
}
/**
 * @brief Simulates a synapse
 * 1 layer contains x*y columns (number of rfs after convolution with a kernel). Determines the number of output wire groups.
 * 1 neuron column contains numNeuronPerColumn neurons, output 1 spike on numNeuronPerColumn lines
 * each neuron has numSynapsePerNeuron synapses
 * @param[inout] weights initial synapse weights, and final updated weights array
 * @param[in]    spike_time_in matrix of input spike times
 *              (img_x, img_y: pixel's position in the img)
                (nchannels * img_width) * img_height * dataLength
              __|<-------------img_width/_x = 4-------------------->|
               ^|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|
   img_height/_y|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|
               4|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|
               v|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|
                Concatenate additionial input imgs vertically
 * @param[out]   spike_time_out matrix of output spike times
                (layer_x, layer_y: the column's position in the layer)
                (layer_width * nNeuronPerColumn) * layer_height * dataLength
               __|<------------------layer_width/_x = 4------------->|
                ^|n1n2n3n4n5n6|n1n2n3n4n5n6|n1n2n3n4n5n6|n1n2n3n4n5n6|
  layer_height/_y|n1n2n3n4n5n6|n1n2n3n4n5n6|n1n2n3n4n5n6|n1n2n3n4n5n6|
                4|n1n2n3n4n5n6|n1n2n3n4n5n6|n1n2n3n4n5n6|n1n2n3n4n5n6|
                v|n1n2n3n4n5n6|n1n2n3n4n5n6|n1n2n3n4n5n6|n1n2n3n4n5n6|
                Concatenate additionial output imgs vertically
                output is after 1-WTA
                (but still retaining the non-spiking outputs so it can be connected to the next layer)
                TODO: maybe change to k-WTA
 * @param[in]    dataLength number of images in the input and output
 */
__global__ void column_kernel(float* weights, char* spike_time_in, char* spike_time_out, int dataLength) {
    // TODO add to params
    int rfSize;
    int yBatchSize;
    int xBatchSize;
    int nPrevChan;
    int rfSize;
    int numSynapsePerNeuron = rfSize * rfSize * nPrevChan;

    const int gammaLength = 1 << cuConstTNNParams.wres;
    const int synapsesPerThread = xBatchSize * yBatchSize;
    // rows * columns of output spike times
    const int numColumns = gridDim.x * gridDim.y;
    const int columnID = blockIdx.y * gridDim.x + blockIdx.x; // for output

    const int numNeuronPerColumn = blockDim.x;
    // const int rfSize = blockDim.y; // TODO add to struct
    // const int nChan = blockDim.z / rfSize; // TODO add to struct
    // const int numSynapsePerNeuron = rfSize * rfSize * nChan; // TODO add to struct

    const int neuronID = threadIdx.x; //!< within a column


    // Let each 3d thread block represent a neuron column:
    // Within each neuron, different threads after updating the rnl function for each synapse
    // 1 thread update the body potential or use parallel reduce?


    // ==== Shared within each column, for each synapse of each neuron
    // synapse increment indicator
    // if true, synapse contributes to body potential of neuron this cycle
    //__shared__ bool synapseRNL_shared[numSynapsePerNeuron * numNeuronPerColumn];
    // TODO optimize: load weights into shared mem to reduce mem access
    //__shared__ int weights_shared[numSynapsePerNeuron * numNeuronPerColumn];

    // ===========================
    // shared within each column, for each neuron in the column
    // body potential of neuron
    //__shared__ int neuronBodyPot_shared[numNeuronPerColumn];

    // output spiketime of neuron, for STDP learning (not needed)
    // may need for k-WTA
    //__shared__ int spikeTimeOutNoInhibit[numNeuronPerColumn];

    

    // record the earliest spike in the column
    // might use an array of these for k-WTA
    // __shared__ int earliestSpikingNeuron_shared;
    // __shared__ int earliestSpikingTime_shared;
    // int totalSharedSize = sizeof(int) * numNeuronPerColumn + sizeof(int)*2 + sizeof(bool) * numSynapsePerNeuron * numNeuronPerColumn;
    extern __shared__ uint8_t _AllShared[];
    bool * const& synapseRNL_shared = (bool*) &_AllShared[sizeof(int) * numNeuronPerColumn + sizeof(int)*2];
    int * const& neuronBodyPot_shared = (int*) &_AllShared[2];
    int & earliestSpikingNeuron_shared = *((int*) &_AllShared[0]);
    int & earliestSpikingTime_shared = *((int*) &_AllShared[1]);
    /*
    Per column:
    nNeuron
    nSynapse per neuron: = rfsize*rfsize*nprevchan
        nPrevChan
        rfsize
    Per neuron: sum synapse at each timestep and check firing
    Within column: WTA across neuron firing time
    STDP requires 1 final neuron firing time for weight update per neuron's synapse
    */

    // small rfsize: nNeuron*nSynapse fits within threadsPerBlock

    // large rfsize: only rfsize*rfsize(maybe nprevchan) fits within threadsPerBlock

    // parallelize as much synapse as possible but loop through neuron
    // phase 1 threads: rfsize*rfsize*fraction of nprevchan,
    // Each thread loops through the rest of the nprevchan fraction,
    // neurons parallelized across blocks, write spike time to spike_time_out 
    // (unable to sync across block, so maybe still loop through neurons)
    // all threads are summing into the same body potential variable

    // parallelize neuron in threads:
    // further parallelize portion of synapse that fits in the threads
    // Each iteration sum portion of synapse into body pot
    // but need synchronization between threads that map to the same neuron only to check for firing
    // reduces race when summing body potential since target body potential is divided up among neurons
    // numOfThreads:
    // batchHeight, batchWidth (nNeuron * nPrevChan * nRfRows * nRfCols / (batchHeight, batchWidth) <= 1024)
    // nNeuron * nPrevChan, nRfRows/batchWidth, nRfCols/batchHeight
    
    // nNeuron * nPrevChan, nRfY, nRfXFrac
    // rfRowRange = threadIdx.y to threadIdx.y + perYIdxNumRows
    // rfColRange = threadIdx.z to threadIdx.z + perZIdxNumCols

    // TODO: for stride 1 only, if supporting other strides, 
    // change this calculation/pass in input size as params
    const int inputImgSizeX = gridDim.x + rfSize - 1;
    const int inputImgSizeY = gridDim.y + rfSize - 1;
    const int numInputPixels = inputImgSizeX * inputImgSizeY;

    // nNeuron, nYbatches, nXbatches* nPrevChan
    const int neuronIdx = threadIdx.x;
    const int yBatchIdx = threadIdx.y;
    const int xBatchIdx = threadIdx.z / nPrevChan;
    const int chanIdx = threadIdx.z % nPrevChan;

    const int rfYIdxStart = yBatchIdx * yBatchSize;
    int _rfYIdxEnd = (yBatchIdx + 1) * yBatchSize;
    const int rfYIdxEnd = _rfYIdxEnd < rfSize ? _rfYIdxEnd : rfSize;

    const int rfXIdxStart = xBatchIdx * xBatchSize;
    int _rfXIdxEnd = (xBatchIdx + 1) * xBatchSize;
    const int rfXIdxEnd = _rfXIdxEnd < rfSize ? _rfXIdxEnd : rfSize;

    // 0 0 1 1 1 0 0 0
    // 0 0 0 1 1 1 1 0
    // 0 1 1 0 0 0 0 0

    // 0 1 3 5 

    // TODO Initialize all the weights in batches
    for(int rfYIdx = rfYIdxStart; rfYIdx < rfYIdxEnd; ++rfYIdx) {
        for(int rfXIdx = rfXIdxStart; rfXIdx < rfXIdxEnd; ++rfXIdx) {
            const int rfPixelIdx = rfSize * rfYIdx + rfXIdx;
            // for indexing into neuron local synapse
            const int neuronSynapseIdx = rfPixelIdx * nPrevChan + chanIdx;
            // for indexing into column local synapse
            const int columnSynapseIdx = neuronID * numSynapsePerNeuron + neuronSynapseIdx;
            // for indexing into layer synpase weights
            const int layerSynapseIdx = numNeuronPerColumn * numSynapsePerNeuron * columnID + columnSynapseIdx;
            // half initialization for now
            weights[layerSynapseIdx] = (gammaLength - 1.) / 2;
            // TODO implement a choice of random initialization
        }
    }

    for (int dataIdx = 0; dataIdx < dataLength; ++dataLength) {
        // neuron corresponds to output wire
        const int outputDataPixelIdx = dataIdx * numColumns + columnID;
        const int outputNeuronIdx = outputDataPixelIdx * numNeuronPerColumn + neuronID;
        if (threadIdx.y == 0 && threadIdx.z == 0) {
            // reset body potential for this gama cycle (single dataIdx)
            neuronBodyPot_shared[neuronID] = 0;
            // reset spike time to gamma (no spike)
            //spikeTimeOutNoInhibit[neuronID] = gammaLength;
            spike_time_out[outputNeuronIdx] = gammaLength;
        }
        __syncthreads();

        int incStart[synapsesPerThread];
        int incEnd[synapsesPerThread];
        // Loop through thread local synapses to initialize incStart and incEnd
        for(int rfYIdx = rfYIdxStart; rfYIdx < rfYIdxEnd; ++rfYIdx) {
            const int threadSynapseYIdx = rfYIdx - rfYIdxStart;
            for(int rfXIdx = rfXIdxStart; rfXIdx < rfXIdxEnd; ++rfXIdx) {
                // non-padded convolution
                // blockIdx is the output matrix x/y index
                // image coord: (blockIdx.x - rfsize/2 to blockIdx.x + rfsize/2) + rfsize/2
                //              (blockIdx.y - rfsize/2 to blockIdx.y + rfsize/2) + rfsize/2
                // TODO: consider different stride calculation, currently assuming stride == 1
                const int spikeTimeInputX = blockIdx.x + rfXIdx;
                const int spikeTimeInputY = blockIdx.y + rfYIdx;

                const int inputPixelIdx = spikeTimeInputY * inputImgSizeX + spikeTimeInputX;

                // index into spike_time_in for this synapse
                const int dataPixelIdx = dataIdx * numInputPixels + inputPixelIdx;
                const int inputSpikeIdx = dataPixelIdx * nPrevChan + chanIdx;

                const int rfPixelIdx = rfSize * rfYIdx + rfXIdx;
                // for indexing into neuron local synapse
                const int neuronSynapseIdx = rfPixelIdx * nPrevChan + chanIdx;
                // for indexing into column local synapse
                const int columnSynapseIdx = neuronID * numSynapsePerNeuron + neuronSynapseIdx;
                // for indexing into layer synpase weights
                const int layerSynapseIdx = numNeuronPerColumn * numSynapsePerNeuron * columnID + columnSynapseIdx;
                const int threadSynapseXIdx = rfXIdx - rfXIdxStart;
                const int threadSynapseIdx = threadSynapseYIdx * xBatchSize + threadSynapseXIdx;
                incStart[threadSynapseIdx] = spike_time_in[inputSpikeIdx];
                incEnd[threadSynapseIdx] = spike_time_in[inputSpikeIdx] + weights[layerSynapseIdx];
            }
        }
        
        // always runs for gammalength cycles
        for (int tickCycles = 0; tickCycles < gammaLength; ++tickCycles) {
            
            for(int rfYIdx = rfYIdxStart; rfYIdx < rfYIdxEnd; ++rfYIdx) {
                for(int rfXIdx = rfXIdxStart; rfXIdx < rfXIdxEnd; ++rfXIdx) {
                    // within a block (column, rf):
                    // threadIdx y and z (synapses in a neuron):
                    //                    |      rfXIdx 0 to 2                   |
                    //                    |<-- z = rfsize*nChan----------------->|
                    //    rfYIdx 0 to 2  ^|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|
                    //          y = rfSize|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|
                    //                   v|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|
                    // threadIdx x (neurons in a column) reuses the above for each neuron
                    
                    // non-padded convolution
                    // blockIdx is the output matrix x/y index
                    // image coord: (blockIdx.x - rfsize/2 to blockIdx.x + rfsize/2) + rfsize/2
                    //              (blockIdx.y - rfsize/2 to blockIdx.y + rfsize/2) + rfsize/2
                    // TODO: consider different stride calculation, currently assuming stride == 1
                    const int spikeTimeInputX = blockIdx.x + rfXIdx;
                    const int spikeTimeInputY = blockIdx.y + rfYIdx;

                    const int inputPixelIdx = spikeTimeInputY * inputImgSizeX + spikeTimeInputX;

                    // index into spike_time_in for this synapse
                    const int dataPixelIdx = dataIdx * numInputPixels + inputPixelIdx;
                    const int inputSpikeIdx = dataPixelIdx * nPrevChan + chanIdx;

                    const int rfPixelIdx = rfSize * rfYIdx + rfXIdx;
                    // for indexing into neuron local synapse
                    const int neuronSynapseIdx = rfPixelIdx * nPrevChan + chanIdx;
                    // for indexing into column local synapse
                    const int columnSynapseIdx = neuronID * numSynapsePerNeuron + neuronSynapseIdx;
                    // for indexing into layer synpase weights
                    const int layerSynapseIdx = numNeuronPerColumn * numSynapsePerNeuron * columnID + columnSynapseIdx;

                    // TODO loop through chanIdx and see if threads local synapses should contribute
                    // local incStart and incEnd to compare to cycle time
                    // for each synapse, check the spike_time_in to see if start spiking
                    const int threadSynapseYIdx = rfYIdx - rfYIdxStart;
                    const int threadSynapseXIdx = rfXIdx - rfXIdxStart;
                    const int threadSynapseIdx = threadSynapseYIdx * xBatchSize + threadSynapseXIdx;
                    if (tickCycles >= incStart[threadSynapseIdx] && tickCycles < incEnd[threadSynapseIdx]) {
                        atomicAdd_block(body_pot_shared[neuronID], 1);
                    }
                    // __syncthreads();
                }
            }
            // TODO calculate local contribution to bodyPot on each thread
            // TODO atomicAdd_block to the bodyPot
    }
    

    // Each thread keeps several local synpase_rnl functions, and at each time tick,
    // each thread computes a local increament and use atomicAdd_block to the shared
    // neuronBodyPot

    // phase 2: copy spike_time_out corresponding to the neuron's column to shared memory
    // and perform WTA within single block for each column? 
    // (1 block within each column performs this, rest of the blocks (corresponding to other neurons in phase 1) are idle)

    // phase 3: each neuron copy final spike_time_out corresponding to itself and perform stdp

    // Phase 1: input spike -> RNL -> neuron body pot -> neuron spike time
    // Phase 2: neuron spike time -> WTA -> spike time out / inhibition
    // Phase 3: spike time out vs spike time in -> STDP

    for (int dataIdx = 0; dataIdx < dataLength; ++dataLength) {
        // TODO change this to consider potential convolution
        // index into spike_time_in for this synapse
        const int dataPixelIdx = dataIdx * numInputPixels + inputPixelIdx;
        const int inputSpikeIdx = dataPixelIdx * nChan + chanID;

        
        
        // local incStart and incEnd to compare to cycle time
        int incStart = spike_time_in[inputSpikeIdx];
        int incEnd = spike_time_in[inputSpikeIdx] + weights[layerSynapseIdx];
        
        // reset increment indicator
        synapseRNL_shared[columnSynapseIdx] = false;
        // reset earliest time
        if (neuronID == 0 && synapseID == 0) {
            earliestSpikingTime_shared = gammaLength;
        }
            
        // neuron corresponds to output wire
        const int outputDataPixelIdx = dataIdx * numColumns + columnID;
        const int outputNeuronIdx = outputDataPixelIdx * numNeuronPerColumn + neuronID;
        if (synapseID == 0) {
            // reset body potential for this gama cycle (single dataIdx)
            neuronBodyPot_shared[neuronID] = 0;
            // reset spike time to gamma (no spike)
            //spikeTimeOutNoInhibit[neuronID] = gammaLength;
            spike_time_out[outputNeuronIdx] = gammaLength;
        }
        __syncthreads();

        // end iteration when body potential reaches threshold before gamma cycle time (spikes)
        // end iteration after cycle time reaches gamma cycle time and no spike
        for (int tickCycles = 0; tickCycles < gammaLength; ++tickCycles) {
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
            spike_time_out[outputDataPixelIdx * numNeuronPerColumn + earliestSpikingNeuron_shared] = earliestSpikingTime_shared;
        }
        
        // STDP and update weights
        bool isCausal = spike_time_in[inputSpikeIdx] <= spike_time_out[outputNeuronIdx];
        bool inHasSpike = spike_time_in[inputSpikeIdx] < gammaLength;
        bool outHasSpike = spike_time_out[outputNeuronIdx] < gammaLength;
        bool isCapture = inHasSpike && outHasSpike && isCausal;
        bool isBackoff = (inHasSpike && outHasSpike && !isCausal) ||
                         (!inHasSpike && outHasSpike);
        bool isSearch = inHasSpike && !outHasSpike;
        //  w_max = gammaLength - 1
        bool weightOverHalf = weights[layerSynapseIdx] >= (gammaLength - 1.) / 2.;
        float stabUp = weightOverHalf ? cuConstTNNParams.rate_capture : (cuConstTNNParams.rate_capture / 2.);
        float stabDown = !weightOverHalf ? cuConstTNNParams.rate_backoff : (cuConstTNNParams.rate_backoff / 2.);
        if (isCapture) weights[layerSynapseIdx] += cuConstTNNParams.rate_capture * stabUp;
        else if (isBackoff) weights[layerSynapseIdx] -= cuConstTNNParams.rate_backoff * stabDown;
        else if (isSearch) weights[layerSynapseIdx] += cuConstTNNParams.rate_search;
        // Clamp weights between 0 and gammalength - 1
        weights[layerSynapseIdx] = weights[layerSynapseIdx] < 0. ? 0. : weights[layerSynapseIdx];
        weights[layerSynapseIdx] = weights[layerSynapseIdx] > (gammaLength - 1.) ? (gammaLength - 1.) : weights[layerSynapseIdx];
    }
}

void setup() {
    GlobalConstants params;

    // Fill in params
    params.wres = 3;
    params.rate_capture = 1./2.;
    params.rate_backoff = 1./1024.;
    params.rate_search = 1./2.;
    params.spikeThreshold = 400;

    cudaMemcpyToSymbol(&cuConstTNNParams, &params, sizeof(GlobalConstants));
}



/**
 * @brief 
 * 
 * @param[in] spike_time_in loaded spike time data (device pointer)
 * @param[in] dataLength how many sets are in there
 * @param[out] spike_time_out array for output spike time data (device pointer)
 */
void launch_column(layerParams& params, int dataLength, char* spike_time_in) {
    assert(params.stride == 1);
    params.outputDim = (params.inputDim - params.rfsize)/params.stride + 1;
    
    params.nSynapsesPerNeuron = params.rfsize * params.rfsize * params.nPrevChan;
    
    // weights are initialized in the kernel
    cudaMalloc(&params.weights, sizeof(float) * params.nSynapsesPerNeuron 
                                            * params.nNeurons 
                                            * params.outputDim * params.outputDim);
    
    cudaMalloc(&params.spike_time_out, sizeof(char) * params.nNeurons * params.outputDim * params.outputDim * dataLength);
    int totalSharedSize = sizeof(int) * params.nNeurons + sizeof(int)*2 + sizeof(bool) * params.nSynapsesPerNeuron * params.nNeurons;
    column_kernel<<<dim3(params.outputDim, params.outputDim), 
                    dim3(params.nNeurons, 
                         params.rfsize, 
                         params.rfsize * params.nPrevChan), totalSharedSize>>>(params.weights, spike_time_in, params.spike_time_out, dataLength);
    cudaError_t err = cudaGetLastError();

     if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err)); 
        printf("outputDim %d, nNeurons %d, rfsize %d, nPrevChan %d\n", params.outputDim, params.nNeurons, params.rfsize, params.nPrevChan);
        printf("Total grid dim %d, total block dim %d\n", params.outputDim * params.outputDim, params.nNeurons * params.rfsize * params.rfsize * params.nPrevChan);
        printf("Total shared %d\n", totalSharedSize);
     }
}

// weights float -> weights uchar (* UINT8_MAX / gammaLength)
/*
 * Expected weights output format:
 *  <---------------------> numNeuronPerColumn * rfSize
 *  pos pos pos pos pos pos ^
 *  neg neg neg neg neg neg |
 *            ...           |
 *  neg neg neg neg neg neg | numColumns * nChan * rfSize
 *  pos pos pos pos pos pos |
 *  neg neg neg neg neg neg v
 * 
 * Raw Weights output format for 1 neuron:
 * <-            rfSize * nChans        ->
 * |pos neg|pos neg|pos neg| ... |pos neg| ^
 * |pos neg|pos neg|pos neg| ... |pos neg| |
 * |pos neg|pos neg|pos neg| ... |pos neg| |
 * |                   ...               | rfSize
 * |pos neg|pos neg|pos neg| ... |pos neg| |
 * |pos neg|pos neg|pos neg| ... |pos neg| |
 * |pos neg|pos neg|pos neg| ... |pos neg| v
 */
    // column_kernel<<<dim3(params.outputDim, params.outputDim), 
    //                 dim3(params.nNeurons, 
    //                      params.rfsize, 
    //                      params.rfsize * params.nPrevChan)>>>(params.weights, spike_time_in, params.spike_time_out, dataLength);
    // const int numNeurons, const int rfSize, const int nChan,
    // Each block generates image for 1 column, each thread writes pixel for 1 synapse weight
__global__  void weightsToImg (const float* weights, uint8_t* device_img) {
    const int columnID = blockIdx.y * gridDim.x + blockIdx.x; // for output

    const int numNeuronPerColumn = blockDim.x;
    const int rfSize = blockDim.y;
    const int nChan = blockDim.z / rfSize;
    const int numSynapsePerNeuron = rfSize * rfSize * nChan;
    const int neuronID = threadIdx.x; //!< within a column
    // synapseID (row major order, 2 channels for 1 pixel interleaves)
    const int synapseID = threadIdx.y * blockDim.z + threadIdx.z; //!< within a neuron
    // each synapse in each neuron has its own weights, updated across data samples
    const int columnSynapseIdx = neuronID * numSynapsePerNeuron + synapseID;
    const int layerSynapseIdx = numNeuronPerColumn * numSynapsePerNeuron * columnID + columnSynapseIdx;

    const int rfXIdx = threadIdx.z / nChan;
    const int chanID = threadIdx.z % nChan;
    const int rfYIdx = threadIdx.y;
    
    const int gammaLength = 1 << cuConstTNNParams.wres;
    
    // TODO verify the below
    const int outputPixelIdx = rfSize * rfSize * (numNeuronPerColumn * (columnID * nChan + chanID) + neuronID) + rfYIdx * rfSize + rfXIdx;
    device_img[outputPixelIdx] = (uint8_t)(weights[layerSynapseIdx] * UINT8_MAX / gammaLength);
}

uint8_t* convertToHostImg(layerParams& params) {
    uint8_t* device_img;
    const int imgSize = params.rfsize * params.rfsize * params.nNeurons * params.nPrevChan * params.outputDim * params.outputDim;
    cudaMalloc(&device_img, imgSize);
    uint8_t* host_img = (uint8_t *)malloc(imgSize);

    int dataLength = 10000;
    int spike_time_out_size = sizeof(char) * params.nNeurons * params.outputDim * params.outputDim * dataLength;
    float * host_weights = (float *)malloc(imgSize*sizeof(float));
    char* host_spike_time_out = (char *)malloc(spike_time_out_size*sizeof(char));
    cudaMemcpy(host_weights, params.weights, imgSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_spike_time_out, params.spike_time_out, spike_time_out_size, cudaMemcpyDeviceToHost);
    bool allZero = true;
    int numZeros = 0;
    int numNonzeros = 0;
    for (int i = 0; i < imgSize; ++i) {
        if (host_weights[i] == 0)
            ++numZeros;
        else
            ++numNonzeros;
    }
    printf("numZeros = %d\n", numZeros);
    printf("numNonzeros = %d\n", numNonzeros);
    free(host_weights);

    int numSpikes = 0;
    int numInfs = 0;
    const int gammaLength = 8;
    for (int i = 0; i < spike_time_out_size; ++i) {
        if (host_spike_time_out[i] == 0)
            ++numSpikes;
        else
            ++numInfs;
    }
    printf("numSpikes = %d\n", numSpikes);
    printf("numInfs = %d\n", numInfs);
    free(host_spike_time_out);

    weightsToImg<<<dim3(params.outputDim, params.outputDim), 
                    dim3(params.nNeurons, 
                         params.rfsize, 
                         params.rfsize * params.nPrevChan)>>>(params.weights, device_img);
    cudaMemcpy(host_img, device_img, imgSize, cudaMemcpyDeviceToHost);
    return host_img;
}
__global__  void spikesToImg (const char* spikes, uint8_t* device_img) {
    // const int columnID = blockIdx.y * gridDim.x + blockIdx.x; // for output

    // const int numNeuronPerColumn = blockDim.x;
    // const int rfSize = blockDim.y;
    // const int nChan = blockDim.z / rfSize;
    // const int numSynapsePerNeuron = rfSize * rfSize * nChan;
    // const int neuronID = threadIdx.x; //!< within a column
    // // synapseID (row major order, 2 channels for 1 pixel interleaves)
    // const int synapseID = threadIdx.y * blockDim.z + threadIdx.z; //!< within a neuron
    // // each synapse in each neuron has its own weights, updated across data samples
    // const int columnSynapseIdx = neuronID * numSynapsePerNeuron + synapseID;
    // const int layerSynapseIdx = numNeuronPerColumn * numSynapsePerNeuron * columnID + columnSynapseIdx;

    // const int rfXIdx = threadIdx.z / nChan;
    // const int chanID = threadIdx.z % nChan;
    // const int rfYIdx = threadIdx.y;
    
    const int gammaLength = 1 << cuConstTNNParams.wres;
    
    // TODO verify the below
    const int nImg = gridDim.z;
    const int imgId = blockIdx.z;
    const int imgDim = gridDim.y;
    const int spikePixelIdx = imgDim * imgDim * imgId + blockIdx.y * imgDim + blockIdx.x;

    const int nChan = blockDim.x;
    const int chanId = threadIdx.x;

    const int spikeIdx = spikePixelIdx * nChan + chanId;
    const int outputPixelXIdx = imgId * imgDim + blockIdx.x;
    const int outputPixelYIdx = imgDim * chanId + blockIdx.y;
    const int outputImgWidth = imgDim * nImg;
    const int outputPixelIdx = outputPixelYIdx * outputImgWidth + outputPixelXIdx;
    device_img[outputPixelIdx] = (uint8_t)(spikes[spikeIdx] * UINT8_MAX / gammaLength);
}

uint8_t* convertSpikesToHostImg(layerParams& params) {
    // uint8_t* hostimg = (uint8_t *) malloc(params.inputDim * params.inputDim * 2 * 12);
    // cudaMemcpy(hostimg, params.spike_time_out, params.inputDim * params.inputDim * 2 * 12, cudaMemcpyDeviceToHost);
    // return hostimg;
    uint8_t* device_img;
    const int imgSize = params.rfsize * params.rfsize * params.nNeurons * params.nPrevChan * params.outputDim * params.outputDim;
    cudaMalloc(&device_img, imgSize);
    uint8_t* host_img = (uint8_t *)malloc(imgSize);

    spikesToImg<<<dim3(params.outputDim, params.outputDim, 12), 
                    2>>>(params.spike_time_out, device_img);
    cudaMemcpy(host_img, device_img, imgSize, cudaMemcpyDeviceToHost);
    cudaFree(device_img);
    return host_img;
}