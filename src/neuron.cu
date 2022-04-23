#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <string>
#include <arpa/inet.h>

#include "CycleTimer.h"
struct GlobalConstants {
    // Spike delay and weight resolution
    int wres;
    int rate_capture;
    int rate_backoff;
    int rate_search;
    int spikeThreshold;
};
__constant__ GlobalConstants cuConstTNNParams;
/**
 * @brief Simulates a synapse
 * 1 layer contains x*y columns (number of rfs after convolution with a kernel). Determines the number of output wire groups.
 * 1 neuron column contains numNeuronPerColumn neurons, output 1 spike on numNeuronPerColumn lines
 * each neuron has numSynapsePerNeuron synapses
 * @param[inout] weight initial synapse weight, and final updated weight array
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
 //TODO: change int that are used for spike time to char
__global__ void column_kernel(int* weight, char* spike_time_in, char* spike_time_out, int dataLength) {
    // rows * columns of output spike times
    const int numColumns = gridDim.x * gridDim.y;
    const int columnID = blockIdx.y * gridDim.x + blockIdx.x; // for output

    const int numNeuronPerColumn = blockDim.x;
    const int rfSize = blockDim.y;
    const int nChan = blockDim.z / rfSize;
    const int numSynapsePerNeuron = rfSize * rfSize * nChan;

    // within a block (column, rf):
    // threadIdx y and z (synapses in a neuron):
    //                    |      rfXIdx 0 to 2                   |
    //                    |<-- z = rfsize*nChan----------------->|
    //    rfYIdx 0 to 2  ^|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|
    //          y = rfSize|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|
    //                   v|ch0ch1ch2ch3|ch0ch1ch2ch3|ch0ch1ch2ch3|
    // threadIdx x (neurons in a column) reuses the above for each neuron
    
    const int neuronID = threadIdx.x; //!< within a column
    const int synapseID = threadIdx.y * blockDim.z + threadIdx.z; //!< within a neuron
    
    const int rfXIdx = threadIdx.z / nChan;
    const int chanID = threadIdx.z % nChan;
    const int rfYIdx = threadIdx.y;
    
    const int gammaLength = 1 << cuConstTNNParams.wres;

    // non-padded convolution
    // blockIdx is the output matrix x/y index
    // image coord: (blockIdx.x - rfsize/2 to blockIdx.x + rfsize/2) + rfsize/2
    //              (blockIdx.y - rfsize/2 to blockIdx.y + rfsize/2) + rfsize/2
    // TODO: consider different stride calculation, currently assuming stride == 1
    const int spikeTimeInputX = blockIdx.x + rfXIdx;
    const int spikeTimeInputY = blockIdx.y + rfYIdx;

    const int inputImgSizeX = gridDim.x + rfSize/2 + rfSize/2;
    const int inputImgSizeY = gridDim.y + rfSize/2 + rfSize/2;
    const int numInputPixels = inputImgSizeX * inputImgSizeY;
    const int inputPixelIdx = spikeTimeInputY * inputImgSizeX + spikeTimeInputX;

    // Let each 3d thread block represent a neuron column:
    // Within each neuron, different threads after updating the rnl function for each synapse
    // 1 thread update the body potential or use parallel reduce?


    // ==== Shared within each column, for each synapse of each neuron
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
    // may need for k-WTA
    //__shared__ int spikeTimeOutNoInhibit[numNeuronPerColumn];

    

    // record the earliest spike in the column
    // might use an array of these for k-WTA
    __shared__ int earliestSpikingNeuron_shared;
    __shared__ int earliestSpikingTime_shared;

    // 0 0 1 1 1 0 0 0
    // 0 0 0 1 1 1 1 0
    // 0 1 1 0 0 0 0 0

    // 0 1 3 5 


    for (int dataIdx = 0; dataIdx < dataLength; ++dataLength) {
        // TODO change this to consider potential convolution
        // index into spike_time_in for this synapse
        const int dataPixelIdx = dataIdx * numInputPixels + inputPixelIdx;
        const int inputSpikeIdx = dataPixelIdx * nChan + chanID;

        // each synapse in each neuron has its own weight, updated across data samples
        const int columnSynapseIdx = neuronID * numSynapsePerNeuron + synapseID;
        
        // local incStart and incEnd to compare to cycle time
        int incStart = spike_time_in[inputSpikeIdx];
        int incEnd = spike_time_in[inputSpikeIdx] + weight[columnSynapseIdx];
        
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
        
        // STDP and update weight
        bool isCausal = spike_time_in[inputSpikeIdx] <= spike_time_out[outputNeuronIdx];
        bool inHasSpike = spike_time_in[inputSpikeIdx] < gammaLength;
        bool outHasSpike = spike_time_out[outputNeuronIdx] < gammaLength;
        bool isCapture = inHasSpike && outHasSpike && isCausal;
        bool isBackoff = (inHasSpike && outHasSpike && !isCausal) ||
                         (!inHasSpike && outHasSpike);
        bool isSearch = inHasSpike && !outHasSpike;
        //  w_max = gammaLength - 1
        // since integral comparison, not doing -1 on gammaLength
        bool weightOverHalf = weight[columnSynapseIdx] >= gammaLength / 2;
        int stabUp = weightOverHalf ? cuConstTNNParams.rate_capture : cuConstTNNParams.rate_capture / 2;
        int stabDown = !weightOverHalf ? cuConstTNNParams.rate_backoff : cuConstTNNParams.rate_backoff / 2;
        if (isCapture) weight += cuConstTNNParams.rate_capture * stabUp;
        else if (isBackoff) weight -= cuConstTNNParams.rate_backoff * stabDown;
        else if (isSearch) weight += cuConstTNNParams.rate_search;
    }
}
void setup() {
    GlobalConstants params;
    // TODO: fill in params
    cudaMemcpyToSymbol(cuConstTNNParams, &params, sizeof(GlobalConstants));
}

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
__global__ void MNIST_loader_kernel(const int pnThreshold, char * cuMNISTdata, char * spike_time_in) {
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

    spike_time_in[posSpikeIdx] = cuMNISTdata[dataIdx] < pnThreshold ? 0 : gammaLength;
    spike_time_in[negSpikeIdx] = cuMNISTdata[dataIdx] >= pnThreshold ? 0 : gammaLength;
}

bool assertAtFileEnd(FILE * file) {
    int unused;
    int elemRead = fread(&unused, 1, 1, file);
    return elemRead == 0 && feof(file);
}
/**
 * @brief 
 * 
 * @param[in] input_file 
 * @param[out] spike_time_in 
 */
void load_MNIST(const char* image_input_file, const char* label_input_file, int dataLength, char* spike_time_in, char* labels) {
    const uint32_t IDX3_MAGIC = 0x803;
    const uint32_t IDX1_MAGIC = 0x801;
    // Load input MNIST dataset and cudaMemcpy
    // CudaMalloc spike_time_in and spike_time_out
    // MNIST_loader_kernel
    FILE * mnistImageFile = fopen(image_input_file, "r");
    assert(mnistImageFile != NULL);
    FILE * mnistLabelFile = fopen(label_input_file, "r");
    assert(mnistLabelFile != NULL);

    // Read image data
    int32_t magic, dims3[3];
    size_t elemRead = fread(&magic, sizeof(uint32_t), 1, mnistImageFile);
    assert(elemRead == 1);
    // MNIST dataset is big endian for the int32s
    magic = ntohl(magic); // network order to host order long
    assert(magic == IDX3_MAGIC);

    elemRead = fread(&dims3, sizeof(uint32_t), 3, mnistImageFile);
    assert(elemRead == 3);
    for(int i = 0; i < 3; ++i) {
        dims3[i] = ntohl(dims3[i]);
    }
    int32_t & nImgs = dims3[0];
    int32_t & nRows = dims3[1];
    int32_t & nCols = dims3[2];
    assert(nRows == nCols && nCols == 28);

    int nImageBytes = nImgs * nRows * nCols; // Each pixel has 1 byte.
    uint8_t* imgData = (uint8_t*) malloc(nImageBytes);
    elemRead = fread(&imgData, sizeof(uint8_t), nImageBytes, mnistImageFile);
    assert(elemRead == nImageBytes);
    assert(assertAtFileEnd(mnistImageFile));
    fclose(mnistImageFile);

    // copy pixel data to device
    char* imgData_device = NULL;
    cudaError_t cuerr = cudaMalloc(&imgData_device, nImageBytes);
    assert(cuerr == cudaSuccess);
    cuerr = cudaMemcpy(imgData_device, imgData, nImageBytes, cudaMemcpyHostToDevice);
    assert(cuerr == cudaSuccess);
    free(imgData);

    // launch kernel to convert to spike time
    MNIST_loader_kernel<<<nImgs, dim3(nRows, nCols)>>>(UINT8_MAX/2, imgData_device, spike_time_in);
    

    // Read label data into device memory as well for later classification
    elemRead = fread(&magic, sizeof(uint32_t), 1, mnistLabelFile);
    assert(elemRead == 1);
    magic = ntohl(magic);
    assert(magic == IDX1_MAGIC);
    
    int nLabels;
    elemRead = fread(&nLabels, sizeof(uint32_t), 1, mnistLabelFile);
    assert(elemRead == 1);
    nLabels = ntohl(nLabels);

    uint8_t * labelData = (uint8_t *)malloc(nLabels);
    elemRead = fread(labelData, 1, nLabels, mnistLabelFile);
    assert(elemRead == nLabels);
    assert(assertAtFileEnd(mnistLabelFile));
    fclose(mnistLabelFile);

    cuerr = cudaMemcpy(labels, labelData, nLabels, cudaMemcpyHostToDevice);
    assert(cuerr == cudaSuccess);
    free(labelData);
}

/**
 * @brief 
 * 
 * @param[in] spike_time_in loaded spike time data (device pointer)
 * @param[in] dataLength how many sets are in there
 * @param[out] spike_time_out array for output spike time data (device pointer)
 */
void launch_column(char* spike_time_in, int dataLength, char* spike_time_out) {
    
    
    // neuron count determined by number of classes at current layer
    int numNeuronPerColumn = 12;
    // synapse count determined by receptive field size
    //int numSynapsePerNeuron = rfsize * rfsize * chanleng;
    // rows and cols is the number of receptive fields
    //column_kernel<<<dim3(rows, cols), dim3(numNeuronPerColumn, rfsize, )>>>();

}
struct layerParams {
    int rfsize;
    int stride;
    int nNeurons;
    int nSynapse;
};

void main(){
    layerParams layers[3];
    
    int dataLength = 10000;

    char* spike_time_in = NULL;
    cudaMalloc(&spike_time_in, sizeof(char) * dataLength * something);
    load_MNIST("MNISTfilepath", dataLength, spike_time_in); // perform parallel load

    // setup constants
    setup();

    char* spike_time_out = NULL;
    char* weight = NULL;
    cudaMalloc(&weight, sizeof(int) * something);
    cudaMalloc(&spike_time_in, sizeof(char) * something);
    cudaMalloc(&spike_time_out, sizeof(char) * something);
    launch_column(spike_time_in, dataLength, spike_time_out);
}