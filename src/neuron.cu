#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <string>

#include <cassert>
#include <cmath>

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
    // rfSize
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


void launch_load_MNIST(int nImgs, int nRows, int nCols, uint8_t* imgData, uint8_t* & spike_time_in) {
    // copy raw MNIST to device
    int nImageBytes = nImgs * nRows * nCols; // Each pixel is 1 byte.
    uint8_t * imgData_device = NULL;
    cudaError_t cuerr = cudaMalloc(&imgData_device, nImageBytes);
    assert(cuerr == cudaSuccess);
    cuerr = cudaMemcpy(imgData_device, imgData, nImageBytes, cudaMemcpyHostToDevice);
    assert(cuerr == cudaSuccess);
    int nChan = 2;
    cuerr = cudaMalloc(&spike_time_in, sizeof(uint8_t) * nImageBytes * nChan);
    assert(cuerr == cudaSuccess);
    // launch kernel to convert to spike time
    MNIST_loader_kernel<<<nImgs, dim3(nRows, nCols)>>>(UINT8_MAX/2, imgData_device, spike_time_in);
    cudaDeviceSynchronize();
    cuerr = cudaGetLastError(); // clear any previous error
    if (cuerr != cudaSuccess) {
        printf("Cuda error after MNIST_loader_kernel: %s\n", cudaGetErrorString(cuerr)); 
    }
}
void copyLabelToDevice(int nLabels, uint8_t * labelData, uint8_t *& labels) {
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
__global__ void column_kernel(layerParams params, int dataLength, uint8_t* spike_time_in) {
    const uint8_t gammaLength = 1 << cuConstTNNParams.wres;

    float* weights = params.weights;
    uint8_t* spike_time_out = params.spike_time_out;
    const int rfSize = params.rfSize;
    const int yBatchSize = params.yBatchSize;
    const int xBatchSize = params.xBatchSize;
    const int nPrevChan = params.nPrevChan;
    const int numSynapsePerNeuron = rfSize * rfSize * nPrevChan;

    
    // const int synapsesPerThread = xBatchSize * yBatchSize;
    //assert(synapsesPerThread < 1024);
    // rows * columns of output spike times
    const int numColumns = gridDim.x * gridDim.y;
    const int columnIdx = blockIdx.y * gridDim.x + blockIdx.x; // for output

    const int numNeuronPerColumn = blockDim.x;

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
    extern __shared__ int _AllShared[];
    // bool * const& synapseRNL_shared = (bool*) &_AllShared[sizeof(int) * numNeuronPerColumn + sizeof(int)*2];
    // int totalSharedSize = sizeof(int) * params.nNeurons + sizeof(int)*2;
    int * const& neuronBodyPot_shared = _AllShared;
    int * const& neuronSpikeTime_shared = &_AllShared[numNeuronPerColumn];
    int * const& neuronSpikePot_shared = &_AllShared[numNeuronPerColumn * 2];
    // int & earliestSpikingNeuron_shared = _AllShared[numNeuronPerColumn];
    // int & earliestSpikingTime_shared = _AllShared[numNeuronPerColumn + 1];
    // uint8_t & synapseHitCount_shared = *(uint8_t*)&_AllShared[numNeuronPerColumn + 2];
    /*
    Per column:
    nNeuron
    nSynapse per neuron: = rfSize*rfSize*nprevchan
        nPrevChan
        rfSize
    Per neuron: sum synapse at each timestep and check firing
    Within column: WTA across neuron firing time
    STDP requires 1 final neuron firing time for weight update per neuron's synapse
    */

    // small rfSize: nNeuron*nSynapse fits within threadsPerBlock

    // large rfSize: only rfSize*rfSize(maybe nprevchan) fits within threadsPerBlock

    // parallelize as much synapse as possible but loop through neuron
    // phase 1 threads: rfSize*rfSize*fraction of nprevchan,
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

    // Actual kernal launch parameters for now:
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

    // Initialize all the weights in batches
    if (params.stdpEn) {
        for(int rfYIdx = rfYIdxStart; rfYIdx < rfYIdxEnd; ++rfYIdx) {
            for(int rfXIdx = rfXIdxStart; rfXIdx < rfXIdxEnd; ++rfXIdx) {
                const int rfPixelIdx = rfSize * rfYIdx + rfXIdx;
                // for indexing into neuron local synapse
                const int neuronSynapseIdx = rfPixelIdx * nPrevChan + chanIdx;
                // for indexing into column local synapse
                const int columnSynapseIdx = neuronIdx * numSynapsePerNeuron + neuronSynapseIdx;
                // for indexing into layer synpase weights
                const int layerSynapseIdx = numNeuronPerColumn * numSynapsePerNeuron * columnIdx + columnSynapseIdx;
                // half initialization for now
                weights[layerSynapseIdx] = (gammaLength - 1.) / 2;
                // TODO implement a choice of random initialization
            }
        }
    }

    for (int dataIdx = 0; dataIdx < dataLength; ++dataIdx) {
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            printf("dataIdx %d\r", dataIdx);
        }
        // if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        //     earliestSpikingTime_shared = gammaLength;
        //     earliestSpikingNeuron_shared = 0;
        // }
        // neuron corresponds to output wire
        const int outputDataPixelIdx = dataIdx * numColumns + columnIdx;
        const int outputNeuronIdx = outputDataPixelIdx * numNeuronPerColumn + neuronIdx;
        if (threadIdx.y == 0 && threadIdx.z == 0) {
            // reset body potential for this gama cycle (single dataIdx)
            neuronBodyPot_shared[neuronIdx] = 0;
            neuronSpikeTime_shared[neuronIdx] = gammaLength;
            neuronSpikePot_shared[neuronIdx] = 0;
            // reset spike time to gamma (no spike)
            //spikeTimeOutNoInhibit[neuronIdx] = gammaLength;
            spike_time_out[outputNeuronIdx] = gammaLength;
        }
        
        // TODO use larger consts later
        int incStart[49];
        int incEnd[49];
        // Loop through thread local synapses to initialize incStart and incEnd
        for(int rfYIdx = rfYIdxStart; rfYIdx < rfYIdxEnd; ++rfYIdx) {
            const int threadSynapseYIdx = rfYIdx - rfYIdxStart;
            for(int rfXIdx = rfXIdxStart; rfXIdx < rfXIdxEnd; ++rfXIdx) {
                // non-padded convolution
                // blockIdx is the output matrix x/y index
                // image coord: (blockIdx.x - rfSize/2 to blockIdx.x + rfSize/2) + rfSize/2
                //              (blockIdx.y - rfSize/2 to blockIdx.y + rfSize/2) + rfSize/2
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
                const int columnSynapseIdx = neuronIdx * numSynapsePerNeuron + neuronSynapseIdx;
                // for indexing into layer synpase weights
                const int layerSynapseIdx = numNeuronPerColumn * numSynapsePerNeuron * columnIdx + columnSynapseIdx;
                const int threadSynapseXIdx = rfXIdx - rfXIdxStart;
                const int threadSynapseIdx = threadSynapseYIdx * xBatchSize + threadSynapseXIdx;
                incStart[threadSynapseIdx] = spike_time_in[inputSpikeIdx];
                incEnd[threadSynapseIdx] = spike_time_in[inputSpikeIdx] + static_cast<uint8_t>(weights[layerSynapseIdx]);
            }
        }
        __syncthreads();
   
        // always runs for gammalength cycles
        for (int tickCycles = 0; tickCycles < gammaLength; ++tickCycles) {
            int localBodyPot = 0;
            for(int rfYIdx = rfYIdxStart; rfYIdx < rfYIdxEnd; ++rfYIdx) {
                for(int rfXIdx = rfXIdxStart; rfXIdx < rfXIdxEnd; ++rfXIdx) {
                    const int threadSynapseYIdx = rfYIdx - rfYIdxStart;
                    const int threadSynapseXIdx = rfXIdx - rfXIdxStart;
                    const int threadSynapseIdx = threadSynapseYIdx * xBatchSize + threadSynapseXIdx;
                    if (tickCycles >= incStart[threadSynapseIdx] && tickCycles < incEnd[threadSynapseIdx]) {
                        ++localBodyPot;
                    }
                }
            }
            atomicAdd_block(&neuronBodyPot_shared[neuronIdx], localBodyPot);
            __syncthreads();
                            
            if(threadIdx.y == 0 && threadIdx.z == 0) {            
                if (neuronBodyPot_shared[neuronIdx] >= cuConstTNNParams.spikeThreshold) {
                    // neuronSpikingTime_shared[numNeuron];
                    // neuronSpikingBodyPot_shared[numNeuron];
                    // record the earliest spike in the column
                    // Only synapse 0 need to do this, but not adding if condition
                    // to avoid a conditional lane mask
                    
                    // NOTE: We assume when multiple threads writes the same address, HW will
                    //       sequentialize writes. This would slow down the program.
                    // NOTE: the assumption is that if multiple neuron spike at the same time, a race doesn't matter
                    //       but that's with the assumption that tickCycles is synced
                    // TODO: doublecheck if possible race condition among neurons affect correctness
                    //       e.g. if neurons are across multiple warps and they are at different tickCycles
                    if (neuronSpikeTime_shared[neuronIdx] > tickCycles) {
                        neuronSpikePot_shared[neuronIdx] = neuronBodyPot_shared[neuronIdx];
                        neuronSpikeTime_shared[neuronIdx] = tickCycles;
                    }
                }
            }

            __syncthreads();
        }

        __syncthreads();
        // 1-WTA spike time out
        // reduce min for global spike time out
        // thread 0 write earliestSpikingTime_shared to output at earliestSpikingNeuron_shared
        // all the later spikes are inhibited by default
        int earliestSpikingNeuron_local = neuronIdx;
        for(int crtNeuronIdx = 0; crtNeuronIdx < numNeuronPerColumn; ++crtNeuronIdx) {
            if (neuronSpikeTime_shared[crtNeuronIdx] < neuronSpikeTime_shared[earliestSpikingNeuron_local] ||
                (neuronSpikeTime_shared[crtNeuronIdx] == neuronSpikeTime_shared[earliestSpikingNeuron_local] &&
                     neuronSpikePot_shared[crtNeuronIdx] > neuronSpikePot_shared[earliestSpikingNeuron_local])) {
                earliestSpikingNeuron_local = crtNeuronIdx;
            }
        }
        if (threadIdx.y == 0 && threadIdx.z == 0 && neuronIdx == earliestSpikingNeuron_local) {
            spike_time_out[outputNeuronIdx] = neuronSpikeTime_shared[neuronIdx];
        }

        __syncthreads();
        // STDP and update weights
        if (!params.stdpEn) {
            continue;
        }
        for(int rfYIdx = rfYIdxStart; rfYIdx < rfYIdxEnd; ++rfYIdx) {
            for(int rfXIdx = rfXIdxStart; rfXIdx < rfXIdxEnd; ++rfXIdx) {
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
                const int columnSynapseIdx = neuronIdx * numSynapsePerNeuron + neuronSynapseIdx;
                // for indexing into layer synpase weights
                const int layerSynapseIdx = numNeuronPerColumn * numSynapsePerNeuron * columnIdx + columnSynapseIdx;

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
        __syncthreads();
    }
}

void setup() {
    GlobalConstants params;

    // Fill in params
    params.wres = 3;
    params.rate_capture = 1./2.;
    params.rate_backoff = 1./2.;
    params.rate_search = 1./1024.;
    params.spikeThreshold = 3000;
    cudaError_t err = cudaGetLastError(); // clear any previous error
    if (err != cudaSuccess) {
        printf("Cuda error before memcpy to symbol: %s\n", cudaGetErrorString(err)); 
    }
    cudaMemcpyToSymbol(cuConstTNNParams, &params, sizeof(GlobalConstants));
    err = cudaGetLastError(); // clear any previous error
    if (err != cudaSuccess) {
        printf("Cuda error while memcpy to symbol: %s\n", cudaGetErrorString(err)); 
    }
}



/**
 * @brief 
 * 
 * @param[in] spike_time_in loaded spike time data (device pointer)
 * @param[in] dataLength how many sets are in there
 * @param[out] spike_time_out array for output spike time data (device pointer)
 */
void launch_column(layerParams& params, int dataLength, uint8_t* spike_time_in) {
    assert(params.stride == 1);
    params.outputDim = (params.inputDim - params.rfSize)/params.stride + 1;
    
    params.nSynapsesPerNeuron = params.rfSize * params.rfSize * params.nPrevChan;
    
    // weights are initialized in the kernel
    cudaError_t err;
    if(params.stdpEn) {
        cudaMalloc(&params.weights, sizeof(float) * params.nSynapsesPerNeuron 
                                                * params.nNeurons 
                                                * params.outputDim * params.outputDim);
        err = cudaGetLastError(); // clear any previous error
        if (err != cudaSuccess) {
            printf("Cuda error after malloc weights: %s\n", cudaGetErrorString(err)); 
        }
    }
    else {
        assert(params.weights != NULL);
    }
    cudaMalloc(&params.spike_time_out, sizeof(uint8_t) * params.nNeurons * params.outputDim * params.outputDim * dataLength);
    err = cudaGetLastError(); // clear any previous error
    if (err != cudaSuccess) {
        printf("Cuda error after malloc spike_time_out: %s\n", cudaGetErrorString(err)); 
    }
    int totalSharedSize = sizeof(int) * params.nNeurons * 3;
    // int totalSharedSize = sizeof(int) * params.nNeurons + sizeof(int)*3;
    int nXYthreads = static_cast<int>(floor(sqrt(1024.0/params.nNeurons/params.nPrevChan)));
    if (nXYthreads < 1) assert(false && "nNeuron and nPrevChan not supported");
    int batchSize = (params.rfSize + nXYthreads - 1)/nXYthreads;
    params.xBatchSize = batchSize;
    params.yBatchSize = batchSize;
    err = cudaGetLastError(); // clear any previous error
    if (err != cudaSuccess) {
        printf("Cuda error before kernel launch: %s\n", cudaGetErrorString(err)); 
    }
    // nXYthreads from batch size: (params.rfSize + params.yBatchSize - 1) / params.yBatchSize
    column_kernel<<<dim3(params.outputDim, params.outputDim),
                    dim3(params.nNeurons, nXYthreads, nXYthreads * params.nPrevChan),
                    totalSharedSize>>>(params, dataLength, spike_time_in);
    cudaDeviceSynchronize();
    err = cudaGetLastError();

     if (err != cudaSuccess) {
        printf("Cuda error after kernel launch: %s\n", cudaGetErrorString(err)); 
        printf("outputDim %d, nNeurons %d, rfSize %d, nPrevChan %d\n", params.outputDim, params.nNeurons, params.rfSize, params.nPrevChan);
        printf("Total grid dim %d, total block dim %d\n", params.outputDim * params.outputDim, params.nNeurons * params.rfSize * params.rfSize * params.nPrevChan);
        printf("Total shared %d\n", totalSharedSize);
     }
    cudaFree(spike_time_in);
}

void cudaFreeHostWrap(void* ptr) {
    cudaFree(ptr);
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
    //                      params.rfSize, 
    //                      params.rfSize * params.nPrevChan)>>>(params.weights, spike_time_in, params.spike_time_out, dataLength);
    // const int numNeurons, const int rfSize, const int nChan,
    // Each block generates image for 1 column, each thread writes pixel for 1 synapse weight
// TODO change kernel configuration
__global__  void weightsToImg (layerParams params, uint8_t* device_img) {
    float* weights = params.weights;
    const int rfSize = params.rfSize;
    const int yBatchSize = params.yBatchSize;
    const int xBatchSize = params.xBatchSize;
    const int nPrevChan = params.nPrevChan;
    const int numSynapsePerNeuron = rfSize * rfSize * nPrevChan;

    const int gammaLength = 1 << cuConstTNNParams.wres;
    // rows * columns of output spike times
    const int columnIdx = blockIdx.y * gridDim.x + blockIdx.x; // for output

    const int numNeuronPerColumn = blockDim.x;

    // Actual kernal launch parameters for now:
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
    for(int rfYIdx = rfYIdxStart; rfYIdx < rfYIdxEnd; ++rfYIdx) {
        for(int rfXIdx = rfXIdxStart; rfXIdx < rfXIdxEnd; ++rfXIdx) {
            const int rfPixelIdx = rfSize * rfYIdx + rfXIdx;
            // for indexing into neuron local synapse
            const int neuronSynapseIdx = rfPixelIdx * nPrevChan + chanIdx;
            // for indexing into column local synapse
            const int columnSynapseIdx = neuronIdx * numSynapsePerNeuron + neuronSynapseIdx;
            // for indexing into layer synpase weights
            const int layerSynapseIdx = numNeuronPerColumn * numSynapsePerNeuron * columnIdx + columnSynapseIdx;
            // TODO verify the below
            const int outputPixelIdx = rfSize * rfSize * (numNeuronPerColumn * (columnIdx * nPrevChan + chanIdx) + neuronIdx) + rfPixelIdx;
            device_img[outputPixelIdx] = (uint8_t)(weights[layerSynapseIdx] * UINT8_MAX / gammaLength);
        }
    }
}

uint8_t * weightsToImg_sequential(layerParams& params) {
    // just outputs 1 column for now
    assert(params.outputDim == 1);
    // float* weights = params.weights;
    const int rfSize = params.rfSize;
    // const int yBatchSize = params.yBatchSize;
    // const int xBatchSize = params.xBatchSize;
    const int nPrevChan = params.nPrevChan;
    const int numSynapsePerNeuron = rfSize * rfSize * nPrevChan;

    // const int gammaLength = 1 << cuConstTNNParams.wres;
    int weightArrSize = params.outputDim * params.outputDim // columns
                                          * params.nNeurons
                                          * params.rfSize * params.rfSize * params.nPrevChan; // synapses
    float * hostWeights = (float*) malloc(weightArrSize * sizeof(float));
    cudaError_t err = cudaMemcpy(hostWeights, params.weights, weightArrSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Cuda error doing memcpy: %s\n", cudaGetErrorString(err)); 
    }
    // weightsToImg<<<dim3(params.outputDim, params.outputDim), 
    //                 dim3(params.nNeurons, nXYthreads, nXYthreads * params.nPrevChan)>>>(params, device_img);

    // // rows * columns of output spike times
    // for (int columnIdx = 0; columnIdx < params.outputDim * params.outputDim; ++columnIdx){}
    uint8_t * outimg = (uint8_t*) malloc(weightArrSize);
    const int numNeuronPerColumn = params.nNeurons;

    // // Actual kernal launch parameters for now:
    // // nNeuron, nYbatches, nXbatches* nPrevChan
    // const int neuronIdx = threadIdx.x;
    // const int chanIdx = threadIdx.z % nPrevChan;
    // xbatch0|xbatch1|xbatch2
    // ch0ch1 |ch0ch1 |ch0ch1 
    // th0th1  th2th3  th4th5
    // thread 0: x01234 ch0
    // thread 1: x01234 ch1

    int outImgWidth = params.nNeurons * params.rfSize;
    int columnIdx = 0;
    for(int chanIdx = 0; chanIdx < params.nPrevChan; ++chanIdx) {
        for(int rfYIdx = 0; rfYIdx < rfSize; ++rfYIdx) {
            for(int neuronIdx = 0; neuronIdx < params.nNeurons; ++neuronIdx){
                for(int rfXIdx = 0; rfXIdx < rfSize; ++rfXIdx) {
                // column0,0|rf0,0 |rf0,1 |rf0,2 |rf1,0 |rf1,1 |rf1,2 |rf2,0 |rf2,1 |rf2,2 |
                //   neuron0|ch0ch1|ch0ch1|ch0ch1|ch0ch1|ch0ch1|ch0ch1|ch0ch1|ch0ch1|ch0ch1|
                //   neuron1
                //   ...
                // column0,1
                    const int rfPixelIdx = rfSize * rfYIdx + rfXIdx;
                    // for indexing into neuron local synapse
                    const int neuronSynapseIdx = rfPixelIdx * nPrevChan + chanIdx;
                    // for indexing into column local synapse
                    const int columnSynapseIdx = neuronIdx * numSynapsePerNeuron + neuronSynapseIdx;
                    // for indexing into layer synpase weights
                    const int layerSynapseIdx = numNeuronPerColumn * numSynapsePerNeuron * columnIdx + columnSynapseIdx;
                    int outImgX = neuronIdx * params.rfSize + rfXIdx;
                    int outImgY = chanIdx * params.rfSize + rfYIdx;
                    outimg[outImgY * outImgWidth + outImgX] = static_cast<uint8_t>(hostWeights[layerSynapseIdx] * 256.0 / 8.0);
                }
            }
        }
    }
    free(hostWeights);
    return outimg;
}

uint8_t* convertToHostImg(layerParams& params) {
    // uint8_t* device_img;
    // const int imgSize = params.rfSize * params.rfSize * params.nNeurons * params.nPrevChan * params.outputDim * params.outputDim;
    //cudaMalloc(&device_img, imgSize);
    //uint8_t* host_img = (uint8_t *)malloc(imgSize);

    // TODO change kernel configuration
    int nXYthreads = static_cast<int>(floor(sqrt(1024.0/params.nNeurons/params.nPrevChan)));
    if (nXYthreads < 1) assert(false && "nNeuron and nPrevChan not supported");
    // weightsToImg<<<dim3(params.outputDim, params.outputDim), 
    //                 dim3(params.nNeurons, nXYthreads, nXYthreads * params.nPrevChan)>>>(params, device_img);
    //cudaMemcpy(host_img, device_img, imgSize, cudaMemcpyDeviceToHost);

    return weightsToImg_sequential(params);//host_img;
}
__global__  void spikesToImg (const uint8_t* spikes, uint8_t* device_img) {
    // const int columnIdx = blockIdx.y * gridDim.x + blockIdx.x; // for output

    // const int numNeuronPerColumn = blockDim.x;
    // const int rfSize = blockDim.y;
    // const int nChan = blockDim.z / rfSize;
    // const int numSynapsePerNeuron = rfSize * rfSize * nChan;
    // const int neuronIdx = threadIdx.x; //!< within a column
    // // synapseID (row major order, 2 channels for 1 pixel interleaves)
    // const int synapseID = threadIdx.y * blockDim.z + threadIdx.z; //!< within a neuron
    // // each synapse in each neuron has its own weights, updated across data samples
    // const int columnSynapseIdx = neuronIdx * numSynapsePerNeuron + synapseID;
    // const int layerSynapseIdx = numNeuronPerColumn * numSynapsePerNeuron * columnIdx + columnSynapseIdx;

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
    const int imgSize = params.rfSize * params.rfSize * params.nNeurons * params.nPrevChan * params.outputDim * params.outputDim;
    cudaMalloc(&device_img, imgSize);
    uint8_t* host_img = (uint8_t *)malloc(imgSize);

    spikesToImg<<<dim3(params.outputDim, params.outputDim, 12), 
                    2>>>(params.spike_time_out, device_img);
    cudaMemcpy(host_img, device_img, imgSize, cudaMemcpyDeviceToHost);
    cudaFree(device_img);
    return host_img;
}

void getConfusionMat(layerParams& params, uint8_t* labels, uint32_t* & confMat, int dataLength) {
// TODO build a confusion matrix: each row is a neuron, each column is a label
//      for each input image, at most 1 neuron could spike, we will inc the confMat[labelIdx, neuronIdx]
// TODO calculate purity and coverage
//      coverage = sum(confMat) / numLabels
//      purity = sum(max(entry in each row(same neuron))) / sum(confMat)
    const int gammaLength = 1 << 3; // wres = 3
    const int numClasses = 10; // 10 digits
    int totalSpikes = 0;
    confMat = (uint32_t*)calloc(params.nNeurons*numClasses, sizeof(uint32_t));
    size_t spikeTimeOutLength = params.nNeurons* params.outputDim * params.outputDim * dataLength;
    uint8_t * hostSpikeTimeOut = (uint8_t*)malloc(spikeTimeOutLength);
    cudaMemcpy(hostSpikeTimeOut, params.spike_time_out, spikeTimeOutLength, cudaMemcpyDeviceToHost);
    for (int dataIdx = 0; dataIdx < dataLength; ++dataIdx) {
        int correctLabel = labels[dataIdx];
        int spikingNeuronIdx = -1; // use negative number to represent no spikes
        for (int neuronIdx = 0; neuronIdx < params.nNeurons; ++neuronIdx) {
            if (hostSpikeTimeOut[dataIdx * params.nNeurons + neuronIdx] < gammaLength) {
                spikingNeuronIdx = neuronIdx;
                break;
            }
        }
        if (spikingNeuronIdx >= 0) { // has spike
            ++confMat[spikingNeuronIdx * numClasses + correctLabel];
            ++totalSpikes;
        }
    }
    float purity = 0;
    for (int neuronIdx = 0; neuronIdx < params.nNeurons; ++neuronIdx) {
        // int maxClass = 0;
        int maxCount = 0;
        for (int classIdx = 0; classIdx < numClasses; ++classIdx) {
            if (confMat[neuronIdx * numClasses + classIdx] > maxCount) {
                // maxClass = classIdx;
                maxCount = confMat[neuronIdx * numClasses + classIdx];
            }
        }
        purity += maxCount;
    }
    purity /= static_cast<float>(totalSpikes);
    float coverage = static_cast<float>(totalSpikes) / dataLength;
    printf("purity: %f\n", purity);
    printf("coverage: %f\n", coverage);
}

