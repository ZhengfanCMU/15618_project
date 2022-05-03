#include <stdint.h>
struct layerParams {
    int inputDim;
    int rfSize;
    int stride; //stride only supports 1 for now
    int nNeurons; // number of neurons per column / number of output channels
    int nPrevChan; // prev layer's nNeurons / number of input channels

    // Derived parameters
    int outputDim; // Dimension of the output image
    int nSynapsesPerNeuron;
    
    // Launch parameters
    int xBatchSize;
    int yBatchSize;

    // Records layer data pointers
    // (device addresses)
    uint8_t* spike_time_out;
    float* weights;
};
uint8_t* convertSpikesToHostImg(layerParams& params);
void launch_load_MNIST(int nImgs, int nRows, int nCols, uint8_t* imgData, uint8_t* & spike_time_in);
void copyLabelToDevice(int nLabels, uint8_t * labelData, uint8_t *& labels);
void launch_column(layerParams& params, int dataLength, uint8_t* spike_time_in);
void setup();
uint8_t* convertToHostImg(layerParams& params);

