#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <errno.h>
#include <libgen.h>
#include "neuron.h"
#include <cassert>
#include <arpa/inet.h> // for endianness conversion
#include <string>
#include "CycleTimer.h"
void printCudaInfo();

// return GB/s
float toBW(int bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

void usage(const char *progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}

bool assertAtFileEnd(FILE * file) {
    int unused;
    int elemRead = fread(&unused, 1, 1, file);
    return elemRead == 0 && feof(file);
}
void outputToBitmap(int x, int y, uint8_t* img, char* outputFile){
    FILE* outputfp = fopen(outputFile, "w");
    assert(outputfp != NULL);
    struct BMPhdr {
        char magic[2] = {'B', 'M'};
        uint32_t bmpSize; // file size
        uint16_t reserved_1;
        uint16_t reserved_2;
        uint32_t offset; // offset to start of pixel data
        BMPhdr(uint32_t bmpSize, uint32_t offset) : bmpSize(bmpSize), offset(offset){}
    }__attribute__((packed));
    static_assert(sizeof(BMPhdr) == 14, "wrong packed size for BMP header");
    struct BITMAPINFOHEADER {
        const uint32_t size = 40;
        int32_t bmpWid;
        int32_t bmpHei;
        const uint16_t nColorPlanes = 1;
        const uint16_t nbpp = 8; // bits per pixel
        const uint32_t compressMtd = 0; //BI_RGB
        const uint32_t imgSize = 0; //dummy image size for BI_RGB
        const int32_t horiz_ppm = 4000; //pixel per meter, set to equivalent of 22 in 1080p according to https://www.sven.de/dpi/
        const int32_t verti_ppm = 4000;
        const uint32_t nPaletteColors = 0; // default to 2^n
        const uint32_t nImportantColors = 0;
        BITMAPINFOHEADER(int32_t bmpWid, int32_t bmpHei) : bmpWid(bmpWid), bmpHei(bmpHei){}
    }__attribute__((packed));
    static_assert(sizeof(BITMAPINFOHEADER) == 40, "wrong packed size for BMP map info header");
    struct colorTableEntry {
        uint8_t b;
        uint8_t g;
        uint8_t r;
        const uint8_t _pad = 0;
    };

    static_assert(sizeof(colorTableEntry) == 4, "wrong size for color table entry");
    BITMAPINFOHEADER infohdr{x, -y};

    int nColorTableEntry = 1 << infohdr.nbpp;
    colorTableEntry colortable[nColorTableEntry];
    for(int i = 0; i < nColorTableEntry; ++i){
        colortable[i].b = (uint8_t)i;
        colortable[i].g = (uint8_t)i;
        colortable[i].r = (uint8_t)i;
    }
    int nPadBytes = x % 4; // number of bytes to pad for each row
    if(nPadBytes) nPadBytes = 4 - nPadBytes;
    uint32_t pixelArrSize = (x + nPadBytes) * y;
    uint32_t pixelDataOffset = sizeof(BMPhdr) + sizeof(BITMAPINFOHEADER) + sizeof(colorTableEntry) * nColorTableEntry;
    uint32_t nPadBeforePixelArr = ((pixelDataOffset + 3) / 4) * 4 - pixelDataOffset;
    BMPhdr bmphdr{pixelDataOffset + nPadBeforePixelArr + pixelArrSize, pixelDataOffset + nPadBeforePixelArr};
    fwrite(&bmphdr, sizeof(BMPhdr), 1, outputfp);
    fwrite(&infohdr, sizeof(BITMAPINFOHEADER), 1, outputfp);
    fwrite(&colortable, sizeof(colorTableEntry), nColorTableEntry, outputfp);
    int zero = 0;
    fwrite(&zero, 1, nPadBeforePixelArr, outputfp);
    for(int i = 0; i < y; ++i){
        fwrite(&img[i * x], 1, x, outputfp);
        fwrite(&zero, 1, nPadBytes, outputfp);
    }
    fclose(outputfp);
}

/**
 * @brief 
 * 
 * @param[in] input_file 
 * @param[out] spike_time_in 
 */
void load_MNIST(const char* image_input_file, const char* label_input_file, uint8_t* & spike_time_in, uint8_t* & labels) {
    const uint32_t IDX3_MAGIC = 0x803;
    const uint32_t IDX1_MAGIC = 0x801;
    // Load input MNIST dataset and cudaMemcpy
    // CudaMalloc spike_time_in and spike_time_out
    // MNIST_loader_kernel
    FILE * mnistImageFile = fopen(image_input_file, "r");
    int err = errno;
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

    size_t nImageBytes = nImgs * nRows * nCols; // Each pixel has 1 byte.
    uint8_t* imgData = (uint8_t*) malloc(nImageBytes);
    assert(imgData != NULL);
    elemRead = fread(imgData, sizeof(uint8_t), nImageBytes, mnistImageFile);
    printf("nImageBytes %lu; elemRead %lu; \nnImgs %d nRows %d nCols %d\n", nImageBytes, elemRead, nImgs, nRows, nCols);
    assert(elemRead == nImageBytes);
    assert(assertAtFileEnd(mnistImageFile));
    fclose(mnistImageFile);
    outputToBitmap(28, 28*12, imgData, "mnistDirect.bmp");
    // copy pixel data to device and launch kernel to convert
    launch_load_MNIST(nImgs, nRows, nCols, imgData, spike_time_in);
    free(imgData);

    // Read label data into device memory as well for later classification
    elemRead = fread(&magic, sizeof(uint32_t), 1, mnistLabelFile);
    assert(elemRead == 1);
    magic = ntohl(magic);
    assert(magic == IDX1_MAGIC);
    
    int nLabels;
    elemRead = fread(&nLabels, sizeof(uint32_t), 1, mnistLabelFile);
    assert(elemRead == 1);
    nLabels = ntohl(nLabels);


    //uint8_t * labelData 
    labels = (uint8_t *)malloc(nLabels);
    elemRead = fread(labels /*labelData*/, 1, nLabels, mnistLabelFile);
    assert(elemRead == (size_t)nLabels);
    assert(assertAtFileEnd(mnistLabelFile));
    fclose(mnistLabelFile);
/*
    copyLabelToDevice(nLabels, labelData, labels);
    free(labelData);*/
    
}

int main(int argc, char **argv) {
    using std::string;
    static_assert(sizeof(int) == 4, "int is not 32 bit");
    int dataLength = 10000;
    uint8_t* spike_time_in = NULL;
    uint8_t* labels = NULL;
    string dirpath = dirname(argv[0]);
    dirpath += string("/");
    string mnistImgPath = "data/train-images-idx3-ubyte";
    string mnistLabelPath = "data/train-labels-idx1-ubyte";
    // setup constants
    setup();
    layerParams layers[3];
    layers[0] = {.inputDim = 28, .rfSize = 28, .stride = 1, .nNeurons = 12, .nPrevChan = 2};
    layers[0].stdpEn = true;
    load_MNIST((dirpath+mnistImgPath).c_str(), (dirpath+mnistLabelPath).c_str(), spike_time_in, labels); // perform parallel load
    // layers[0].outputDim = 28;
    // layers[0].nNeurons = 12;
    // layers[0].stdpEn = true;
    // layers[0].spike_time_out = spike_time_in;
    // outputToBitmap(28*12, 28*2, convertSpikesToHostImg(layers[0]), "mnistSpikeDirect3.bmp");
    double startSeconds = CycleTimer::currentSeconds();
    launch_column(layers[0], dataLength, spike_time_in);
    double endSeconds = CycleTimer::currentSeconds();
    printf("Training time: %lf\n", endSeconds - startSeconds);
    cudaFreeHostWrap(layers[0].spike_time_out);

    int outputxsize = layers[0].rfSize * layers[0].nNeurons;
    int outputysize = layers[0].rfSize * layers[0].nPrevChan * layers[0].outputDim * layers[0].outputDim;
    outputToBitmap(outputxsize, outputysize, convertToHostImg(layers[0]), "weights.bmp");
    
    string mnistTestImgPath = "data/t10k-images-idx3-ubyte";
    string mnistTestLabelPath = "data/t10k-labels-idx1-ubyte";
    free(labels);
    layers[0].stdpEn = false;
    load_MNIST((dirpath+mnistTestImgPath).c_str(), (dirpath+mnistTestLabelPath).c_str(), spike_time_in, labels); // perform parallel load
    startSeconds = CycleTimer::currentSeconds();
    launch_column(layers[0], dataLength, spike_time_in);
    endSeconds = CycleTimer::currentSeconds();
    printf("Testing time: %lf\n", endSeconds - startSeconds);
    uint32_t* confMat;
    getConfusionMat(layers[0], labels, confMat, dataLength);
    for(int neuronIdx = 0; neuronIdx < layers[0].nNeurons; ++neuronIdx) {
        for(int classIdx = 0; classIdx < 10; ++classIdx){
            printf("%d, ",confMat[neuronIdx * 10 + classIdx]);
        }
        printf("\n");
    }
}
