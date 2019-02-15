#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include "NvInfer.h"

#include <signal.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <chrono>

#include "cudaResize.h"
#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

bool signalRecieved = false;

void signalHandler( int sigNo )
{
  if( sigNo == SIGINT )
  {
    signalRecieved = true;
  }
}

class Logger : public nvinfer1::ILogger
{
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override
  {
    std::cout << msg << std::endl;
  }
} gLogger;


int main( int argc, char** argv )
{
  const int   DEFAULT_CAMERA = -1;
  const int   MAX_BATCH_SIZE = 1;
  const char* MODEL_NAME     = "test_an.trt";
  const char* INPUT_BLOB     = "Placeholder";
  const char* OUTPUT_BLOB    = "w_out/dense_3/BiasAdd";

  if( signal(SIGINT, signalHandler) == SIG_ERR )
  { 
    std::cout << "\ncan't catch SIGINT" <<std::endl;
  }
 
  gstCamera* camera = gstCamera::Create(320, 240, DEFAULT_CAMERA);
	
  if( !camera )
  {
    std::cout << "\nsegnet-camera:  failed to initialize video device" << std::endl;
    return 0;
  }
	
  std::cout << "\nsegnet-camera:  successfully initialized video device" << std::endl;
  std::cout << "    width:  " << camera->GetWidth() << std::endl;
  std::cout << "   height:  " << camera->GetHeight() << std::endl;
  std::cout << "    depth:  "<< camera->GetPixelDepth() << std::endl;

  float* outCPU  = nullptr;
  float* outCUDA = nullptr;

  if( !cudaAllocMapped( (void**)&outCPU, (void**)&outCUDA, camera->GetWidth() * camera->GetHeight() * sizeof(float) * 4 ) )
  {
    std::cout << "Failed to allocate CUDA memory" << std::endl;
    return 0;
  }
  
//Read model from file, deserialize it and create runtime, engine and exec context
  std::ifstream model( MODEL_NAME, std::ios::binary );

  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(model), {});
  std::size_t modelSize = buffer.size() * sizeof( unsigned char );

  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine( buffer.data(), modelSize, nullptr );
  nvinfer1::IExecutionContext* context = engine->createExecutionContext(); 

  if( !context )
  {
    std::cout << "Filed to create execution context" << std::endl;
    return 0;
  }


//Set input info + alloc memory
  const int inputIndex = engine->getBindingIndex( INPUT_BLOB );
  nvinfer1::Dims inputDims = engine->getBindingDimensions( inputIndex );
  std::cout << "-  Input binding index: " << inputIndex << std::endl;
  std::cout << "- Number of dimensions: " << inputDims.nbDims << std::endl;
  std::cout << "- c: " << DIMS_C(inputDims) << " h: " << DIMS_H(inputDims) << " w: " << DIMS_W(inputDims) << std::endl;

  std::size_t inputSize = MAX_BATCH_SIZE * DIMS_C(inputDims) * DIMS_H(inputDims) * DIMS_W(inputDims) * sizeof(float);

  void* inputCPU  = nullptr;
  void* inputCUDA = nullptr;

  if( !cudaAllocMapped( (void**)&inputCPU, (void**)&inputCUDA, inputSize ) )
  {
    std::cout << "Failed to alloc CUDA memory for input" << std::endl;
    return 0;
  }

//Set output info + alloc memory 
  void* outputCPU  = nullptr;
  void* outputCUDA = nullptr;


  const int outputIndex = engine->getBindingIndex( OUTPUT_BLOB );
  nvinfer1::Dims outputDims = engine->getBindingDimensions( outputIndex );
  std::size_t outputSize = MAX_BATCH_SIZE * DIMS_C(outputDims) * DIMS_H(outputDims) * DIMS_W(outputDims) * sizeof(float);

  std::cout << "- Output binding index: " << outputIndex << std::endl;
  std::cout << "- Number of dimensions: " << outputDims.nbDims << std::endl;
  std::cout << "- c: " << DIMS_C(outputDims) << " h: " << DIMS_H(outputDims) << " w: " << DIMS_W(outputDims) << std::endl;

  if( !cudaAllocMapped( (void**)&outputCPU, (void**)&outputCUDA, outputSize ) )
  {
    std::cout << "Failed to alloc CUDA memory for output" << std::endl;
    return 0;
  }

//Create CUDA stream for GPU inner data sync
  cudaStream_t stream = nullptr;
  CUDA_FAILED( cudaStreamCreateWithFlags(&stream, cudaStreamDefault ) );




  if( !camera->Open() )
  {
    std::cout << "Failed to open camera" << std::endl;
  }

  std::cout << "Camera is open for streaming" << std::endl;

  

// Main loop
  while( !signalRecieved )
  {
    auto start = std::chrono::high_resolution_clock::now();

    void* imgCPU  = nullptr;
    void* imgCUDA = nullptr;

    if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
    {
      std::cout << "Failed to capture frame" << std::endl;
    }

//    std::cout << "Captured frame" << std::endl;

    void* imgRGBA = nullptr;

    if( !camera->ConvertRGBA(imgCUDA, &imgRGBA, true) )
    {
      std::cout << "Failed to convert from NV12 to RGBA" << std::endl;
    }

//    preprocess( imgCUDA, DIMS_W(inputDims), DIMS_H(inputDims), inputCUDA, DIMS_W(inputDims), DIMS_H( inputDims ), cudaStream );

    cudaResizeRGBA( (float4*)imgCUDA, camera->GetWidth(), camera->GetHeight(), (float4*)inputCUDA, DIMS_W(inputDims), DIMS_H(inputDims) );
    void* bindings[] = {inputCUDA, outputCUDA};
    context->execute( MAX_BATCH_SIZE, bindings );

    //std::cout << "Converted frame"  << std::endl;
 
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Time to process single frame: " << elapsed.count() << std::endl;
  }

  if( camera != nullptr )
  {
    camera->Close();
    delete camera;
  }


  return 0;
}
