#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <cstring>
#include <algorithm>
#include <map>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.h"
//#include "FlattenLayer.h"  
//#include "ReshapeLayer.h"  

static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

// stuff we know about the network and the caffe input/detection_out blobs
static const int INPUT_C = 3;
static const int INPUT_H = 300;
static const int INPUT_W = 300;

static const int outputsize=1400;
//static const int OUTPUT_DETECTION_OUT = 1400;
static const int OUTPUT_DETECTION_OUT = outputsize;

const char* INPUT_BLOB_NAME0 = "data";
const char* OUTPUT_BLOB_NAME0 = "detection_out";
//const char* OUTPUT_BLOB_NAME0 = "mbox_conf_softmax";

void cudaSoftmax(int n, int channels,  float* x, float*y);
	//kernelSoftmax<<< (n/channels), channels, channels*sizeof(float)>>>( x, channels, y);
	//cudaDeviceSynchronize();

struct PPM
{
	std::string magic, fileName;
	int h, w, max;
	uint8_t buffer[INPUT_C*INPUT_H*INPUT_W];
};


std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/samples/ssd/", "data/ssd/"};
    return locateFile(input, dirs);
}

// simple PPM (portable pixel map) reader
void readPPMFile(const std::string& filename, PPM& ppm)
{
	ppm.fileName = filename;
	std::ifstream infile(locateFile(filename), std::ifstream::binary);
	infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

void caffeToGIEModel(const std::string& deployFile,			// name for caffe prototxt
	const std::string& modelFile,			// name for model 
	const std::vector<std::string>& detection_outs,		// network detection_outs
	unsigned int maxBatchSize,				// batch size - NB must be at least as large as the batch we want to run with)
	nvcaffeparser1::IPluginFactory* pluginFactory,	// factory for plugin layers
	IHostMemory **gieModelStream)			// detection_out stream for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the detection_outs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactory(pluginFactory);

	std::cout << "Begin parsing model..." << std::endl;

	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
		locateFile(modelFile).c_str(),
		*network,
		DataType::kFLOAT);
	std::cout << "End parsing model..." << std::endl;
	// specify which tensors are detection_outs
	for (auto& s : detection_outs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	for (int i = 0, n = network->getNbOutputs(); i < n; i++)
	{
		DimsCHW dims = static_cast<DimsCHW&&>(network->getOutput(i)->getDimensions());
		std::cout << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.c() << "x" << dims.h() << "x" << dims.w() << std::endl;
	}

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1024 << 20);	// we need about 6MB of scratch space for the plugin layer for batch size 5
    	builder->setHalf2Mode(false);

	std::cout << "Begin building engine..." << std::endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);
	std::cout << "End building engine..." << std::endl;

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	(*gieModelStream) = engine->serialize();

	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* inputData, float* detection_out, int batchSize) {
	const ICudaEngine& engine = context.getEngine();
	// input and detection_out buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly 2 inputs and 3 detection_outs.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and detection_out tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex0 = engine.getBindingIndex(INPUT_BLOB_NAME0),
		detection_outIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0);


	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex0], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));   // data
	CHECK(cudaMalloc(&buffers[detection_outIndex0], batchSize * OUTPUT_DETECTION_OUT * sizeof(float))); // bbox_pred

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	cudaEvent_t start, end;
	CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
	CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

	float total = 0, ms;
	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	cudaEventRecord(start, stream);
	context.enqueue(batchSize, buffers, stream, nullptr);
	cudaEventRecord(end, stream);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&ms, start, end);
	total += ms;
	std::cout << " runs  " << total << " ms." << std::endl;
	CHECK(cudaMemcpyAsync(detection_out, buffers[detection_outIndex0], batchSize * OUTPUT_DETECTION_OUT * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);


	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex0]));
	CHECK(cudaFree(buffers[detection_outIndex0]));
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

//Softmax layer.TensorRT softmax only support cross channel
class SoftmaxPlugin : public IPlugin
{
    //You need to implement it when softmax parameter axis is 2.
public:
    int initialize() override { return 0; }
    inline void terminate() override {}

    SoftmaxPlugin(){}
    SoftmaxPlugin( const void* buffer, size_t size)
    {
        assert(size == sizeof(mCopySize));
        mCopySize = *reinterpret_cast<const size_t*>(buffer);
    }
    inline int getNbOutputs() const override
    {
        //@TODO:  As the number of outputs are only 1, because there is only layer in top.
        return 1;
    }
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        assert(inputs[index].nbDims == 3);
//        assert((inputs[0].d[0])*(inputs[0].d[1]) % OutC == 0);

        // @TODO: Understood this.
        return DimsCHW( inputs[0].d[0] , inputs[0].d[1] , inputs[0].d[2] );
    }

    size_t getWorkspaceSize(int) const override
    {
        // @TODO: 1 is the batch size.
        return mCopySize*1;
    }

    int enqueue(int batchSize, const void*const *inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        std::cout<<"flatten enqueue:"<<batchSize<<";"<< mCopySize<<std::endl;
//        CHECK(cudaMemcpyAsync(outputs[0],inputs[0],batchSize*mCopySize*sizeof(float),cudaMemcpyDeviceToDevice,stream));

        cudaSoftmax( 8732*21, 21, (float *) *inputs, static_cast<float *>(*outputs));

        return 0;
    }

    size_t getSerializationSize() override
    {
        return sizeof(mCopySize);
    }
    void serialize(void* buffer) override
    {
        *reinterpret_cast<size_t*>(buffer) = mCopySize;
    }
    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
    {
        mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
    }

protected:
    size_t mCopySize;

};

class FlattenLayer: public IPlugin  
{  
  
protected:  
  
  DimsCHW dimBottom;  
  
  int _size;  
  
public:  
  
  FlattenLayer()  
  {  
  }  
  
  FlattenLayer(const void* buffer, size_t size)  
  {  
    assert(size == 3 * sizeof(int));  
  
    const int* d = reinterpret_cast<const int*>(buffer);  
  
    _size = d[0] * d[1] * d[2];  
  
    dimBottom = DimsCHW  
        { d[0], d[1], d[2] };  
  
  }  
  
  inline int getNbOutputs() const override  
  {  
    return 1;  
  }  
  ;  
  
  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override  
  {  
  
    assert(1 == nbInputDims);  
    assert(0 == index);  
    assert(3 == inputs[index].nbDims);  
  
    _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];  
  
    return DimsCHW(_size, 1, 1);  
  
  }  
  
  int initialize() override  
  {  
  
    return 0;  
  
  }  
  
  inline void terminate() override  
  {  
  
  }  
  
  inline size_t getWorkspaceSize(int) const override  
  {  
    return 0;  
  }  
  
  int enqueue(int batchSize, const void* const *inputs, void** detection_outs, void*, cudaStream_t stream) override  
  {  
  
    CHECK(cudaMemcpyAsync(detection_outs[0], inputs[0], batchSize * _size * sizeof(float), cudaMemcpyDeviceToDevice, stream));  
  
    return 0;  
  
  }  
  
  size_t getSerializationSize() override  
  {  
  
    return 3 * sizeof(int);  
  
  }  
  
  void serialize(void* buffer) override  
  {  
  
    int* d = reinterpret_cast<int*>(buffer);  
  
    d[0] = dimBottom.c();  
    d[1] = dimBottom.h();  
    d[2] = dimBottom.w();  
  
  }  
  
  void configure(const Dims*inputs, int nbInputs, const Dims* detection_outs, int nbOutputs, int) override  
  {  
  
    dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);  
  
  }  
  
};  
  

template<int OutC>  
class ReshapeLayer: public IPlugin  
{  
public:  
  ReshapeLayer()  
  {  
  }  
  ReshapeLayer(const void* buffer, size_t size)  
  {  
    assert(size == sizeof(mCopySize));  
    mCopySize = *reinterpret_cast<const size_t*>(buffer);  
  }  
  
  int getNbOutputs() const override  
  {  
    return 1;  
  }  
  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override  
  {  
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);  
    assert(nbInputDims == 1);  
    assert(index == 0);  
    assert(inputs[index].nbDims == 3);  
    assert((inputs[0].d[0]) * (inputs[0].d[1]) % OutC == 0);  
    return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);  
  }  
  
  int initialize() override  
  {  
    return 0;  
  }  
  
  void terminate() override  
  {  
  }  
  
  size_t getWorkspaceSize(int) const override  
  {  
    return 0;  
  }  
  
  // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the detection_out buffer  
  int enqueue(int batchSize, const void* const *inputs, void** detection_outs, void*, cudaStream_t stream) override  
  {  
    CHECK(cudaMemcpyAsync(detection_outs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));  
    return 0;  
  }  
  
  size_t getSerializationSize() override  
  {  
    return sizeof(mCopySize);  
  }  
  
  void serialize(void* buffer) override  
  {  
    *reinterpret_cast<size_t*>(buffer) = mCopySize;  
  }  
  
  void configure(const Dims*inputs, int nbInputs, const Dims* detection_outs, int nbOutputs, int) override  
  {  
    mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);  
  }  
  
protected:  
  size_t mCopySize;  
};  


struct Profiler : public IProfiler  
{  
    typedef std::pair<std::string, float> Record;  
    std::vector<Record> mProfile;  
  
    virtual void reportLayerTime(const char* layerName, float ms)  
    {  
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });  
  
        if (record == mProfile.end()) mProfile.push_back(std::make_pair(layerName, ms));  
        else record->second += ms;  
    }  
  
    void printLayerTimes(const int TIMING_ITERATIONS)  
    {  
        float totalTime = 0;  
        for (size_t i = 0; i < mProfile.size(); i++)  
        {  
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);  
            totalTime += mProfile[i].second;  
        }  
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);  
    }  
};  

class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory {
public:  
  
        // caffe parser plugin implementation  
        bool isPlugin(const char* name) override  
        { 
        return (!strcmp(name, "conv4_3_norm")  
            || !strcmp(name, "conv4_3_norm_mbox_conf_perm")  
            || !strcmp(name, "conv4_3_norm_mbox_conf_flat")  
            || !strcmp(name, "conv4_3_norm_mbox_loc_perm")  
            || !strcmp(name, "conv4_3_norm_mbox_loc_flat")  
            || !strcmp(name, "fc7_mbox_conf_perm")  
            || !strcmp(name, "fc7_mbox_conf_flat")  
            || !strcmp(name, "fc7_mbox_loc_perm")  
            || !strcmp(name, "fc7_mbox_loc_flat")  
            || !strcmp(name, "conv6_2_mbox_conf_perm")  
            || !strcmp(name, "conv6_2_mbox_conf_flat")  
            || !strcmp(name, "conv6_2_mbox_loc_perm")  
            || !strcmp(name, "conv6_2_mbox_loc_flat")  
            || !strcmp(name, "conv7_2_mbox_conf_perm")  
            || !strcmp(name, "conv7_2_mbox_conf_flat")  
            || !strcmp(name, "conv7_2_mbox_loc_perm")  
            || !strcmp(name, "conv7_2_mbox_loc_flat")  
            || !strcmp(name, "conv8_2_mbox_conf_perm")  
            || !strcmp(name, "conv8_2_mbox_conf_flat")  
            || !strcmp(name, "conv8_2_mbox_loc_perm")  
            || !strcmp(name, "conv8_2_mbox_loc_flat")  
            || !strcmp(name, "pool6_mbox_conf_perm")  
            || !strcmp(name, "pool6_mbox_conf_flat")  
            || !strcmp(name, "pool6_mbox_loc_perm")  
            || !strcmp(name, "pool6_mbox_loc_flat")  
            || !strcmp(name, "conv4_3_norm_mbox_priorbox")  
            || !strcmp(name, "fc7_mbox_priorbox")  
            || !strcmp(name, "conv6_2_mbox_priorbox")  
            || !strcmp(name, "conv7_2_mbox_priorbox")  
            || !strcmp(name, "conv8_2_mbox_priorbox")  
            || !strcmp(name, "pool6_mbox_priorbox")  
            || !strcmp(name, "mbox_conf_reshape")  
            || !strcmp(name, "mbox_conf_flatten")  
            || !strcmp(name, "mbox_loc")  
            || !strcmp(name, "mbox_conf")  
            || !strcmp(name, "mbox_priorbox")  
            || !strcmp(name, "detection_out")  
            ||  !strcmp(name, "detection_out2")  
        ||  !strcmp(name, "mbox_conf_softmax")  
        );  
  
        }  
  
        virtual IPlugin* createPlugin(const char* layerName, const Weights* weights, int nbWeights) override  
        {  
                // there's no way to pass parameters through from the model definition, so we have to define it here explicitly  
                if(!strcmp(layerName, "conv4_3_norm")){  
  
            //INvPlugin *   plugin::createSSDNormalizePlugin (const Weights *scales, bool acrossSpatial, bool channelShared, float eps)  
  
            _nvPlugins[layerName] = plugin::createSSDNormalizePlugin(weights,false,false,1e-10);  
  
            return _nvPlugins.at(layerName);  
  
                }else if(!strcmp(layerName, "conv4_3_norm_mbox_loc_perm")  
            ||  !strcmp(layerName, "conv4_3_norm_mbox_conf_perm")  
            ||  !strcmp(layerName,"fc7_mbox_loc_perm")  
            ||  !strcmp(layerName,"fc7_mbox_conf_perm")  
            ||  !strcmp(layerName,"conv6_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv6_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv7_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv7_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv8_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv8_2_mbox_conf_perm")  
            || !strcmp(layerName,"pool6_mbox_loc_perm")  
            || !strcmp(layerName,"pool6_mbox_conf_perm")  
        ){  
  
            _nvPlugins[layerName] = plugin::createSSDPermutePlugin(Quadruple({0,2,3,1}));  
  
            return _nvPlugins.at(layerName);  
  
                } else if(!strcmp(layerName,"conv4_3_norm_mbox_priorbox")){  
  
            plugin::PriorBoxParameters params = {0};  
            float minSize[1] = {30.0f};   
            //float maxSize[1] = {60.0f};   
            float aspectRatios[1] = {2.0f};   
            params.minSize = (float*)minSize;  
            //params.maxSize = (float*)maxSize;  
            params.aspectRatios = (float*)aspectRatios;  
            params.numMinSize = 1;  
            //params.numMaxSize = 1;  
            params.numAspectRatios = 1;  
            params.flip = true;  
            params.clip = false;  
            params.variance[0] = 0.10000000149;  
            params.variance[1] = 0.10000000149;  
            params.variance[2] = 0.20000000298;  
            params.variance[3] = 0.20000000298;  
            params.imgH = 0;  
            params.imgW = 0;  
            params.stepH = 8.0f;  
            params.stepW = 8.0f;  
            params.offset = 0.5f;  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"fc7_mbox_priorbox")){  
  
            plugin::PriorBoxParameters params = {0};  
            float minSize[1] = {60.0f};   
            float maxSize[1] = {114.0f};   
            float aspectRatios[2] = {2.0f, 3.0f};   
            params.minSize = (float*)minSize;  
            params.maxSize = (float*)maxSize;  
            params.aspectRatios = (float*)aspectRatios;  
            params.numMinSize = 1;  
            params.numMaxSize = 1;  
            params.numAspectRatios = 2;  
            params.flip = true;  
            params.clip = false;  
            params.variance[0] = 0.10000000149;  
            params.variance[1] = 0.10000000149;  
            params.variance[2] = 0.20000000298;  
            params.variance[3] = 0.20000000298;  
            params.imgH = 0;  
            params.imgW = 0;  
            params.stepH = 16.0f;  
            params.stepW = 16.0f;  
            params.offset = 0.5f;  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"conv6_2_mbox_priorbox")){  
            plugin::PriorBoxParameters params = {0};  
            float minSize[1] = {114.0f};   
            float maxSize[1] = {168.0f};   
            float aspectRatios[2] = {2.0f, 3.0f};   
            params.minSize = (float*)minSize;  
            params.maxSize = (float*)maxSize;  
            params.aspectRatios = (float*)aspectRatios;  
            params.numMinSize = 1;  
            params.numMaxSize = 1;  
            params.numAspectRatios = 2;  
            params.flip = true;  
            params.clip = false;  
            params.variance[0] = 0.10000000149;  
            params.variance[1] = 0.10000000149;  
            params.variance[2] = 0.20000000298;  
            params.variance[3] = 0.20000000298;  
            params.imgH = 0;  
            params.imgW = 0;  
            params.stepH = 32.0f;  
            params.stepW = 32.0f;  
            params.offset = 0.5f;  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"conv7_2_mbox_priorbox")){  
  
            plugin::PriorBoxParameters params = {0};  
            float minSize[1] = {168.0f};   
            float maxSize[1] = {222.0f};   
            float aspectRatios[2] = {2.0f, 3.0f};   
            params.minSize = (float*)minSize;  
            params.maxSize = (float*)maxSize;  
            params.aspectRatios = (float*)aspectRatios;  
            params.numMinSize = 1;  
            params.numMaxSize = 1;  
            params.numAspectRatios = 2;  
            params.flip = true;  
            params.clip = false;  
            params.variance[0] = 0.10000000149;  
            params.variance[1] = 0.10000000149;  
            params.variance[2] = 0.20000000298;  
            params.variance[3] = 0.20000000298;  
            params.imgH = 0;  
            params.imgW = 0;  
            params.stepH = 64.0f;  
            params.stepW = 64.0f;  
            params.offset = 0.5f;  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"conv8_2_mbox_priorbox")){  
  
            plugin::PriorBoxParameters params = {0};  
            float minSize[1] = {222.0f};   
            float maxSize[1] = {276.0f};   
            float aspectRatios[2] = {2.0f,3.0f};   
            params.minSize = (float*)minSize;  
            params.maxSize = (float*)maxSize;  
            params.aspectRatios = (float*)aspectRatios;  
            params.numMinSize = 1;  
            params.numMaxSize = 1;  
            params.numAspectRatios = 2;  
            params.flip = true;  
            params.clip = false;  
            params.variance[0] = 0.10000000149;  
            params.variance[1] = 0.10000000149;  
            params.variance[2] = 0.20000000298;  
            params.variance[3] = 0.20000000298;  
            params.imgH = 0;  
            params.imgW = 0;  
            params.stepH = 100.0f;  
            params.stepW = 100.0f;  
            params.offset = 0.5f;  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"pool6_mbox_priorbox")){  
            plugin::PriorBoxParameters params = {0};  
            float minSize[1] = {276.0f};   
            float maxSize[1] = {330.0f};   
            float aspectRatios[2] = {2.0f,3.0f};   
            params.minSize = (float*)minSize;  
            params.maxSize = (float*)maxSize;  
            params.aspectRatios = (float*)aspectRatios;  
            params.numMinSize = 1;  
            params.numMaxSize = 1;  
            params.numAspectRatios = 2;  
            params.flip = true;  
            params.clip = false;  
            params.variance[0] = 0.10000000149;  
            params.variance[1] = 0.10000000149;  
            params.variance[2] = 0.20000000298;  
            params.variance[3] = 0.20000000298;  
            params.imgH = 0;  
            params.imgW = 0;  
            params.stepH = 300.0f;  
            params.stepW = 300.0f;  
            params.offset = 0.5f;  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"detection_out")  
            ||!strcmp(layerName,"detection_out2")  
        ){  
            /*  
            bool    shareLocation  
            bool    varianceEncodedInTarget  
            int     backgroundLabelId  
            int     numClasses  
            int     topK  
            int     keepTopK  
            float   confidenceThreshold  
            float   nmsThreshold  
            CodeType_t  codeType  
            */  
            plugin::DetectionOutputParameters params = {0};  
            params.numClasses = 21;  
            params.shareLocation = true;  
            params.varianceEncodedInTarget = false;  
            params.backgroundLabelId = 0;  
            params.keepTopK = 200;  
            params.codeType = CENTER_SIZE;  
            params.nmsThreshold = 0.45;  
            params.topK = 400;  
            params.confidenceThreshold = 0.01;  
            _nvPlugins[layerName] = plugin::createSSDDetectionOutputPlugin (params);  
            return _nvPlugins.at(layerName);  
        }else if (  
            !strcmp(layerName, "conv4_3_norm_mbox_conf_flat")  
            ||!strcmp(layerName,"conv4_3_norm_mbox_loc_flat")  
            ||!strcmp(layerName,"fc7_mbox_loc_flat")  
            ||!strcmp(layerName,"fc7_mbox_conf_flat")  
            ||!strcmp(layerName,"conv6_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv6_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv7_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv7_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv8_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv8_2_mbox_loc_flat")  
            ||!strcmp(layerName,"pool6_mbox_conf_flat")  
            ||!strcmp(layerName,"pool6_mbox_loc_flat")  
            ||!strcmp(layerName,"mbox_conf_flatten")  
        ){  
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new FlattenLayer());  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf_reshape")){  
            _nvPlugins[layerName] = (plugin::INvPlugin*)new ReshapeLayer<21>();  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf_softmax")){  
            _nvPlugins[layerName] = (plugin::INvPlugin*)new SoftmaxPlugin();  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_loc")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (1,false);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (1,false);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_priorbox")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (2,false);  
            return _nvPlugins.at(layerName);  
	}else {  
            assert(0);  
            return nullptr;  
        }  
    }  
  
    // deserialization plugin implementation  
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override {
        if(!strcmp(layerName, "conv4_3_norm"))  
        {  
            _nvPlugins[layerName] = plugin::createSSDNormalizePlugin(serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(  
            !strcmp(layerName, "conv4_3_norm_mbox_loc_perm")  
            || !strcmp(layerName, "conv4_3_norm_mbox_conf_perm")  
            || !strcmp(layerName,"fc7_mbox_loc_perm")  
            || !strcmp(layerName,"fc7_mbox_conf_perm")  
            || !strcmp(layerName,"conv6_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv6_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv7_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv7_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv8_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv8_2_mbox_conf_perm")  
            || !strcmp(layerName,"pool6_mbox_loc_perm")  
            || !strcmp(layerName,"pool6_mbox_conf_perm")  
        ){  
            _nvPlugins[layerName] = plugin::createSSDPermutePlugin(serialData, serialLength);  
            return _nvPlugins.at(layerName);  
  
        }else if(!strcmp(layerName,"conv4_3_norm_mbox_priorbox")  
            || !strcmp(layerName,"fc7_mbox_priorbox")     
            || !strcmp(layerName,"conv6_2_mbox_priorbox")  
            || !strcmp(layerName,"conv7_2_mbox_priorbox")  
            || !strcmp(layerName,"conv8_2_mbox_priorbox")  
            || !strcmp(layerName,"pool6_mbox_priorbox")  
        ){  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"detection_out")  
            || !strcmp(layerName,"detection_out2")  
            ){  
            _nvPlugins[layerName] = plugin::createSSDDetectionOutputPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf_reshape")){  
            _nvPlugins[layerName] = (plugin::INvPlugin*)new ReshapeLayer<21>(serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf_softmax")){  
            _nvPlugins[layerName] = (plugin::INvPlugin*)new SoftmaxPlugin(serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if (  
            !strcmp(layerName, "conv4_3_norm_mbox_conf_flat")  
            ||!strcmp(layerName,"conv4_3_norm_mbox_loc_flat")  
            ||!strcmp(layerName,"fc7_mbox_loc_flat")  
            ||!strcmp(layerName,"fc7_mbox_conf_flat")  
            ||!strcmp(layerName,"conv6_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv6_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv7_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv7_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv8_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv8_2_mbox_loc_flat")  
            ||!strcmp(layerName,"pool6_mbox_conf_flat")  
            ||!strcmp(layerName,"pool6_mbox_loc_flat")  
            ||!strcmp(layerName,"mbox_conf_flatten")  
        ){  
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new FlattenLayer(serialData, serialLength));  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_loc")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_priorbox")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else{  
            assert(0);  
            return nullptr;  
        }  
    }  
  
  
    void destroyPlugin()  
    {  
        for (auto it=_nvPlugins.begin(); it!=_nvPlugins.end(); ++it){  
            it->second->destroy();  
            _nvPlugins.erase(it);  
        }  
    }  
  
  
private:  
  
        std::map<std::string, plugin::INvPlugin*> _nvPlugins;   
};  


int main(int argc, char** argv)
{
	// create a GIE model from the caffe model and serialize it to a stream
	PluginFactory pluginFactory;
	IHostMemory *gieModelStream{ nullptr };
	// batch size
	const int N = 1;
	caffeToGIEModel("ssd_deploy_iplugin.prototxt",
		"VGG_VOC0712_SSD_300x300_iter_60000.caffemodel",
		std::vector < std::string > { OUTPUT_BLOB_NAME0},
		N, &pluginFactory, &gieModelStream);
	string label_file = "/usr/src/tensorrt/data/ssd/label_map.txt";
	std::vector<string> labels_;
	   {
	     std::ifstream labels(label_file);
	     string line;
	     while (std::getline(labels, line)) {
	       labels_.push_back(string(line));
	     }
	     labels.close();
	   }
	//pluginFactory.destroyPlugin();
	// read a random sample image
	srand(unsigned(time(nullptr)));
	// available images 
	std::vector<std::string> imageList = { "000456.ppm"};
	std::vector<PPM> ppms(N);

	std::random_shuffle(imageList.begin(), imageList.end(), [](int i) {return rand() % i; });
	assert(ppms.size() <= imageList.size());

	for (int i = 0; i < N; ++i)
	{
		readPPMFile(imageList[i], ppms[i]);
	}

	float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];
	float pixelMean[3]{ 104.0f,117.0f,123.0f }; // also in BGR order
	for (int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < N; ++i)
	{
		for (int c = 0; c < INPUT_C; ++c)
		{
			// the color image to input should be in BGR order
			for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j){
				data[i*volImg + c*volChl + j] = float(ppms[i].buffer[j*INPUT_C + 2 - c]) - pixelMean[c];
				//std::cout << "input:" << float(ppms[i].buffer[j*INPUT_C + 2 - c]) - pixelMean[c]<< std::endl;
			}
		}
	}

	// deserialize the engine 
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);

	IExecutionContext *context = engine->createExecutionContext();


	// host memory for detection_outs 
	float* detection_out = new float[N * OUTPUT_DETECTION_OUT];


	// run inference
	doInference(*context, data, detection_out, N);
	std::cout << "after inference:"<< std::endl;

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	//pluginFactory.destroyPlugin();
	 
	//for (int i = 0; i < N; ++i)
	//{
    //    	for (int k=0; k<outputsize; k++)
    //    	{
    //    	    std::cout << detection_out[k] << "\n";
	//	}
	//}

#if 1
	for (int i = 0; i < N; ++i)
	{
		cv::Mat temp = cv::Mat(INPUT_W,INPUT_H,CV_8UC3,ppms[i].buffer);
        	for (int k=0; k<10; k++)
        	{
				if (detection_out[7*k+2] > 0.6){
        	    	std::cout << detection_out[7*k+0] << " "
        	    	          << detection_out[7*k+1] << " "
        	    	          << detection_out[7*k+2] << " "
        	    	          << detection_out[7*k+3] << " "
        	    	          << detection_out[7*k+4] << " "
        	    	          << detection_out[7*k+5] << " "
        	    	          << detection_out[7*k+6] << "\n";
        	    	if(detection_out[7*k+1] == -1) continue;
        	    	float xmin = 300 * detection_out[7*k + 3];
        	    	float ymin = 300 * detection_out[7*k + 4];
        	    	float xmax = 300 * detection_out[7*k + 5];
        	    	float ymax = 300 * detection_out[7*k + 6];
        	    	using cv::Point2f;
        	    	using cv::line;
        	    	using cv::Scalar;
        	    	Point2f a = Point2f(xmin, ymin);
        	    	Point2f b = Point2f(xmin, ymax);
        	    	Point2f c = Point2f(xmax, ymax);
        	    	Point2f d = Point2f(xmax, ymin);
        	    	line(temp, a, b, Scalar(0.0, 255.0, 255.0));
        	    	line(temp, b, c, Scalar(0.0, 255.0, 255.0));
        	    	line(temp, c, d, Scalar(0.0, 255.0, 255.0));
        	    	line(temp, d, a, Scalar(0.0, 255.0, 255.0));
        	    	std::cout << xmin << ", " << ymin << ", " << xmax << ", " << ymax << "\n";
					std::stringstream s0;
					s0 << detection_out[7*k+2];
					string s1 = s0.str();
					cv::putText(temp, labels_[detection_out[7*k+1] - 1], cv::Point(xmin, ymin+10), 1, 1, cv::Scalar(0, 255, 255), 0, 8, 0);
                    cv::putText(temp, s1.c_str(), cv::Point(xmin, ymin+20), 1, 1, cv::Scalar(0, 255, 255), 0, 8, 0);
				}
			}
        	cv::imshow("Objects Detected", temp);
        	cv::waitKey(10000);
	}
#endif

	delete[] data;
	delete[] detection_out;
	return 0;
}
