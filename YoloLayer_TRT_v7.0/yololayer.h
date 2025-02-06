#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include <NvInfer.h>
#include <assert.h>
#include <iostream>
#include "macros.h"

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    static constexpr int KEY_POINTS_NUM = 17;

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };

    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 1;
    static constexpr int INPUT_H = 960;  // yolov5's input height and width must be divisible by 32.
    static constexpr int INPUT_W = 960;

    static constexpr int LOCATIONS = 4;
    struct Keypoint {
        float x;
        float y;
        float kpt_conf;
    };

    struct alignas(float) Detection {
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id; //person 0
        // 17 keypoints
        Keypoint kpts[KEY_POINTS_NUM];
    };

    static constexpr YoloKernel yolo1 = {
        INPUT_W / 64,
        INPUT_H / 64,
        {436.0f,615.0f, 739.0f,380.0f, 925.0f,792.0f}
    };
    static constexpr YoloKernel yolo2 = {
        INPUT_W / 32,
        INPUT_H / 32,
        {140.0f,301.0f, 303.0f,264.0f, 238.0f,542.0f}
    };
    static constexpr YoloKernel yolo3 = {
        INPUT_W / 16,
        INPUT_H / 16,
        {96.0f,68.0f, 86.0f,152.0f, 180.0f,137.0f}
    };
    static constexpr YoloKernel yolo4 = {
        INPUT_W / 8,
        INPUT_H / 8,
        {19.0f,27.0f, 44.0f,40.0f, 38.0f,94.0f}
    };


}

namespace nvinfer1
{
    class API YoloLayerPlugin : public IPluginV2DynamicExt 
    {
    public:
        YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin();

        int getNbOutputs() const TRT_NOEXCEPT override
        {
            return 1;
        }

        nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) TRT_NOEXCEPT override;

        int initialize() TRT_NOEXCEPT override;

        virtual void terminate() TRT_NOEXCEPT override {};

        size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override { return 0; }

        int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc*  outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

        virtual size_t getSerializationSize() const TRT_NOEXCEPT override;

        virtual void serialize(void* buffer) const TRT_NOEXCEPT override;

        bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override {
            return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kFLOAT;
        }

        const char* getPluginType() const TRT_NOEXCEPT override;

        const char* getPluginVersion() const TRT_NOEXCEPT override;

        void destroy() TRT_NOEXCEPT override;

        IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override;

        void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override;

        const char* getPluginNamespace() const TRT_NOEXCEPT override;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT override;

        // bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT override;

        // bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override;

        void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT override;

        void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInput,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutput) noexcept override;

        // using IPluginV2DynamicExt::configurePlugin;

        void detachFromContext() TRT_NOEXCEPT override;

    private:
        void forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize = 1);
        int mThreadCount = 256;
        const char* mPluginNamespace;
        int mKernelCount;
        int mClassCount;
        int mYoloV5NetWidth;
        int mYoloV5NetHeight;
        int mMaxOutObject;
        std::vector<Yolo::YoloKernel> mYoloKernel;
        void** mAnchor;
    };

    class API YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        const char* getPluginName() const TRT_NOEXCEPT override;

        const char* getPluginVersion() const TRT_NOEXCEPT override;

        const PluginFieldCollection* getFieldNames() noexcept override;

        IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

        IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT override;

        void setPluginNamespace(const char* libNamespace) TRT_NOEXCEPT override
        {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const TRT_NOEXCEPT override
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif  // _YOLO_LAYER_H