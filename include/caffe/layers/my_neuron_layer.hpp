#ifndef MY_NEURON_LAYER_HPP
#define MY_NEURON_LAYER_HPP
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
//继承自neuron_layer
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {
template <typename Dtype> //该自定义层继承自NeuronLayer层。
class MyNeuronLayer : public NeuronLayer<Dtype> {
public: explicit MyNeuronLayer(const LayerParameter& param) :
        NeuronLayer<Dtype>(param) {}
    //声明LayerSetUp层
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    //重新设置Layer名
    virtual inline const char* type() const { return "MyNeuron"; }
protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 //   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 //   virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    Dtype power_;//需要定义一个成员变量power_作为指数运算的幂。
};

} // namespace caffe


#endif // MY_NEURON_LAYER_HPP
