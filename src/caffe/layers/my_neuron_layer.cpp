#include <vector>
#include "caffe/layers/my_neuron_layer.hpp"
//#include "my_neuron_layer.hpp"
#include "caffe/util/math_functions.hpp"
//#define PRINT_BLOB_SHAPE YS
namespace caffe {

template <typename Dtype>
void MyNeuronLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
    //调用父类的LayerSetUp函数，实现从prototxt读参数。
    NeuronLayer<Dtype>::LayerSetUp(bottom,top);
    //1、layer_param_是基类Layer层的成员变量，在caffe.proto可以看到，类型为LayerParameter。
    //2、在步骤3中，我们在LayerParameter中增加了参数my_neuron_param，类型为MyNeuronParameter。
    //3、MyNeuronParameter类型也是为该层添加的，里面定义了参数power。
    power_ = this->layer_param_.my_neuron_param().power();
    }
// Compute y = x^power
template <typename Dtype> void MyNeuronLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
    //bottom和top都是一组向量，向量的元素是Blob类型的指针。
    //Forward操作要更新top值，所以这里用top的mutable_cpu_data数据指针。因为我们定义的网络结构只有一个top，所以只使用top[0]。
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    const int num_batch = bottom[0]->num();
    const int ch = bottom[0]->channels();
    const int width = bottom[0]->width();
    const int height = bottom[0]->height();



    //caffe_powx在math_functions.hpp中定义
    /*
    void caffe_powx<float>(const int n, const float* a, const float b, float* y)
    作用：y[i] = a[i] ^ b,n表示元素个数
  */
#ifdef PRINT_BLOB_SHAPE
    LOG(INFO) << "bottom[0]->count()" << count;
#endif
    caffe_powx(count, bottom[0]->cpu_data(), Dtype(power_), top_data);


#ifdef PRINT_BLOB_SHAPE
    for(int i=0;i<count;i++)
    {

        LOG(INFO)<< "numbatch "<< num_batch;
        LOG(INFO)<< "ch "<< ch;
        LOG(INFO)<< "width "<<width;
        LOG(INFO)<< "height "<<height;


        LOG(INFO)<< top_data[i];
        LOG(INFO) << " top_data is" << count;

        LOG(INFO)<< top_data[i];
    }
#endif
}
template <typename Dtype>
void MyNeuronLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
    //反向传播：由top计算出bottom，更进一步是由top的cpu_diff对bottom的cpu_data求导。cpu_diff是实际误差，bo
    const int count = top[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    if(propagate_down[0]){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        //y = power * x ^ (power-1)
        //bottom_diff表示梯度下降算法的权值更新的量
        caffe_powx(count, bottom_data, Dtype(power_ - 1), bottom_diff);
        caffe_scal(count, Dtype(power_), bottom_diff); //
        caffe_mul(count, bottom_diff, top_diff, bottom_diff);
    }
}
#ifdef CPU_ONLY STUB_GPU(MyNeuronLayer);
#endif
//实例化模板类MyNeuronLayer
INSTANTIATE_CLASS(MyNeuronLayer);
REGISTER_LAYER_CLASS(MyNeuron);
}// namespace caffe

