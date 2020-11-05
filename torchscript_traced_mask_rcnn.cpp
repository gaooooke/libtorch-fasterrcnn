// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <math.h>
#include <typeinfo>
#include <cassert>
#include <chrono>
// #include <iomanip>
#include <ctime>

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/script.h>
#include <torch/types.h>
#include <ATen/Tensor.h>
#include <ATen/Context.h>
#include <ATen/ATen.h>

using namespace std;
using namespace cv;

// experimental. don't use
int main(int argc, const char* argv[]) {
  if (argc != 3) {
    return 1;
  }
  std::string image_file = argv[2];

  torch::autograd::AutoGradMode guard(false);
  auto module = torch::jit::load(argv[1]);

  assert(module.buffers().size() > 0);
  // Assume that the entire model is on the same device.
  // We just put input to this device.
  auto device = (*begin(module.buffers())).device();

  string classes[4] = {"hair","nohair","clothes","nohat"};

  cv::Mat input_img = cv::imread(image_file, cv::IMREAD_COLOR);
  // cv::imshow("source",input_img);
  // cv::waitKey(0);
  int height = input_img.rows;
  int width = input_img.cols;

  // Padding
  int padh = 0,padw = 0;
  double h = height, w = width;
  if(height % 32 !=0 || width % 32 != 0)
  {
    padh = 32 * (ceil(h/32) - h/32);
    padw = 32 * (ceil(w/32) - w/32);
    cv::copyMakeBorder(input_img,input_img,ceil(padh/2),padh-ceil(padh/2),ceil(padw/2),padw-ceil(padw/2),BORDER_CONSTANT,Scalar(128,128,128));
  }

  // cv::imshow("padding",input_img);
  // cv::waitKey(0);
  height = input_img.rows;
  width = input_img.cols;

  // FPN models require divisibility of 32
  assert(height % 32 == 0 && width % 32 == 0);
  const int channels = 3;

  auto input = torch::from_blob(
      input_img.data, {1, height, width, channels}, torch::kUInt8);
  
  // NHWC to NCHW
  input = input.to(device, torch::kFloat).permute({0, 3, 1, 2}).contiguous();
  std::array<float, 3> im_info_data{height * 1.0f, width * 1.0f, 1.0f};
  auto im_info = torch::from_blob(im_info_data.data(), {1, 3}).to(device);
  // run the network
  auto output = module.forward({std::make_tuple(input, im_info)});
  if (device.is_cuda())
    c10::cuda::getCurrentCUDAStream().synchronize();
  
  // run 3 more times to benchmark
  int N_benchmark = 3;
  auto start_time = chrono::high_resolution_clock::now();
  for (int i = 0; i < N_benchmark; ++i) {
    output = module.forward({std::make_tuple(input, im_info)});
    if (device.is_cuda())
      c10::cuda::getCurrentCUDAStream().synchronize();
  }
  auto end_time = chrono::high_resolution_clock::now();
  auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time)
                .count();
  cout << "Latency (should vary with different inputs): "
       << ms * 1.0 / 1e6 / N_benchmark << " seconds" << endl;

  auto outputs = output.toTuple()->elements();
  // parse Mask R-CNN outputs
  auto bbox   = outputs[0].toTensor(), 
       scores = outputs[1].toTensor(),
       labels = outputs[2].toTensor();  
       // mask_probs = outputs[3].toTensor();
  
  // cout << "bbox: " << bbox.toString() << " " << bbox.sizes() << endl;
  // cout << "scores: " << scores.toString() << " " << scores.sizes() << endl;
  // cout << "labels: " << labels.toString() << " " << labels.sizes() << endl; 
  // cout << "mask_probs: " << mask_probs.toString() << " " << mask_probs.sizes()
  //      << endl;
  
  int num_instances = bbox.sizes()[0];
  // cout << num_instances << bbox << scores << labels << endl;
  int label;
  float score = 0.0;
  float p1,p2,p3,p4 = 0;

  RNG rng((unsigned)time(NULL));
  // RNG rng(500);
  // draw rectangle
  for (int i=0;i<num_instances;++i)
  {
    label = labels[i].item().toInt();
    score = scores[i].item().toFloat();
    
    if (score < 0.8)  // Threshold
      continue; // skip them
    p1 = bbox[i][0].item().toFloat();
    p2 = bbox[i][1].item().toFloat();
    p3 = bbox[i][2].item().toFloat();
    p4 = bbox[i][3].item().toFloat();

    // clip
    if(p1<0) p1=0;
    if(p2<0) p2=0;
    if(p3>w) p3=w;
    if(p4>h) p4=h;

    char strScore[10] = {0};
    snprintf(strScore, 9, "%.3f", score);
    int baseline = 0;
    Size textSize = cv::getTextSize(classes[label]+" : "+string(strScore),cv::FONT_HERSHEY_SIMPLEX,0.4,1,&baseline);
    cv::rectangle(input_img,Point(p1,p2),Point(p3,p4),Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)),3,1,0);
    cv::rectangle(input_img,Point(p1,p2-textSize.height-10),Point(p1+textSize.width,p2+textSize.height-12),Scalar(0,255,255),CV_FILLED,1,0);
    cv::putText(input_img,classes[label]+" : "+string(strScore),Point(p1,p2-textSize.height),cv::FONT_HERSHEY_SIMPLEX,0.4,Scalar(105,80,250),1,1,false);
  }

  // restore the image
  cv::Mat result = input_img(Range(ceil(padh/2),(height-padh+ceil(padh/2))),Range(ceil(padw/2),(width-padw+ceil(padw/2))));
  
  cv::imshow("result",result);
  cv::waitKey(0);
  
  return 0;
}
