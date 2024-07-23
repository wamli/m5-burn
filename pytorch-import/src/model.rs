use burn::{
   nn::{
      BatchNorm, BatchNormConfig,
      Linear, LinearConfig,
      conv::{Conv1d, Conv1dConfig},
      pool::MaxPool1d,
      // Dropout, DropoutConfig, Linear, LinearConfig, Relu,
   },
   prelude::*,
};
use nn::pool::MaxPool1dConfig;

const CHANNELS_IN:usize = 1;
const CHANNELS_OUT:usize = 35;
const KERNEL_SIZE:usize = 80;
const STRIDE:usize = 16;
const N_CHANNELS:usize = 32; // `num_features` in in PyTorch parlance

// #[derive(Module, Debug)]
// pub struct Model<B: Backend> {
//    conv1: Conv2d<B>,
//    conv2: Conv2d<B>,
//    pool: AdaptiveAvgPool2d,
//    dropout: Dropout,
//    linear1: Linear<B>,
//    linear2: Linear<B>,
//    activation: Relu,
// }

#[derive(Module, Debug)]
pub struct M5<B: Backend> {
   conv1: Conv1d<B>,
   bn1:   BatchNorm<B, 1>,
   pool1: MaxPool1d,
   conv2: Conv1d<B>,
   bn2:   BatchNorm<B, 1>,
   pool2: MaxPool1d,
   conv3: Conv1d<B>,
   bn3:   BatchNorm<B, 1>,
   pool3: MaxPool1d,
   conv4: Conv1d<B>,
   bn4:   BatchNorm<B, 1>,
   pool4: MaxPool1d,
   fc1:   Linear<B>,
}

impl<B: Backend> M5<B> {
   pub fn init(device: &B::Device) -> Self {
      let conv1 = Conv1dConfig::new(CHANNELS_IN, CHANNELS_OUT, KERNEL_SIZE).with_stride(STRIDE).init(device);
      let bn1 = BatchNormConfig::new(N_CHANNELS).init(device);
      let pool1 = MaxPool1dConfig::new(KERNEL_SIZE);
      
      let conv2 = Conv1dConfig::new(CHANNELS_IN, CHANNELS_OUT, KERNEL_SIZE).with_stride(STRIDE).init(device);
      let bn2 = BatchNormConfig::new(N_CHANNELS).init(device);
      let pool2 = MaxPool1dConfig::new(KERNEL_SIZE);

      let conv3 = Conv1dConfig::new(CHANNELS_IN, CHANNELS_OUT, KERNEL_SIZE).with_stride(STRIDE).init(device);
      let bn3 = BatchNormConfig::new(N_CHANNELS).init(device);
      let pool3 = MaxPool1dConfig::new(KERNEL_SIZE);

      let conv4 = Conv1dConfig::new(CHANNELS_IN, CHANNELS_OUT, KERNEL_SIZE).with_stride(STRIDE).init(device);
      let bn4 = BatchNormConfig::new(N_CHANNELS).init(device);
      let pool4 = MaxPool1dConfig::new(KERNEL_SIZE);

      let fc1 = LinearConfig::new(2 * N_CHANNELS, CHANNELS_OUT).init(device);

      Self {
         conv1,
         bn1,
         pool1,
         
         conv2,
         bn2,
         pool2,

         conv3,
         bn3,
         pool3,

         conv4,
         bn4,
         pool4,

         fc1,
      }
   }

   pub fn forward(&self, input1: Tensor<B, 4>) -> Tensor<B, 2> {
       let conv1_out1 = self.conv1.forward(input1);
       let relu1_out1 = relu(conv1_out1);
       let conv2_out1 = self.conv2.forward(relu1_out1);
       let relu2_out1 = relu(conv2_out1);
       let conv3_out1 = self.conv3.forward(relu2_out1);
       let relu3_out1 = relu(conv3_out1);
       let norm1_out1 = self.norm1.forward(relu3_out1);
       let flatten1_out1 = norm1_out1.flatten(1, 3);
       let fc1_out1 = self.fc1.forward(flatten1_out1);
       let relu4_out1 = relu(fc1_out1);
       let fc2_out1 = self.fc2.forward(relu4_out1);
       let norm2_out1 = self.norm2.forward(fc2_out1);
       log_softmax(norm2_out1, 1)
   }
}
