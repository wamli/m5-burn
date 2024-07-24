use std::{
   env,
   path::Path,
};
use burn::{
   prelude::*,
   nn::{
      BatchNorm, BatchNormConfig,
      Linear, LinearConfig,
      conv::{Conv1d, Conv1dConfig},
      pool::{MaxPool1d, MaxPool1dConfig, AvgPool1d, AvgPool1dConfig},
   },
   tensor::{
      activation::{log_softmax, relu},
      Tensor,
   },
   record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};


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

   avg_pool: AvgPool1d,

   fc1:   Linear<B>,
}

impl<B: Backend> Default for M5<B> {
   fn default() -> Self {
       let device = B::Device::default();
       let out_dir = env::var_os("OUT_DIR").unwrap();
       let file_path = Path::new(&out_dir).join("model/mnist");

       let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
           .load(file_path, &device)
           .expect("Failed to decode state");

       Self::init(&device).load_record(record)
   }
}

impl<B: Backend> M5<B> {
   pub fn init(device: &B::Device) -> Self {
      let conv1 = Conv1dConfig::new(CHANNELS_IN, CHANNELS_OUT, KERNEL_SIZE).with_stride(STRIDE).init(device);
      let bn1 = BatchNormConfig::new(N_CHANNELS).init(device);
      let pool1 = MaxPool1dConfig::new(KERNEL_SIZE).init();
      
      let conv2 = Conv1dConfig::new(CHANNELS_IN, CHANNELS_OUT, KERNEL_SIZE).with_stride(STRIDE).init(device);
      let bn2 = BatchNormConfig::new(N_CHANNELS).init(device);
      let pool2 = MaxPool1dConfig::new(KERNEL_SIZE).init();

      let conv3 = Conv1dConfig::new(CHANNELS_IN, CHANNELS_OUT, KERNEL_SIZE).with_stride(STRIDE).init(device);
      let bn3 = BatchNormConfig::new(N_CHANNELS).init(device);
      let pool3 = MaxPool1dConfig::new(KERNEL_SIZE).init();

      let conv4 = Conv1dConfig::new(CHANNELS_IN, CHANNELS_OUT, KERNEL_SIZE).with_stride(STRIDE).init(device);
      let bn4 = BatchNormConfig::new(N_CHANNELS).init(device);
      let pool4 = MaxPool1dConfig::new(KERNEL_SIZE).init();

      let avg_pool = AvgPool1dConfig::new(KERNEL_SIZE).init();

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

         avg_pool,

         fc1,
      }
   }

   pub fn forward(&self, input1: Tensor<B, 3>) -> Tensor<B, 3> {
      let conv1_out = self.conv1.forward(input1);
      let bn1_out = self.bn1.forward(conv1_out);
      let relu_out1 = relu(bn1_out);
      let pool1_out = self.pool1.forward(relu_out1);

      let conv2_out = self.conv2.forward(pool1_out);
      let bn2_out = self.bn2.forward(conv2_out);
      let relu_out2 = relu(bn2_out);
      let pool2_out = self.pool2.forward(relu_out2);
      
      let conv3_out = self.conv2.forward(pool2_out);
      let bn3_out = self.bn2.forward(conv3_out);
      let relu_out3 = relu(bn3_out);
      let pool3_out = self.pool2.forward(relu_out3);

      let conv4_out = self.conv2.forward(pool3_out);
      let bn4_out = self.bn2.forward(conv4_out);
      let relu_out4 = relu(bn4_out);
      let pool4_out = self.pool2.forward(relu_out4);

      let avg_pool = self.avg_pool.forward(pool4_out);
      let permuted = Tensor::permute(avg_pool, [0,2,1]);

      let fc1_out = self.fc1.forward(permuted);

      log_softmax(fc1_out, 2)
   }
}
