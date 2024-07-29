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
      pool::{MaxPool1d, MaxPool1dConfig},
   },
   tensor::{
      Tensor,
      module::avg_pool1d,
      activation::{log_softmax, relu},
   },
   record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};

const N_INPUT:usize = 1;
const N_OUTPUT:usize = 35;
const KERNEL_SIZE:usize = 80;
const STRIDE:usize = 16;
const N_CHANNEL:usize = 32; // `num_features` in in PyTorch parlance

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

   // avg_pool: AvgPool1d,

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
      let pc1 = burn::nn::PaddingConfig1d::Explicit(0);
      let pc2 = burn::nn::PaddingConfig1d::Explicit(0);
      let pc3 = burn::nn::PaddingConfig1d::Explicit(0);
      let pc4 = burn::nn::PaddingConfig1d::Explicit(0);

      let conv1 = Conv1dConfig::new(N_INPUT, N_CHANNEL, KERNEL_SIZE).with_stride(STRIDE).init(device);
      let bn1 = BatchNormConfig::new(N_CHANNEL).init(device);
      let pool1 = MaxPool1dConfig::new(4).with_stride(4).with_padding(pc1).with_dilation(1).init();
      
      let conv2 = Conv1dConfig::new(N_CHANNEL, N_CHANNEL, 3).init(device);
      let bn2 = BatchNormConfig::new(N_CHANNEL).init(device);
      let pool2 = MaxPool1dConfig::new(4).with_stride(4).with_padding(pc2).with_dilation(1).init();

      let conv3 = Conv1dConfig::new(N_CHANNEL, 2 * N_CHANNEL, 3).init(device);
      let bn3 = BatchNormConfig::new(2 * N_CHANNEL).init(device);
      let pool3 = MaxPool1dConfig::new(4).with_stride(4).with_padding(pc3).with_dilation(1).init();

      let conv4 = Conv1dConfig::new(2 * N_CHANNEL, 2 * N_CHANNEL, 3).init(device);
      let bn4 = BatchNormConfig::new(2 * N_CHANNEL).init(device);
      let pool4 = MaxPool1dConfig::new(4).with_stride(4).with_padding(pc4).with_dilation(1).init();

      // let avg_pool = AvgPool1dConfig::new(KERNEL_SIZE).init();

      let fc1 = LinearConfig::new(2 * N_CHANNEL, N_OUTPUT).init(device);

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

         // avg_pool,

         fc1,
      }
   }

   pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
      let conv1_out = self.conv1.forward(input);
      let bn1_out = self.bn1.forward(conv1_out);
      let relu_out1 = relu(bn1_out);
      let pool1_out = self.pool1.forward(relu_out1);
      let conv2_out = self.conv2.forward(pool1_out); 
      let bn2_out = self.bn2.forward(conv2_out);
      let relu_out2 = relu(bn2_out);
      let pool2_out = self.pool2.forward(relu_out2);
      let conv3_out = self.conv3.forward(pool2_out);
      let bn3_out = self.bn3.forward(conv3_out);
      let relu_out3 = relu(bn3_out);
      let pool3_out = self.pool3.forward(relu_out3);
      let conv4_out = self.conv4.forward(pool3_out);
      let bn4_out = self.bn4.forward(conv4_out);
      let relu_out4 = relu(bn4_out);
      let pool4_out = self.pool4.forward(relu_out4);
      let avg_pool = avg_pool1d(pool4_out, 1, 1, 0, true);
      let permuted = Tensor::permute(avg_pool, [0,2,1]);
      // let fc1_out = self.fc1.forward(permuted);
      let fc1_out = self.fc1.forward(permuted);
      log_softmax(fc1_out, 2)
   }


   // pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
   //    print!("shape of input: {:?}", input.shape());

   //    let conv1_out = self.conv1.forward(input);
   //    print!("shape of conv1: {:?}", conv1_out.shape());

   //    let bn1_out = self.bn1.forward(conv1_out);
   //    print!("shape of bn1: {:?}", bn1_out.shape());

   //    let relu_out1 = relu(bn1_out);

   //    let pool1_out = self.pool1.forward(relu_out1);
   //    print!("shape of pool1: {:?}", pool1_out.shape());

   //    let conv2_out = self.conv2.forward(pool1_out); 
   //    print!("shape of conv2: {:?}", conv2_out.shape());

   //    let bn2_out = self.bn2.forward(conv2_out);
   //    print!("shape of bn2: {:?}", bn2_out.shape());

   //    let relu_out2 = relu(bn2_out);

   //    let pool2_out = self.pool2.forward(relu_out2);
   //    print!("shape of pool2: {:?}", pool2_out.shape());
      
   //    let conv3_out = self.conv3.forward(pool2_out);
   //    print!("shape of conv3: {:?}", conv3_out.shape());

   //    let bn3_out = self.bn3.forward(conv3_out);
   //    print!("shape of bn3: {:?}", bn3_out.shape());

   //    let relu_out3 = relu(bn3_out);

   //    let pool3_out = self.pool3.forward(relu_out3);
   //    print!("shape of pool3: {:?}", pool3_out.shape());

   //    let conv4_out = self.conv4.forward(pool3_out);
   //    print!("shape of conv4: {:?}", conv4_out.shape());

   //    let bn4_out = self.bn4.forward(conv4_out);
   //    print!("shape of bn4: {:?}", bn4_out.shape());

   //    let relu_out4 = relu(bn4_out);

   //    let pool4_out = self.pool4.forward(relu_out4);
   //    print!("shape of pool4: {:?}", pool4_out.shape());

   //    let avg_pool = avg_pool1d(pool4_out, 1, 1, 0, true);
   //    // print!("shape of avg_pool: {:?}", avg_pool.shape());
   //    // let permuted = Tensor::permute(avg_pool, [0,2,1]);
   //    let permuted = Tensor::permute(avg_pool, [0,2,1]);

   //    // let fc1_out = self.fc1.forward(permuted);
   //    let fc1_out = self.fc1.forward(permuted);
   //    print!("shape of fc1_out: {:?}", fc1_out.shape());

   //    log_softmax(fc1_out, 2)
   // }
}
