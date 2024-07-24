use std::env::args;
use std::path::Path;

use burn::{
    backend::ndarray::NdArray,
    // data::dataset::{vision::MnistDataset, Dataset},
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::Tensor,
};

use model::M5;

const IMAGE_INX: usize = 42; // <- Change this to test a different image

// Build output direct that contains converted model weight file path
const OUT_DIR: &str = concat!(env!("OUT_DIR"), "/model/m5");

fn main() {
    type Backend = NdArray<f32>;
    let device = Default::default();

    // Load the model record from converted PyTorch file by the build script
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load(Path::new(OUT_DIR).into(), &device)
        .expect("Failed to decode state");

    // Create a new model and load the state
    let model: M5<Backend> = M5::init(&device).load_record(record);

    // Load the MNIST dataset and get an item

    // Create a tensor from the image data

    // Normalize the input

    // Run the model on the input

    // Get the index of the maximum value

    // Check if the index matches the label

    // Print the image URL if the image index is less than 100 (the online dataset only has 100 images)

}
