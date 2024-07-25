use std::path::Path;

use burn::{
    prelude::*,
    // module::Module,
    // tensor::Tensor,
    backend::ndarray::NdArray,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use m5_burn::model::{self};
use burn_import::pytorch::PyTorchFileRecorder;

// Basic backend type (not used directly here).
type B = NdArray<f32>;

// Build output direct that contains converted model weight file path
// const OUT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/binaries");

// fn import_model<B>() {
//     let device = Default::default();

//     // Load PyTorch weights into a model record.
//     let record: model::M5Record<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
//     .load("pytorch/m5_model_weights.pt".into(), &device)
//     .expect("Failed to decode state");

//     // Save the model record to a file.
//     let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

//     // Save into the OUT_DIR directory so that the model can be loaded by the
//     let out_dir = std::env::var(OUT_DIR).unwrap();
//     let file_path = Path::new(&out_dir).join("model/m5");
//     println!("Writing to '{}'", &file_path);

//     recorder
//         .record(record, file_path)
//         .expect("Failed to save model record");
// }

fn main() {
    let device = Default::default();

    // println!("Imported model '{:?}'", model::M5);
    
    // Load PyTorch weights into a model record.
    let record: model::M5Record<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
    .load("models/pytorch/m5_model_weights.pt".into(), &device)
    .expect("Failed to decode state");

    // Save the model record to a file.
    // Save to CARGO_MANIFEST_DIR/binaries/model directory so that the model can be loaded
    let out_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file_path = Path::new(&out_dir).join("models/burn/m5");
    println!("Writing imported model to '{:?}'", &file_path);
    
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(record, file_path)
        .expect("Failed to save model record");


    // type Backend = NdArray<f32>;
    // let device = Default::default();

    // // Load the model record from converted PyTorch file by the build script
    // let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
    //     .load(Path::new(OUT_DIR).into(), &device)
    //     .expect("Failed to decode state");

    // // Create a new model and load the state
    // let model: model::M5<Backend> = M5::init(&device).load_record(record);

    // Load the MNIST dataset and get an item

    // Create a tensor from the image data

    // Normalize the input

    // Run the model on the input

    // Get the index of the maximum value

    // Check if the index matches the label

    // Print the image URL if the image index is less than 100 (the online dataset only has 100 images)

}
