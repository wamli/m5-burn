// mod data;

use burn::{
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use hound::{WavReader, WavSpec};
use m5_burn::model;
use std::path::{Path, PathBuf};

type Backend = burn_ndarray::NdArray<f32>;

/// Load the model from the file in your source code (not in build.rs or script).
fn load_model(model_path: PathBuf) -> model::M5::<Backend> {
    let device = Default::default();
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load(model_path.into(), &device)
        .expect("Should decode state successfully");

    model::M5::<Backend>::init(&device).load_record(record)
}

fn load_wav_to_tensor<B: burn::prelude::Backend>(file_path: &str) -> Tensor<B, 3> {

    let device = Default::default();

    // Load the .wav file
    let mut reader = WavReader::open(file_path).unwrap();
    let spec = reader.spec();
    println!("Going to consume file of {:?} ..", spec);
    // let samples: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32 / i16::MAX as f32).collect();
    let samples: Vec<f32> = reader.samples::<f32>().collect::<Result<_, _>>().unwrap();

    // Reshape the audio samples
    let seq_length = samples.len();

    println!("\t with a sequence length of {:?}", seq_length);

    let audio_tensor = Data::<f32, 3>::new(samples, Shape::new([1, 1, seq_length]));
    // let audio_tensor = Array3::from_shape_vec((1, 1, seq_length), samples).unwrap();

    // Create the 3D tensor
    Tensor::<B, 3>::from_data(audio_tensor.convert(), &device)
}

fn main() {
    println!("Ramping up ..");

    let root_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let import_path = Path::new(&root_dir).join("models/burn/m5");

    let m5 = load_model(import_path);
    
    println!("Loaded model '{}'", m5);

    let input = load_wav_to_tensor("notebooks/zero.wav");

    println!("Looking for some inference result for {:?} ..", input.shape());

    let x = m5.forward(input);

    println!("Output: {:?}", x);

}