use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use webrtc_vad::{Vad, VadMode, SampleRate};

fn main() {
    println!("Ramping up ..");

    let host = cpal::default_host();
    let device = host.default_input_device().expect("Failed to get default input device");
    let config = device.default_input_config().expect("Failed to get default input config");
    let sample_rate = config.sample_rate().0 as f32;
    let mut vad = Vad::new_with_rate(webrtc_vad::SampleRate::Rate16kHz);
    vad.set_mode(VadMode::Aggressive);

    println!("Sample rate is at '{}'", sample_rate);

}