use tokio::io::Error;
use chrono::prelude::*;
use tokio::time::{sleep, Duration};
use tokio::sync::{mpsc, oneshot};
use rtrb::{Producer, RingBuffer};
use webrtc_vad::{Vad, VadMode, SampleRate};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub type Result<T> = std::result::Result<T, Error>;

const MAXIMUM_INACTIVE_CNT: usize = 35;    // 350 ms
const MINIMUM_SAMPLE_CNT: usize = 80 *  60; // 80 values à 10 ms (8kHz) :=  600ms
const MAXIMUM_SAMPLE_CNT: usize = 80 * 100; // 80 values à 10 ms (8kHz) := 1000ms

fn input_stream_callback(
    data: &[f32],
    sample_rate: f32,
    producer: &mut Producer<i16>,
) {
    // let now = Local::now();
    // let min = data.iter().cloned().fold(f32::MAX, f32::min);
    // let max = data.iter().cloned().fold(f32::MIN, f32::max);
    // let length = data.len();
    // println!("Min: {}, Max: {}, Len: {} @ {}", min, max, length, now.format("%Y-%m-%d %H:%M:%S%.3f"));
    
    let data_8k = normalize_audio_data_to_8k(data, &sample_rate);
    let vad_data_i16_8k: Vec<i16> = data_8k.iter().map(|x| (*x * 32767.0) as i16).collect();
    for sample in vad_data_i16_8k {
        producer.push(sample).expect("Failed to push sample to ring buffer");
    }

    // let data_16k = normalize_audio_data_to_16k(data, &sample_rate);
    // let vad_data_i16_16k: Vec<i16> = data_16k.iter().map(|x| (*x * 32767.0) as i16).collect();
    // for sample in vad_data_i16_16k {
    //     producer.push(sample).expect("Failed to push sample to ring buffer");
    // }

    // println!("producer: {}", data.len());
}

async fn record_audio(sender: mpsc::Sender<Vec<i16>>) {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("Failed to get default input device");
    let config = device.default_input_config().expect("Failed to get default input config");
    let sample_rate = config.sample_rate().0 as f32;
    let mut vad = Vad::new_with_rate(webrtc_vad::SampleRate::Rate8kHz);
    vad.set_mode(VadMode::Quality);

    println!("Sound configuration is '{:?}'", config);

    // Create a stream with the default input format
    let (mut producer, mut consumer) = RingBuffer::<i16>::new(16384);
    
    let stream = device.build_input_stream(
        &config.config(),
        move |data: &[f32], _info: &cpal::InputCallbackInfo| {
            input_stream_callback(data, sample_rate, &mut producer)
        },
        move |err| eprintln!("Error: {}", err),
        None,
    ).expect("Failed to build input stream");
    
    // Play the stream
    stream.play().expect("Failed to play stream");

    let n_frames = 80;
    let mut cnt_active:usize = 0;
    let mut cnt_inactive:usize = 0;
    let mut speech_segment = Vec::<i16>::new();

    loop {
        if consumer.slots() > n_frames {
            let mut audio_frame = Vec::<i16>::new();
            for _ in 0..n_frames {
                match consumer.pop() {
                    Ok(value) => {
                        audio_frame.push(value);
                    }
                    Err(err) => {
                        println!("Error: {}", err);
                        break;
                    },
                }
            }

            let is_active = vad.is_voice_segment(&audio_frame).expect("Failed to check voice segment");

            match (is_active, cnt_active < MAXIMUM_SAMPLE_CNT, cnt_active > MINIMUM_SAMPLE_CNT, cnt_inactive > MAXIMUM_INACTIVE_CNT) {
                // active & cnt_active still able to grow -> grow the buffer
                (true, true, _, _)  => {
                    speech_segment.extend(audio_frame);
                },

                // active & cnt_active at its upper limit -> send buffer to AI
                (true, false, _, _) => {
                    sender.send(speech_segment.clone()).await.expect("Failed to send data");
                    speech_segment.clear();
                    cnt_inactive = 0;
                    cnt_active = 0;
                },
                
                // not active for a little while          -> wait
                (false, _, _, false) => {
                    cnt_inactive += 1;
                },

                // not active for a longer time 
                // but we gathered enough frames already  -> send buffer to AI
                (false, _, true, true) => {
                    sender.send(speech_segment.clone()).await.expect("Failed to send data");
                    speech_segment.clear();
                    cnt_inactive = 0;
                    cnt_active = 0;
                },

                // not active for a longer time
                // and we haven't gathered many samples   -> discard and start over
                (false, _, false, true) => {
                    speech_segment.clear();
                    cnt_inactive = 0;
                    cnt_active = 0;
                },

            };
            
            // let now = Local::now();
            // match speech_active {
            //     true => println!(">>> A {}", now.format("%Y-%m-%d %H:%M:%S%.3f")),
            //     false=> println!("<<< I {}", now.format("%Y-%m-%d %H:%M:%S%.3f")),
            // };
        }
    }
}

fn normalize_audio_data_to_16k(input_data: &[f32], input_sample_rate: &f32) -> Vec<f32> {
    let target_sample_rate = 16000f32;
    let resample_ratio = target_sample_rate / input_sample_rate;
    let mut resampled_data = Vec::new();
    for i in 0..(input_data.len() as f32 * resample_ratio) as usize {
        let x = i as f32 / resample_ratio;
        let x1 = x.floor() as usize;
        let x2 = x.ceil() as usize;

        if x2 >= input_data.len() {
            break;
        }

        let y1 = input_data[x1];
        let y2 = input_data[x2];

        let y = y1 + (y2 - y1) * (x - x1 as f32);
        resampled_data.push(y);
    }

    resampled_data
}

fn normalize_audio_data_to_8k(input_data: &[f32], input_sample_rate: &f32) -> Vec<f32> {
    let target_sample_rate = 8000f32;
    let resample_ratio = target_sample_rate / input_sample_rate;
    let mut resampled_data = Vec::new();
    for i in 0..(input_data.len() as f32 * resample_ratio) as usize {
        let x = i as f32 / resample_ratio;
        let x1 = x.floor() as usize;
        let x2 = x.ceil() as usize;

        if x2 >= input_data.len() {
            break;
        }

        let y1 = input_data[x1];
        let y2 = input_data[x2];

        let y = y1 + (y2 - y1) * (x - x1 as f32);
        resampled_data.push(y);
    }

    resampled_data
}

async fn consume_audio(receiver: mpsc::Receiver<Vec<i16>>) {
    loop{}
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Ramping up ..");

    let (sender, receiver) = mpsc::channel(32);

    tokio::spawn(async move {
        record_audio(sender).await;
    });

    tokio::spawn(async move {
        consume_audio(receiver).await;
    });

    
    let print_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        loop {
            interval.tick().await;
        }
    });

    let _ = tokio::join!(print_task);

    Ok(())
}
