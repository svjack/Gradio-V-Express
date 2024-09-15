import argparse
import shutil
import subprocess

from inference import InferenceEngine
from sequence_utils import extract_kps_sequence_from_video

output_dir = "output"
temp_audio_path = "temp.mp3"

DEFAULT_MODEL_ARGS = {
    'unet_config_path': './model_ckpts/stable-diffusion-v1-5/unet/config.json',
    'vae_path': './model_ckpts/sd-vae-ft-mse/',
    'audio_encoder_path': './model_ckpts/wav2vec2-base-960h/',
    'insightface_model_path': './model_ckpts/insightface_models/',
    'denoising_unet_path': './model_ckpts/v-express/denoising_unet.bin',
    'reference_net_path': './model_ckpts/v-express/reference_net.bin',
    'v_kps_guider_path': './model_ckpts/v-express/v_kps_guider.bin',
    'audio_projection_path': './model_ckpts/v-express/audio_projection.bin',
    'motion_module_path': './model_ckpts/v-express/motion_module.bin',
    'device': 'cuda',
    'gpu_id': 0,
    'dtype': 'fp16',
    'num_pad_audio_frames': 2,
    'standard_audio_sampling_rate': 16000,
    'image_width': 512,
    'image_height': 512,
    'fps': 30.0,
    'seed': 42,
    'num_inference_steps': 25,
    'guidance_scale': 3.5,
    'context_frames': 12,
    'context_stride': 1,
    'context_overlap': 4,
}

INFERENCE_ENGINE = InferenceEngine(DEFAULT_MODEL_ARGS)

def infer(reference_image, audio_path, kps_sequence_save_path,
          output_path,
          retarget_strategy,
          reference_attention_weight, audio_attention_weight):
    global INFERENCE_ENGINE
    INFERENCE_ENGINE.infer(
        reference_image, audio_path, kps_sequence_save_path,
        output_path,
        retarget_strategy,
        reference_attention_weight, audio_attention_weight
    )
    return output_path, kps_sequence_save_path

def run_demo(
        reference_image, audio, video,
        kps_path, output_path, retarget_strategy,
        reference_attention_weight=0.95,
        audio_attention_weight=3.0):
    kps_sequence_save_path = f"{output_dir}/kps.pth"

    if video:
        audio_path = video.replace(".mp4", ".mp3")
        extract_kps_sequence_from_video(
            INFERENCE_ENGINE.app,
            video,
            audio_path,
            kps_sequence_save_path
        )
    else:
        audio_path = audio
        shutil.copy(kps_path, kps_sequence_save_path)

    subprocess.run(["ffmpeg", "-i", audio_path, "-c:v", "libx264", "-crf", "18", "-preset", "slow", temp_audio_path])
    shutil.move(temp_audio_path, audio_path)

    output_path, kps_sequence_save_path = infer(
        reference_image, audio_path, kps_sequence_save_path,
        output_path,
        retarget_strategy,
        reference_attention_weight, audio_attention_weight
    )

    print(f"Video generated successfully. Saved at: {output_path}")
    return output_path, kps_sequence_save_path

#### python v_express_cli.py --reference_image input/ref.jpg --audio input/audi.wav --kps_path input/kps.pth --output_path output_video.mp4
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V-Express Command Line Interface")
    parser.add_argument("--reference_image", type=str, required=True, help="Path to the reference image")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--video", type=str, help="Path to the video file (optional)")
    parser.add_argument("--kps_path", type=str, required=True, help="Path to the KPS sequences file")
    parser.add_argument("--output_path", type=str, default=f"{output_dir}/output_video.mp4", help="Path to save the generated video")
    parser.add_argument("--retarget_strategy", type=str, choices=["no_retarget", "fix_face", "offset_retarget", "naive_retarget"], default="no_retarget", help="Retargeting strategy")
    parser.add_argument("--reference_attention_weight", type=float, default=0.95, help="Reference attention weight")
    parser.add_argument("--audio_attention_weight", type=float, default=3.0, help="Audio attention weight")

    args = parser.parse_args()

    run_demo(
        args.reference_image, args.audio, args.video,
        args.kps_path, args.output_path, args.retarget_strategy,
        args.reference_attention_weight, args.audio_attention_weight
    )
