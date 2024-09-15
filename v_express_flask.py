from flask import Flask, request, jsonify
import shutil
import subprocess
from inference import InferenceEngine
from sequence_utils import extract_kps_sequence_from_video

app = Flask(__name__)

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

@app.route('/generate_video', methods=['POST'])
def generate_video():
    data = request.json
    reference_image = data.get('reference_image')
    audio = data.get('audio')
    video = data.get('video', None)
    kps_path = data.get('kps_path')
    output_path = data.get('output_path', f"{output_dir}/output_video.mp4")
    retarget_strategy = data.get('retarget_strategy', 'no_retarget')
    reference_attention_weight = data.get('reference_attention_weight', 0.95)
    audio_attention_weight = data.get('audio_attention_weight', 3.0)

    output_path, kps_sequence_save_path = run_demo(
        reference_image, audio, video,
        kps_path, output_path, retarget_strategy,
        reference_attention_weight, audio_attention_weight
    )

    return jsonify({"output_path": output_path, "kps_sequence_save_path": kps_sequence_save_path})

'''
import requests

url = "http://127.0.0.1:5000/generate_video"
data = {
    "reference_image": "input/ref.jpg",
    "audio": "input/audi.wav",
    "kps_path": "input/kps.pth",
    "output_path": "output_video.mp4"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print(f"Video generated successfully. Saved at: {result['output_path']}")
else:
    print(f"Error: {response.status_code}")
'''
if __name__ == "__main__":
    app.run(debug=True)
