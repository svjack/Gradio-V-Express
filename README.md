# Gradio V-Express Project

## Introduction

This repository is created to run the V-Express project via Gradio, compiling various resources and code snippets shared by different individuals. The original project was created by [tencent-ailab](https://github.com/tencent-ailab/V-Express) and the contributions made by [faraday](https://github.com/faraday) and [StableAIHub](https://www.youtube.com/@StableAIHub) made it possible to run this project using Gradio.

Check out the YouTube Video NewGenAI:
[NewGenAI](https://youtu.be/OFt6a2rR8GY?si=S82ZwP1w1OJvlYJR)

## Installation

### Steps

1. **Pre-install Dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y cmake ffmpeg git-lfs
   ```

2. **Clone the Repository and Set Up Environment:**
   ```bash
   git clone https://github.com/svjack/Gradio-V-Express && cd Gradio-V-Express
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages:**
   ```bash
   pip install xformers==0.0.21 torch torchvision torchaudio diffusers==0.24.0 imageio-ffmpeg==0.4.9 omegaconf==2.2.3 onnxruntime-gpu==1.16.3 safetensors==0.4.2 transformers==4.30.2 einops==0.4.1 tqdm==4.66.1 av==11.0.0 accelerate insightface dlib gradio flask
   ```

4. **Download and Set Up Models:**
   ```bash
   git clone https://huggingface.co/tk93/V-Express
   cp -r V-Express/model_ckpts/ .
   cp V-Express/*.bin model_ckpts/v-express/
   ```

<!--
5. **Modify `app.py` to Use `.bin` Files:**
   ```bash
   sed -i 's/\.pth/\.bin/g' app.py
   ```
-->

## Gradio Usage

1. **Start the Application:**
   ```bash
   python app.py
   ```

2. **Open Your Web Browser:**
   Navigate to the provided Gradio link to interact with the application.<br/>
   Use files in input dir

## Cli Usage

1. **Run in Cmd:**
   ```bash
   python v_express_cli.py --reference_image input/ref.jpg --audio input/audi.wav --kps_path input/kps.pth --output_path output_video.mp4
   ```
## Flask Usage

1. **Start Flask Server:**
   ```bash
   python v_express_flask.py
   ```

2. **Call interface by request**
   ```python
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
   ```

## License

This repository does not contain original code, but rather a collection of resources from various contributors. Please refer to the individual licenses of the original projects for more details.

## Contributions

If you have any suggestions or questions, feel free to open an issue or submit a pull request. We welcome contributions from the community!
