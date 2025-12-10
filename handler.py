import os
import torch
import torchaudio
import runpod
import base64
from io import BytesIO
import numpy as np
from scipy.io import wavfile
from scipy import signal
import traceback
import requests
import urllib.parse
from audiocraft.data.audio import audio_write

# --- Global Variables & Model Loading ---
INIT_ERROR_FILE = "/tmp/init_error.log"
model = None

try:
    if os.path.exists(INIT_ERROR_FILE):
        os.remove(INIT_ERROR_FILE)
        
    print("Loading MusicGen melody model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    from audiocraft.models import MusicGen
    model = MusicGen.get_pretrained("melody", device=device)
    model.set_generation_params(duration=30)
    print("Melody model loaded successfully.")

except Exception as e:
    tb_str = traceback.format_exc()
    with open(INIT_ERROR_FILE, "w") as f:
        f.write(f"Failed to initialize melody model: {tb_str}")
    model = None

# --- Helper Functions ---
def upsample_audio(input_wav_bytes, target_sr=48000):
    try:
        with BytesIO(input_wav_bytes) as in_io:
            sr, audio = wavfile.read(in_io)
        if sr == target_sr:
            return input_wav_bytes
        up_factor = target_sr / sr
        upsampled_audio = signal.resample(audio, int(len(audio) * up_factor))
        if audio.dtype == np.int16:
            upsampled_audio = upsampled_audio.astype(np.int16)
        with BytesIO() as out_io:
            wavfile.write(out_io, target_sr, upsampled_audio)
            return out_io.getvalue()
    except Exception:
        return input_wav_bytes

def upload_to_gcs(signed_url, audio_bytes, content_type="audio/wav"):
    try:
        response = requests.put(signed_url, data=audio_bytes, headers={"Content-Type": content_type}, timeout=300)
        response.raise_for_status()
        print(f"Uploaded to GCS: {signed_url[:100]}...")
        return True
    except Exception as e:
        print(f"GCS upload failed: {e}")
        return False

def notify_backend(callback_url, status, error_message=None):
    try:
        parsed = urllib.parse.urlparse(callback_url)
        params = urllib.parse.parse_qs(parsed.query)
        params['status'] = [status]
        if error_message:
            params['error_message'] = [error_message]
        new_query = urllib.parse.urlencode(params, doseq=True)
        webhook_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
        print(f"Calling webhook: {webhook_url}")
        response = requests.post(webhook_url, timeout=30)
        response.raise_for_status()
        print(f"Backend notified: {status}")
        return True
    except Exception as e:
        print(f"Webhook failed: {e}")
        return False

# --- Runpod Handler ---
def handler(event):
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            error_msg = f"Worker init failed: {f.read()}"
        return {"error": error_msg}

    job_input = event.get("input", {})
    melody_url = job_input.get("melody_url")
    melody_base64 = job_input.get("melody_base64")
    prompt = job_input.get("prompt")
    duration = job_input.get("duration", 30)
    callback_url = job_input.get("callback_url")
    upload_urls = job_input.get("upload_urls", {})
    sample_rate = job_input.get("sample_rate", 32000)
    
    if not melody_url and not melody_base64:
        error_msg = "Missing melody_url or melody_base64"
        if callback_url:
            notify_backend(callback_url, "failed", error_msg)
        return {"error": error_msg}
    
    if not prompt:
        error_msg = "Missing prompt"
        if callback_url:
            notify_backend(callback_url, "failed", error_msg)
        return {"error": error_msg}
    
    try:
        print(f"🎵 Melody continuation: '{prompt}', duration={duration}s")
        
        # Load melody audio
        if melody_base64:
            melody_bytes = base64.b64decode(melody_base64)
            melody, sr = torchaudio.load(BytesIO(melody_bytes))
        else:
            resp = requests.get(melody_url, timeout=60)
            resp.raise_for_status()
            melody, sr = torchaudio.load(BytesIO(resp.content))
        
        # Ensure mono
        if melody.shape[0] > 1:
            melody = torch.mean(melody, dim=0, keepdim=True)
        melody_batch = melody[None]
        
        # Generate continuation
        model.set_generation_params(duration=duration)
        wavs = model.generate_with_chroma([prompt], melody_batch, sr)
        one_wav = wavs[0]
        
        # Save with loudness normalization
        buffer = BytesIO()
        audio_write("temp", one_wav.cpu(), model.sample_rate, buffer, strategy="loudness")
        raw_wav_bytes = buffer.getvalue()
        
        final_wav_bytes = raw_wav_bytes
        if sample_rate == 48000:
            final_wav_bytes = upsample_audio(raw_wav_bytes)
        
        result = {
            "audio_base64_0": base64.b64encode(final_wav_bytes).decode('utf-8'),
            "sample_rate": sample_rate,
            "format": "wav"
        }
        
        # Upload if URL provided
        if upload_urls and upload_urls.get("wav_url_0"):
            upload_success = upload_to_gcs(upload_urls["wav_url_0"], final_wav_bytes)
            if upload_success:
                result["wav_url_0"] = upload_urls["wav_url_0"]
        
        print("Generated continuation")
        
        if callback_url:
            notify_backend(callback_url, "completed")
        
        return {
            "status": "completed",
            "result": result,
            "duration": duration,
            "prompt": prompt
        }
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"❌ Error: {error_msg}")
        if callback_url:
            notify_backend(callback_url, "failed", str(e))
        return {"error": error_msg, "status": "failed"}

# --- Start Serverless Worker ---
runpod.serverless.start({"handler": handler})
