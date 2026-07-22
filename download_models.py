"""
download_models.py — fetches the pretrained YuNet + SFace ONNX weights
from the OpenCV Zoo. Run this once, from your own machine (not this
sandbox), before running enroll.py or app.py.

    python download_models.py
"""
import os
import urllib.request

MODELS = {
    "models/face_detection_yunet.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "models/face_recognition_sface.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

os.makedirs("models", exist_ok=True)

for out_path, url in MODELS.items():
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10_000:
        print(f"already have {out_path}, skipping")
        continue
    print(f"downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  done ({size_kb:.0f} KB)")

print("\nAll models downloaded. If a file is only ~1KB, GitHub served a Git-LFS")
print("pointer instead of the binary — in that case, download it manually from")
print("the URL above by opening it in a browser and using the 'Download' button.")
