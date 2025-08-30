import requests
import base64
import os

API_URL = "http://127.0.0.1:8000/generate-image"

# Data prompt yang ingin Anda kirim
payload = {
    "prompt": "a cinematic photo of a robot cat ordering coffee at a cafe, detailed, 8k",
    "negative_prompt": "blurry, low quality",
    "width": 896,
    "height": 896
}

print("Mengirim permintaan ke server AI, ini mungkin butuh beberapa menit...")

try:
    response = requests.post(API_URL, json=payload)
    response.raise_for_status() 
    data = response.json()

    image_base64 = data.get("image_base64")

    if image_base64:
        image_data = base64.b64decode(image_base64)
        output_filename = "hasil_dari_api.png"
        with open(output_filename, "wb") as f:
            f.write(image_data)

        print(f"Gambar berhasil disimpan sebagai '{output_filename}'")
        try:
            os.startfile(output_filename)
        except AttributeError:
            print("Tidak bisa membuka gambar secara otomatis di sistem operasi ini.")

except requests.exceptions.RequestException as e:
    print(f"Gagal menghubungi server: {e}")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")