import numpy as np
import librosa
import sounddevice as sd
import scipy.signal
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

# --- Configuración global ---
RATE = 44100
CHUNK = 2048
LOW_PASS_CUTOFF = 15000

stream = None
PITCH_SHIFT = -10
DISTORTION_LEVEL = 14
REVERB_DECAY = 0.0

# --- Obtener la ruta del script ---
def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()

# --- Archivos en la carpeta raíz ---
ICON_PATH = os.path.join(BASE_PATH, "poc.ico")
IMAGE_PATH = os.path.join(BASE_PATH, "896b093e-d79a-4d87-ba91-d929e366d3bf-8.png")

# --- Inicializar la ventana ---
root = tk.Tk()
root.title("Poc Voice Changer")

# --- Cargar ícono de ventana (opcional, pero recomendado) ---
try:
    if os.path.exists(ICON_PATH):
        root.iconbitmap(ICON_PATH)
    else:
        print(f"[WARN] Ícono no encontrado: {ICON_PATH}")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el ícono: {e}")

root.geometry("600x500")
root.resizable(False, False)
root.configure(bg="black")

monitor_var = tk.BooleanVar(value=False)

# --- Funciones de audio (sin cambios) ---
def clean_audio(audio):
    return np.clip(np.nan_to_num(audio), -1.0, 1.0)

def process_audio(audio):
    audio = clean_audio(audio)
    audio = librosa.effects.pitch_shift(audio, sr=RATE, n_steps=PITCH_SHIFT)
    sos = scipy.signal.butter(6, LOW_PASS_CUTOFF, btype='low', fs=RATE, output='sos')
    audio = scipy.signal.sosfilt(sos, audio)
    audio = np.tanh(DISTORTION_LEVEL * audio)
    echo = np.zeros_like(audio)
    delay = int(0.03 * RATE)
    echo[delay:] = audio[:-delay] * REVERB_DECAY
    return clean_audio(audio + echo)

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    processed_audio = process_audio(indata[:, 0])
    if monitor_var.get():
        outdata[:, 0] = processed_audio[:len(outdata)]
    else:
        outdata[:, 0] = np.zeros_like(outdata[:, 0])

def start_voice_changer():
    global stream
    if stream is None:
        input_name = input_device_var.get()
        output_name = output_device_var.get()
        input_index = get_device_index_by_name(input_name)
        output_index = get_device_index_by_name(output_name)

        if input_index is None or output_index is None:
            status_label.config(text="Device not found", fg="red")
            return

        try:
            stream = sd.Stream(samplerate=RATE,
                               blocksize=CHUNK,
                               dtype='float32',
                               channels=1,
                               callback=callback,
                               device=(input_index, output_index))
            stream.start()
            status_label.config(text="Voice Changer: ON", fg="white")
        except Exception as e:
            status_label.config(text=f"Error: {str(e)}", fg="red")

def stop_voice_changer():
    global stream
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None
        status_label.config(text="Voice Changer: OFF", fg="white")

def get_device_index_by_name(name):
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if name in dev['name']:
            return i
    return None

def get_input_output_devices():
    input_devices = []
    output_devices = []
    for dev in sd.query_devices():
        if dev['max_input_channels'] > 0:
            input_devices.append(dev['name'])
        if dev['max_output_channels'] > 0:
            output_devices.append(dev['name'])
    return input_devices, output_devices

def update_pitch(val):
    global PITCH_SHIFT
    PITCH_SHIFT = float(val)

def update_distortion(val):
    global DISTORTION_LEVEL
    DISTORTION_LEVEL = float(val)

def update_reverb(val):
    global REVERB_DECAY
    REVERB_DECAY = float(val)

# --- Interfaz gráfica ---
settings_frame = tk.Frame(root, bg="gray10", width=200)
settings_frame.pack(side="left", fill="y")

tk.Label(settings_frame, text="Pitch (semitones)", bg="gray10", fg="white").pack(pady=5)
pitch_slider = tk.Scale(settings_frame, from_=-12, to=12, orient="horizontal", command=update_pitch, bg="gray20", fg="white")
pitch_slider.set(PITCH_SHIFT)
pitch_slider.pack(pady=5)

tk.Label(settings_frame, text="Distortion", bg="gray10", fg="white").pack(pady=5)
dist_slider = tk.Scale(settings_frame, from_=1, to=30, orient="horizontal", command=update_distortion, bg="gray20", fg="white")
dist_slider.set(DISTORTION_LEVEL)
dist_slider.pack(pady=5)

tk.Label(settings_frame, text="Reverb", bg="gray10", fg="white").pack(pady=5)
reverb_slider = tk.Scale(settings_frame, from_=0.0, to=1.0, resolution=0.01, orient="horizontal", command=update_reverb, bg="gray20", fg="white")
reverb_slider.set(REVERB_DECAY)
reverb_slider.pack(pady=5)

main_frame = tk.Frame(root, bg="black")
main_frame.pack(side="left", fill="both", expand=True)

# --- Mostrar imagen + texto "WE ARE POC" ---
image_tk = None
if os.path.exists(IMAGE_PATH):
    try:
        img = Image.open(IMAGE_PATH)
        img = img.resize((150, 150), Image.LANCZOS)
        image_tk = ImageTk.PhotoImage(img)
        img_label = tk.Label(main_frame, image=image_tk, bg="black")
        img_label.pack(pady=10)
        
        # Texto "WE ARE POC" debajo de la imagen
        tk.Label(main_frame, text="WE ARE POC", font=("Arial", 16, "bold"), fg="white", bg="black").pack(pady=5)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar la imagen: {e}")
        tk.Label(main_frame, text="WE ARE POC", font=("Arial", 16, "bold"), fg="white", bg="black").pack(pady=10)
else:
    print(f"[WARN] Imagen no encontrada: {IMAGE_PATH}")
    tk.Label(main_frame, text="WE ARE POC", font=("Arial", 16, "bold"), fg="white", bg="black").pack(pady=10)

status_label = tk.Label(main_frame, text="Voice Changer: OFF", font=("Arial", 12), fg="white", bg="black")
status_label.pack(pady=10)

input_devices, output_devices = get_input_output_devices()

input_device_var = tk.StringVar(value=input_devices[0] if input_devices else "")
output_device_var = tk.StringVar(value=output_devices[0] if output_devices else "")

tk.Label(main_frame, text="Input Device:", bg="black", fg="white").pack()
input_menu = ttk.Combobox(main_frame, textvariable=input_device_var, values=input_devices)
input_menu.pack(pady=5)

tk.Label(main_frame, text="Output Device:", bg="black", fg="white").pack()
output_menu = ttk.Combobox(main_frame, textvariable=output_device_var, values=output_devices)
output_menu.pack(pady=5)

monitor_checkbox = tk.Checkbutton(main_frame, text="Monitor Audio", variable=monitor_var, bg="black", fg="white", selectcolor="gray10")
monitor_checkbox.pack(pady=5)

tk.Button(main_frame, text="Start Voice Changer", command=start_voice_changer, fg="white", bg="gray20").pack(pady=10)
tk.Button(main_frame, text="Stop Voice Changer", command=stop_voice_changer, fg="white", bg="gray20").pack(pady=5)

# --- Ejecutar ---
root.mainloop()