
import tensorflow as tf
import numpy as np
import os

# --- 1. VERIFIKASI GPU (DirectML) ---
# Pastikan TensorFlow dapat mendeteksi GPU AMD Anda melalui DirectML.
print(f"TensorFlow Version: {tf.__version__}")

# Dapatkan daftar semua perangkat yang terlihat oleh TensorFlow
physical_devices = tf.config.list_physical_devices()
gpu_devices = [d for d in physical_devices if d.device_type == 'GPU']

if gpu_devices:
    # Cek apakah plugin DirectML aktif
    if any("dml" in d.name.lower() for d in gpu_devices):
        print(f"\nSUCCESS: GPU AMD (DirectML) terdeteksi!")
        for gpu in gpu_devices:
            print(f"  - Name: {gpu.name}")
        print("\nKalkulasi Deep Learning akan dijalankan di GPU.")
    else:
        print("\nWARNING: GPU terdeteksi, tetapi bukan perangkat DirectML. Pelatihan mungkin berjalan di CPU.")
else:
    print("\nERROR: Tidak ada GPU yang terdeteksi oleh TensorFlow. Pelatihan akan berjalan di CPU.")
    print("Pastikan Anda telah menginstal 'tensorflow-directml-plugin' di environment Python yang benar (3.11).")

print("-" * 50)


# --- 2. MEMBUAT DATASET SINTETIS ---
# Kita akan membuat gambar sederhana (lingkaran dan kotak) agar skrip bisa langsung dijalankan.
IMG_SIZE = 64
NUM_SAMPLES_PER_CLASS = 500

def create_shapes(shape_type, num_samples):
    """Membuat gambar berisi lingkaran atau kotak."""
    images = np.zeros((num_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    for i in range(num_samples):
        x = np.random.randint(10, IMG_SIZE - 10)
        y = np.random.randint(10, IMG_SIZE - 10)
        s = np.random.randint(5, 10)
        if shape_type == 'circle':
            for row in range(IMG_SIZE):
                for col in range(IMG_SIZE):
                    if (row - y)**2 + (col - x)**2 < s**2:
                        images[i, row, col, 0] = 1.0
        elif shape_type == 'square':
            images[i, y-s:y+s, x-s:x+s, 0] = 1.0
    return images

print("\nMembuat dataset sintetis (lingkaran vs. kotak)...")
circles = create_shapes('circle', NUM_SAMPLES_PER_CLASS)
squares = create_shapes('square', NUM_SAMPLES_PER_CLASS)

# Gabungkan data dan buat label (0 untuk lingkaran, 1 untuk kotak)
X_train = np.concatenate([circles, squares])
y_train = np.array([0]*NUM_SAMPLES_PER_CLASS + [1]*NUM_SAMPLES_PER_CLASS)

# Acak data
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

print(f"Dataset dibuat dengan {len(X_train)} gambar ({IMG_SIZE}x{IMG_SIZE}).")
print("-" * 50)


# --- 3. MEMBANGUN MODEL CNN SEDERHANA ---
print("\nMembangun model Convolutional Neural Network (CNN)...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Output tunggal untuk klasifikasi biner
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
print("-" * 50)


# --- 4. MELATIH MODEL (KALKULASI DI GPU) ---
print("\nMEMULAI PELATIHAN MODEL...")
print("Perhatikan penggunaan GPU Anda di Task Manager (tab Performance -> GPU).")
print("Penggunaan GPU '3D' atau 'Compute' akan meningkat selama proses ini.")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

print("\nPelatihan Selesai!")
print("-" * 50)


# --- 5. EVALUASI HASIL ---
final_accuracy = history.history['val_accuracy'][-1]
print(f"\nHasil Akhir: Akurasi pada data validasi = {final_accuracy*100:.2f}%")
