# 🖼️ PictureUpgrader — Colorize & Super-Resolution Android App

[![Build APK](https://github.com/YOUR_USERNAME/pictureupgrader/actions/workflows/build.yml/badge.svg)](https://github.com/YOUR_USERNAME/pictureupgrader/actions/workflows/build.yml)

Tingkatkan foto hitam-putih Anda secara otomatis dengan **pewarnaan AI** dan **super-resolution** langsung di Android. Ditenagai oleh [Tencent NCNN](https://github.com/Tencent/ncnn) dan [OpenCV](https://opencv.org/), ditulis dalam Kotlin + C++17.

---

## ✨ Fitur

| Fitur | Deskripsi |
|---|---|
| 🎨 Colorize | Pewarnaan foto hitam-putih dengan AI |
| 🔍 Super-Resolution | Peningkatan resolusi foto menggunakan CodeFormer + RealESRGAN |
| 📉 Down-Sampling | Kompres resolusi sebelum super-resolution |
| 📱 UI Sederhana | Satu layar, tombol langsung aksi |

---

## 🏗️ Tech Stack

- **Kotlin** — UI & Android logic
- **C++17** — Native inference engine
- **NCNN** (Tencent) — Neural network inference (Vulkan GPU)
- **OpenCV 4.6** — Image processing
- **CodeFormer** — Face restoration & enhancement
- **RealESRGAN** — General super-resolution

---

## 🚀 Build di GitHub Actions (Otomatis)

Setiap push ke branch `main` akan otomatis:
1. Download OpenCV 4.6.0 Android SDK
2. Download NCNN 20230816 (Vulkan)
3. Build Debug & Release APK
4. Upload APK sebagai GitHub Release

**Tidak perlu setup apapun secara manual di GitHub.**

---

## 🛠️ Build Manual (Android Studio)

### Prasyarat
- Android Studio Hedgehog atau lebih baru
- NDK 24+ dengan CMake 3.22+
- Android SDK 33

### Langkah

```bash
# 1. Clone repositori
git clone https://github.com/YOUR_USERNAME/pictureupgrader.git
cd pictureupgrader

# 2. Download library ke app/src/main/cpp/
cd app/src/main/cpp

# OpenCV Android SDK
wget https://github.com/opencv/opencv/releases/download/4.6.0/opencv-4.6.0-android-sdk.zip
unzip opencv-4.6.0-android-sdk.zip
mv OpenCV-android-sdk opencv-4.6.0-android-sdk

# NCNN (Vulkan)
wget https://github.com/Tencent/ncnn/releases/download/20230816/ncnn-20230816-android-vulkan.zip
unzip ncnn-20230816-android-vulkan.zip
mv ncnn-20230816-android-vulkan ncnn-custom-android-vulkan

# 3. Download model assets dari Releases dan letakkan di:
#    app/src/main/assets/models/

# 4. Build
cd ../../../..
./gradlew assembleDebug
```

---

## 📁 Struktur Proyek

```
pictureupgrader/
├── .github/workflows/build.yml   ← GitHub Actions CI/CD
├── app/
│   ├── build.gradle
│   ├── keystore.jks               ← Keystore untuk release signing
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/models/         ← Letakkan model AI di sini
│       ├── cpp/
│       │   ├── CMakeLists.txt
│       │   ├── native-lib.cpp     ← JNI bridge
│       │   ├── pipeline.cpp       ← Super-resolution pipeline
│       │   ├── codeformer.cpp     ← Face restoration
│       │   ├── realesrgan.cpp     ← Background SR
│       │   ├── colornet.cpp       ← Colorization
│       │   ├── face.cpp           ← Face detection
│       │   ├── encoder.cpp        ← Feature encoder
│       │   └── generator.cpp      ← Image generator
│       ├── java/.../MainActivity.kt
│       └── res/layout/            ← UI layouts
├── build.gradle
├── settings.gradle
└── gradlew
```

---

## 🔐 Signing

Release APK ditandatangani menggunakan `app/keystore.jks`.

Untuk produksi, simpan keystore sebagai GitHub Secret:
- `KEYSTORE_BASE64` — keystore di-encode base64
- `KEY_ALIAS` — alias key
- `KEY_PASSWORD` — password key
- `STORE_PASSWORD` — password store

---

## 📄 Lisensi

MIT License — silakan fork dan modifikasi.
