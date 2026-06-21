<p align="center">
  <img src="vsn.png" alt="vsnpy dark-orange robotic hand logo" width="100%" max-width="600px"/>
</p>

<p align="center">
  <strong>A high-performance Python vision framework being created to eliminate webcam lag, bypass CPU memory bottlenecks, and stream ultra-fast arrays directly into PyTorch pipelines.</strong>
</p>

---

## 🚀 Overview

When building real-time computer vision pipelines in Python (such as feeding live webcam frames into YOLO, ViT, or custom PyTorch models), the major performance bottleneck is rarely the neural network itself. Instead, systems frequently experience **CPU cache thrashing** and **Python GIL (Global Interpreter Lock) congestion** during image ingestion and frame preprocessing.

`vsnpy` is being developed to solve this exact issue by bringing **Vision + NumPy** directly into harmony. By shifting expensive pixel-level tracking operations to a flat, highly optimized custom C++ backend while maintaining a familiar, array-driven Python interface, `vsnpy` ensures that frame preprocessing loops run **smooth like butter** on standard hardware.

---

## ⚙️ Core Technical Mechanisms (Under Active Development)

### 1. Kernel Operator Fusion (`vs_process_frame`)
Standard computer vision setups handle frame transformations sequentially. For instance, applying a blur and then a binary threshold forces the system to run the entire blur operation, write the intermediate matrix to main system memory (RAM), and then read it back out to perform the thresholding. This results in heavy memory-bus latency.
* **The vsnpy Design:** `vsnpy` fuses consecutive operations into a single execution step. For operations like combined blurs and thresholding (`VSBLURADDBINARY`), pixels are processed entirely within the CPU register cache in a single pass. The array values never bounce back and forth to main RAM unnecessarily.

### 2. Hidden 1-Bit Binary Mask Compression
Traditional frameworks allocate full 8-bit or 24-bit matrices to track processed vision data. 
* **The vsnpy Design:** `vsnpy` compresses spatial tracking data under the hood into a lightweight **1-bit hidden binary mask (0s and 1s)**. Packing boolean mask states into single bits drastically slashes the overall memory footprint of the active ingestion stream. This keeps the data payload small enough to reside inside your hardware's L1/L2 cache lines for maximum execution speed.

### 3. GIL-Free Tracking Loops (`track_pixels()`)
Iterating through arrays using standard Python loops adds immense variable-access overhead and locks up the interpreter.
* **The vsnpy Design:** The entire bounding container search and coordinate computation loop is pushed to a dedicated C++ engine via `track_pixels()`. The framework explicitly releases the Python GIL during the array search, tracking raw pixels at bare-metal speeds and passing the finalized data directly into PyTorch tensor buffers.

---

## 🛠️ API & Core Functions: Pure Vision + NumPy

`vsnpy` bridges the gap between high-level Python syntax and bare-metal C++ speed. It operates on a native `vsarray` structure that mirrors standard NumPy behavior but keeps layout optimizations running smoothly under the hood.

### 1. Core Engine Arrays & Allocation

#### `vsnpy.vsarray(shape, dtype=vsnpy.uint8)`
Allocates a zero-copy memory buffer optimized for high-speed sequential video frame manipulation.
* **Why it matters:** Unlike standard arrays that cause memory-bus latency during continuous webcam ingestion, a `vsarray` locks memory pages directly inside cache lines for instant C++ kernel access.

#### `vsnpy.vs_process_frame(array, operations=[])`
Executes **Kernel Operator Fusion** directly on an incoming frame.
* **Syntax Example:**
  ```python
  # Fuses Gaussian Blur and Binary Thresholding into a single C++ register pass
  fused_mask = vsnpy.vs_process_frame(frame, operations=["VS_BLUR", "VS_BINARY_THRESHOLD"])
Why it matters: Instead of bouncing huge multi-channel color arrays back and forth to main system RAM, the pixel transformations are crunched in one fast pass and instantly crushed down into the lightweight 1-bit hidden binary mask (0s and 1s).

2. High-Speed Frame Tracking
vsnpy.track_pixels(binary_mask, target_color=None)
The main heavy-lifting function of the framework. It searches the hidden 1-bit binary mask to isolate target contours and track movement vectors.

Syntax Example:

Python
# Runs entirely with the Python GIL released
coordinates = vsnpy.track_pixels(fused_mask)
Why it matters: It completely bypasses slow Python loops. The entire bounding search runs inside a flat C++ loop, keeping your frame ingestion running fluidly even on low-spec laptop processors.

3. Deep Learning Handshake
vsnpy.to_pytorch(array)
Instantly casts a processed vsnpy array configuration directly into a native torch.Tensor layout without copying underlying data pointers.

Why it matters: This provides a seamless pipeline where raw video input is grabbed, optimized via C++, compressed into binary tracking maps, and handed straight off to downstream PyTorch models (like YOLO or ViT) with absolutely zero layout duplication overhead.

📊 High-Level Architecture
[Raw Webcam Frame Input]
       │
       ▼  (Zero-Copy Layout Match)
┌────────────────────────────────────────────────────────┐
│            vsnpy Custom C++ Engine Core                │
│                                                        │
│  ► Kernel Operator Fusion (vs_process_frame)           │
│    [Processes blur & threshold inside register cache]   │
│                                                        │
│  ► Layout Compression                                  │
│    [Crushes spatial data down to 1-bit 0/1 binary mask]│
└────────────────────────────────────────────────────────┘
       │
       ▼  (GIL-Released Native Threading)
  track_pixels() ──► [Instant Handshake with PyTorch Tensors]
🎯 Architecture Goals
Smooth Like Butter Delivery: Eliminates frame stuttering and lag on standard computers and laptops by protecting the system from redundant memory allocation traps.

Zero-Copy Performance: The internal vsarray layout is designed to share raw memory pointers natively with scientific computing structures, avoiding internal array duplicating latency before data passes to deep learning layers.

Drop-In Integration: Built to serve as a clean Python framework that accelerates frame ingestion immediately before tensor hand-off.

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
