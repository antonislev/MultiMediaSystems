# 📦 Video Compression using Frame Differencing, Huffman Coding & Motion Compensation

This project explores basic video compression techniques including:

- **Frame Differencing**
- **Huffman Encoding**
- **Motion Compensation**

It is developed as part of the "Multimedia Systems" university course.

---

## 📁 Contents

- `mms.py`: Main code with all stages (preprocessing, compression, analysis)
- `frames/`: Folder containing individual video frames
- `residuals/`: Residual frames computed from frame differencing
- `charts/`: Output charts used for analysis (frequency histograms, pie charts, etc.)
- `video.avi`: Input video used for processing
- `README.md`: This documentation

---

## 🚀 Methodology

### 1. Frame Differencing
We convert each frame to grayscale and compute the difference between each consecutive frame to identify motion.

### 2. Huffman Coding
Using the residual frames, we calculate the frequency of pixel values and build a custom Huffman dictionary to compress the data efficiently.

### 3. Motion Compensation
Motion vectors between frames are computed and used to predict new frames, further reducing the amount of data to be stored.

---

## 📊 Results

- **Original Size**: ~1.87 Gbits  
- **After Huffman**: ~690 Mbits  
- **After Motion Compensation**: ~31.9 Mbits  
- **Compression Ratio**: ~2.71x (Huffman), ~2.08x (Motion Comp)

### Visualization Charts Included:

- Frequency Histogram of Residuals
- Pie Charts of Compression Gains
- Bar Chart Comparison: Raw vs Huffman vs Motion Compensation

---

## 🛠️ Requirements

- Python 3.x
- OpenCV (`cv2`)
- Matplotlib
- NumPy
- Collections

```bash
pip install opencv-python matplotlib numpy
