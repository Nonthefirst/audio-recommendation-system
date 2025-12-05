# Audio Recommendation System
This repository contains an end-to-end audio-based music recommendation system built with Streamlit, PyTorch, and Librosa. The system extracts deep audio embeddings using a contrastive-learning CNN encoder, compares them against a large library of precomputed embeddings, and returns the most similar songs based on cosine similarity. 

```bash
audio-recommendation-system/
├── main.py                  # Streamlit application entry point
├── my_functions.py          # Custom models and functions definition
├── README.md                # Project documentation
├── LICENSE                  # Open-source license
├── .gitignore               # Git ignore configuration
│
└── data/                    # Data and model directory
    ├── music/               # Audio data and precomputed assets
    │   ├── best_contrastive_temp002.zip        # Split-compressed model checkpoint (must be extracted to .pth)
    │   ├── embeddings_contrastive002.npy       # Precomputed audio embeddings for similarity search
    │   └── fma_metadata_clean_CNN.csv          # Cleaned FMA metadata (track IDs, titles, artists, etc.)
    │
    └── ...
```
# Quick Start

## 1. Unzip
Due to its large size (~300 MB), the model file (best_contrastive_temp002.pth) is distributed as a split ZIP archive located at data/best_contrastive_temp002.zip.
You must fully extract all parts before running the app.

```
data/
├── best_contrastive_temp002.zip
├── best_contrastive_temp002.zip.001
├── best_contrastive_temp002.zip.002
└── best_contrastive_temp002.zip.003
```

### Windows
1. Download [360 zip](https://www.360totalsecurity.com/en/360zip/).
2. Right-click best_contrastive_temp002.zip.
3. Select "Extract here" or "Extract to…".

### Macs
1. Download [KeKa](https://www.keka.io/en/).
2. Open Keka, then drag best_contrastive_temp002.zip into the Keka window.
3. Extraction will begin automatically—Keka handles all split parts seamlessly.

### Linux
```
# Navigate to the data directory
cd data

# Install unzip if not already present
sudo apt update && sudo apt install unzip

# Rename split files to standard .z01, .z02, .z03 format
mv best_contrastive_temp002.zip.001 best_contrastive_temp002.z01
mv best_contrastive_temp002.zip.002 best_contrastive_temp002.z02
mv best_contrastive_temp002.zip.003 best_contrastive_temp002.z03

# Extract the archive
unzip best_contrastive_temp002.zip
```
## 2. Dependences
This project relies on the following Python packages. We recommend using a virtual environment (e.g., venv or conda) to avoid conflicts.

### Required Packages

| Package        | Purpose |
|----------------|--------|
| `streamlit`    | Web app interface |
| `pandas`       | Data loading and manipulation |
| `numpy`        | Numerical computing |
| `librosa`      | Audio loading and feature extraction (e.g., spectrograms) |
| `torch`        | Deep learning framework (PyTorch) |
| `scikit-learn` | Cosine similarity computation |
| `glob`         | File path discovery (built-in, no install needed) |
| `os`           | File system operations (built-in) |

> 🔸 `my_functions.py` is a local module in your project that defines the `Cnn14_emb64_Spec` model architecture. Ensure it’s in the same directory as your main script.


## 3. Run the App

Once all dependencies are installed and the model is extracted, launch the Streamlit application:

```bash
streamlit run main.py
```
You can view [Streamlit documentations](https://docs.streamlit.io/) for any error.

## 4. How to Use the App

Once the app is running (`streamlit run main.py`), you'll see a clean interface in your browser. Here's how to get music recommendations:

### 1. Choose Your Input Method
You can either:
- **Upload your own audio file** (supports `.wav`, `.mp3`, `.flac`)
- **Select a sample song** from our built-in demo library (located in `data/music/`)

> Note: Uploaded audio will be automatically trimmed or padded to **30 seconds** for consistent feature extraction.

### 2. Adjust Recommendation Settings (Optional)
Use the **sidebar** to control how many recommendations you want:
- Slider: **Number of recommendations** (5–20, default: 10)

### 3. Get Recommendations
Click the **Get Recommendations** button to:
- Extract deep audio features using our contrastive-learning CNN
- Compare against a precomputed library of song embeddings
- Fetch the most acoustically similar tracks using **cosine similarity**

### 4. View & Export Results
After processing, you’ll see:
- A ranked list of recommended songs with **title, artist, genre, and similarity score**
- An **interactive detail view** for each recommendation (via expandable cards)
- Options to:
  - **Visualize** your audio’s 64-dimensional embedding
  - **Download** the embedding as a JSON file for further analysis

### Tips
- If no recommendations appear, ensure all data files (`embeddings_contrastive002.npy`, `fma_metadata_clean_CNN.csv`, and sample songs) are in the correct directories.
- For best results, use audio clips with clear musical content (avoid silence or heavy noise).

Enjoy discovering new music through sound—not just metadata!

# Acknowledgements

This project builds upon publicly available resources and research code:

- Audio embeddings were trained using the **FMA-small** dataset from the [Free Music Archive (FMA)](https://github.com/mdeff/fma), a large open dataset for music information retrieval research.
- The neural network architecture `Cnn14_emb64_Spec` is adapted from the **PANNs (Pretrained Audio Neural Networks)** framework originally developed by Kong et al. and released in the repository [audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn).

We gratefully acknowledge the authors of these projects for making their data and code openly available. If you use this work, please also cite the original publications:

> Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017, September 5). FMA: A dataset for Music Analysis. arXiv.org. https://arxiv.org/abs/1612.01840 

> Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020, August 23). Panns: Large-scale pretrained audio neural networks for audio pattern recognition. arXiv.org. https://arxiv.org/abs/1912.10211 

