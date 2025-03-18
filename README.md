# Few-Shot-Learning-for-Video-Transcription

## German Lip-Reading with PyTorch AVSR and GLips Dataset

## 📌 Overview  
This project adapts the **PyTorch-based Audio-Visual Speech Recognition (AVSR)** system for **German lip-reading**, utilizing the **GLips dataset**. The adaptation involves **modifying preprocessing, training pipelines, and model architecture** to better capture **German phonetic structures and articulation patterns**.

## 🚀 Key Features

🔹 **GLips Dataset Integration** – Processed parliamentary speech videos for robust lip-reading.  

🔹 **SentencePiece Tokenization** – Adapted the SPM model to German phoneme distributions.  

🔹 **Multimodal Learning** – Utilized both visual (lip movements) and audio cues.  

🔹 **Optimized Training Pipeline** – Custom data augmentation, batch handling, and DDP-based multi-GPU training.  

🔹 **Real-Time Inference** – Implemented efficient frame-wise video processing.  

## 🛠 Installation  

### 💻 Clone the Repository  
```bash
git clone https://github.com/franzjb/Few-Shot-Learning-for-Video-Transcription.git
cd lipreading_fsl
```
### 📦 Install PyTorch and all necessary packages
```bash
pip install torch torchvision torchaudio pytorch-lightning sentencepiece
```
### 🏗 Preprocess Dataset
[Pre-Processing](https://github.com/pytorch/audio/tree/main/examples/avsr/data_prep)

### 📉 Start Trainig and Evaluation
[Training](https://github.com/pytorch/audio/tree/main/examples/avsr)


### 📝 As reference

Real-time ASR/VSR/AV-ASR Examples (PyTorch) [https://github.com/username/another-repo](https://github.com/pytorch/audio/tree/main/examples/avsr)

GLips - German Lipreading Dataset (University of Hamburg) [https://www.fdr.uni-hamburg.de/record/10048](https://www.fdr.uni-hamburg.de/record/10048)


## 🎯 Evaluation

## 📬 Contact

For inquiries or contributions, feel free to reach out or open a pull request.

- 🔗 **GitHub:** [franzjb](https://github.com/franzjb)
- 📧 **Email:** franzjb@icloud.com
