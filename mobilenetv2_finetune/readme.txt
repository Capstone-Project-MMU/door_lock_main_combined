# MobileNetV2 Fine-Tuning for Human, Animal, and None Classification

This project fine-tunes MobileNetV2 to classify images into three categories: `human`, `animal`, and `none`. It is optimized to run on devices like the Raspberry Pi using TensorFlow Lite.

---

## Environment Setup

```bash
conda create --name mn python=3.10 tensorflow
conda activate mn
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

Create the following directory structure manually inside your project folder:

```
dataset/
├── train/
│   ├── human/
│   ├── animal/
│   └── none/
├── val/
│   ├── human/
│   ├── animal/
│   └── none/
```

---

### Download Datasets

Download and unzip the following datasets:

- **Human (LFW):**  
  https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

- **Animal (AFHQ):**  
  https://www.kaggle.com/datasets/andrewmvd/animal-faces

- **None (Room Images):**  
  https://www.kaggle.com/datasets/robinreni/house-rooms-image-dataset

After unzipping, place them in a folder like `archive/` or any location of your choice.

---

## Directory Configuration

Open `datasets.py` and **update all directory paths** to use your **absolute path** (e.g., `/Users/yourname/your/project/archive/...`).

This ensures all dataset scripts can locate and organize the images correctly.

---

## Train the Model

Once everything is set up, run the training script:

```bash
python fine_tune.py
```

This will:
- Fine-tune MobileNetV2
- Save the model as `.h5`
- Export TFLite versions (including quantized)

---

## Output Files

After successful training, the following will be created:

- `mobilenetv2_human_animal_none.h5`
- `mobilenetv2_model.tflite`
- `mobilenetv2_model_quant.tflite`

---

## Notes

- To test just run the test.py. change the picture as wanted.
