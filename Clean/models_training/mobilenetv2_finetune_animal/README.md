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

the following directory structure will be created inside your project folder when you run the file datasets.py :

```
dataset/
├── train/
│   ├── cat/
│   ├── monkey/
|   ├── snake/
│   └── squirrel/
|   
├── val/
│   ├── cat/
│   ├── monkey/
|   ├── snake/
│   └── squirrel/
```

---

### Download Datasets

Download and unzip the following datasets then rename accordingly.(monkey, cats, squirrel, snake) :

- **Cats:**  
  https://www.kaggle.com/datasets/borhanitrash/cat-dataset

- **Monkeys:**  
  https://www.kaggle.com/datasets/slothkong/10-monkey-species

- **Snakes:**  
  https://www.kaggle.com/datasets/goelyash/165-different-snakes-species

- **Squirrel:**  
  https://www.kaggle.com/datasets/harrybaines/squirrels

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

- `mobilenetv2_animal.h5`

---

## Notes

- To test just run the test.py. change the picture as wanted.
