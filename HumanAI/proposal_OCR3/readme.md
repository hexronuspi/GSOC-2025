##### Note: The .ipynb is not getting rendered in Github, please use .pdf converted code(.ipynb) or download and load it into colab to view it

All the weights are stored in google drive and will be downloaded when !gdown cells are triggered.

#### End-to-end Handwritten Text Recognition for Early Modern Spanish Documents

Our goal is to build a robust model for optical text recognition tailored to digitized early modern Spanish manuscripts. We’re leaning on transformer-based or self-supervised architectures, such as TrOCR or similar, to tackle full-document recognition. This means not just extracting text, but also considering the document’s structure: layout, reading order, and more. These manuscripts are tricky—handwritten with faded ink—so we need a system that’s adaptable and precise.

##### Dataset

We begin by splitting the training dataset, which consists of 12,008 old Spanish handwritten manuscripts for training and 3,002 printed Spanish manuscripts for testing (IIT ISM AI-of-GOD 3.0 Dataset). We write a custom dataset class that reads images, converts the Spanish Image Dataset to RGB, and tokenizes text transcriptions. Padding tokens are replaced with `-100` so the loss function ignores them. Next, we instantiate a `TrOCRProcessor` and build both training and evaluation datasets.

##### Model

We then load a `VisionEncoderDecoderModel` (Microsoft’s TrOCR base), set decoder and pad tokens, and specify beam search parameters. Using `Seq2SeqTrainingArguments`, we define batch sizes, evaluation intervals, logging steps, and other hyperparameters. A custom `compute_metrics` function calculates the average Word Error Rate (WER) via `jiwer`. Finally, we wrap everything in a `Seq2SeqTrainer`, supply the model, processor, datasets, and data collator, and run `trainer.train()`.

##### Results

We achieved a WER of 0.901519 on the train set, establishing our baseline for handwritten old Spanish OCR detection. For printed OCR detection, we used the Spanish T-OCR Large Model (due to its training set being printed Spanish sentences). Although the image detection model occasionally required corrections, these were addressed by applying a T5 model on the text after fine-tuning it using a custom algorithm for grammar correction.

### Table 1: Overview of the Hardware Setup

| Parameter         | Value    |
|-------------------|----------|
| CPU count         | 6        |
| Logical CPU count | 12       |
| GPU count         | 1        |
| GPU type          | NVIDIA L4 |

### Table 2: Training and Evaluation Metrics

| Parameter           | Value                  |
|---------------------|------------------------|
| eval/loss           | 0.6435500979423523     |
| train_loss          | 1.5594333657409795     |
| train_runtime       | 4,466.3357             |
| train/epoch         | 2                      |
| train/learning_rate | 0.00000013315579227696 |
| train/loss          | 0.528                  |

##### Algorithm for Grammar Correction Using T5

This dataset was created by collecting a large corpus of Spanish sentences from the train transcriptions. In the first step, “Normalization,” the algorithm systematically replaced and removed characters prone to confusion or misuse. For example, letters like `v` were changed to `u`, `z` to `c`, and `y` to `i`. Accents were removed using a unidecode-like method, and some letters were interchanged (e.g., `b ↔ v`) to introduce realistic variation.

The second step involved further modifying the corpus by removing, interchanging, or randomly splitting characters in words. The occurrences of key letters (`a`, `e`, `o`, `s`, `i`) were removed or swapped, and occurrences involving the letter `u` were manipulated. The algorithm also introduced letter duplications and swaps (e.g., `g ↔ j`, `s ↔ z`), occasionally reintroduced uppercase letters in about 50% of words, and randomly added or removed accents in around 20% of words.

Ultimately, the algorithm was designed to produce a diverse array of artificial errors that reflect realistic mistakes seen during manual checking in Spanish text. These include missing letters, swapped letters, accent misplacements, and duplicated characters. By simulating these errors, the resulting dataset serves as a robust training and testing resource for grammar-correction models, enabling them to learn and correct common Spanish-language mistakes more effectively.

##### Line Segmentation

We implemented an A* algorithm to extract text lines from binary images. Our approach initializes with the binary image, start/end coordinates, and an optional mask, then uses a Manhattan heuristic and eight-directional moves with tailored weights to navigate through regions, effectively balancing path length with pixel value penalties.

We preprocessed the image by converting it to grayscale, applying adaptive thresholding, and performing morphological cleanup. A projection profile identified text line regions that were merged based on gap criteria. Using the A* algorithm, we extracted optimal paths within these regions, allowing us to crop precise text line segments.
