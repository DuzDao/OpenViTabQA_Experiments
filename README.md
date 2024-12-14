# Open-ViTabQA Experiments: Fine-tuning models for Question Answering on Tabular Data

This repo demonstrates how to fine-tune the pretrained language model for question answering on tabular data, using the Open-ViTabQA dataset.

## Project Structure

The project is organized into the following directories:

```
OpenViTabQA_Experiments/
├── data/
│   ├── raw/
│   │   ├── qas_train.json
│   │   ├── qas_dev.json
│   │   ├── qas_test.json
│   │   └── table.json
│   ├── processed/
│   │   ├── train.json
│   │   ├── dev.json
│   │   └── test.json
│   └── utils.py
├── src/
│   ├── models/
│   │   ├── vit5.py
│   │   └── __init__.py
│   ├── datasets/
│   │   ├── vitabqa_dataset.py
│   │   └── __init__.py
│   ├── trainers/
│   │   ├── vit5_trainer.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── config.py
│   │   └── __init__.py
│   ├── main.py
│   └── __init__.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

*   **`data/`**: Contains all data-related files.
    *   **`raw/`**: Stores the original data files (JSON format) obtained from the GitHub repository.
    *   **`processed/`**: Stores the preprocessed data, ready for training.
    *   **`utils.py`**: Contains utility functions for data preprocessing, such as `flatten_html_table` and `format_table`.
*   **`src/`**: Contains the main source code.
    *   **`models/`**: Contains the definition and logic for the models.
        *   **`vit5.py`**: Defines the ViT5 model and its fine-tuning logic.
    *   **`datasets/`**: Contains the definition of the datasets.
        *   **`vitabqa_dataset.py`**: Defines the custom `Dataset` class for ViTabQA.
    *   **`trainers/`**: Contains the training logic.
        *   **`vit5_trainer.py`**: Defines the training logic for ViT5, including optimizer, scheduler, and evaluation.
    *   **`utils/`**: Contains utility functions.
        *   **`metrics.py`**: Contains functions for calculating evaluation metrics (EM, F1, ROUGE-1, METEOR).
        *   **`config.py`**: Stores the project's configurations (not implemented in this version).
    *   **`main.py`**: The main script to run the training and evaluation process.
*   **`requirements.txt`**: Lists the required Python packages.
*   **`README.md`**: This file, providing an overview of the project.
*   **`LICENSE`**: Specifies the license for the project (e.g., MIT, Apache 2.0).
*   **`.gitignore`**: Lists files and directories that should be ignored by Git.

## Setup

### 1. Clone the Repository

```bash
git clone <repository_url>
cd OpenViTabQA_Experiments
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Resources

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

### 4. Prepare Data

First, create the `raw` and `processed` directories inside the `data` directory:

```bash
mkdir data/raw
mkdir data/processed
```

Next, clone the dataset from the provided GitHub repository into the `data/raw` directory:

```bash
git clone https://github.com/DuzDao/Open-ViTabQA.git data/raw
```

```bash
mv data/raw/Open-ViTabQA/*.json data/raw/
rm -rf data/raw/Open-ViTabQA
```


Finally, run the following command to preprocess the data:

```bash
python data/utils.py
python data/process_data.py
```

This will create the processed data files (`train.json`, `dev.json`, `test.json`) in the `data/processed/` directory.

## Training

To fine-tune the ViT5 model, run the `src/main.py` script with the desired arguments. Here's an example:

```bash
python src/main.py \
    --batch_size 16 \
    --num_epochs 5 \
    --output_dir vit5_output \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 100 \
    --save_steps 500 \
    --eval_steps 500 \
    --max_length 512
```

**Arguments:**

*   `--train_data_path`: Path to the training data JSON file (default: `data/processed/train.json`).
*   `--val_data_path`: Path to the validation data JSON file (default: `data/processed/dev.json`).
*   `--pretrained_model_name`: Name of the pretrained T5 model (default: `t5-small`).
*   `--batch_size`: Batch size for training and validation (default: 16).
*   `--learning_rate`: Learning rate for the optimizer (default: 5e-5).
*   `--weight_decay`: Weight decay for the optimizer (default: 0.01).
*   `--num_epochs`: Number of training epochs (default: 5).
*   `--gradient_accumulation_steps`: Number of steps to accumulate gradients before performing an update (default: 1).
*   `--warmup_steps`: Number of warmup steps for the learning rate scheduler (default: 100).
*   `--output_dir`: Directory to save model checkpoints (default: `vit5_output`).
*   `--save_steps`: Number of training steps between saving model checkpoints (default: 500).
*   `--eval_steps`: Number of training steps between evaluating on the validation set (default: 500).
*   `--load_checkpoint`: Path to a checkpoint to load (optional).
*   `--device`: Device to use for training (CPU or CUDA) (default: `cuda` if available, otherwise `cpu`).
*   `--max_length`: Maximum length for input sequences (default: 512).

## Evaluation

During training, the model will be evaluated on the validation set at every `eval_steps`. The evaluation metrics (EM, F1, ROUGE-1, and METEOR) will be printed to the console and saved in the `training_log.json` file in the `output_dir`.

## Checkpoints

Model checkpoints will be saved in the `output_dir` at every `save_steps`. You can use the `--load_checkpoint` argument to resume training from a specific checkpoint.

## Contributing

Contributions to this project are welcome. Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the [MIT License](LICENSE).
```

**Cách sử dụng:**

1.  **Tạo file `README.md`:** Tạo một file có tên `README.md` trong thư mục gốc của dự án.
2.  **Copy nội dung:** Copy toàn bộ nội dung markdown ở trên vào file `README.md`.
3.  **Tùy chỉnh:** Bạn có thể tùy chỉnh nội dung này cho phù hợp với dự án của mình.

**Giải thích:**

*   **Giới thiệu:** Mô tả ngắn gọn về dự án.
*   **Cấu trúc dự án:** Giải thích chi tiết về cấu trúc thư mục và các file trong dự án.
*   **Cài đặt:** Hướng dẫn cách cài đặt các thư viện và chuẩn bị dữ liệu.
*   **Huấn luyện:** Hướng dẫn cách chạy script `main.py` và các tham số có thể tùy chỉnh.
*   **Đánh giá:** Giải thích về quá trình đánh giá và các metrics được sử dụng.
*   **Checkpoint:** Giải thích về cách lưu và tải checkpoint.
*   **Đóng góp:** Hướng dẫn cách đóng góp vào dự án.
*   **Giấy phép:** Thông tin về giấy phép mã nguồn mở.

Với file `README.md` này, người khác sẽ dễ dàng hiểu và sử dụng dự án của bạn. Nếu bạn có bất kỳ câu hỏi nào khác, đừng ngần ngại hỏi nhé!
