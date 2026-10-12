## MiniVLM-EMIDEC: Containerized Multimodal Medical Vision-Language Pipeline

This repository contains a production-ready, containerized pipeline for **MiniVLM**, a multimodal vision-language model optimized for joint cardiac MRI segmentation and clinical narrative classification using the EMIDEC dataset. 

The architecture integrates a **ResNet50** vision branch with a specialized clinical text encoder (**Bio_ClinicalBERT**) via a **Gated Cross-Attention module**, allowing the model to ground its spatial segmentation decisions in unstructured clinical notes.

---

##  Key Features
- **Environment-Aware Runtime:** Seamlessly switches between local container deployment and Google Colab fallback infrastructure.
- **Automated Ablation Suite:** Successively executes and logs benchmarks for `vision_only`, `text_only`, and `multimodal` configurations.
- **Resource-Optimized Training:** Designed to execute robustly within resource-constrained environments using efficient feature alignment and normalized contrastive loss (NCE).
- **Interpretability & Grounding:** Automatically extracts and maps spatial cross-attention layers to visualize clinical-spatial reasoning.

---

##  Repository Layout
```text
├── cmrminivlm.py         # Main training, ablation, and evaluation script
├── Dockerfile            # CUDA-accelerated runtime build specification
├── requirements.txt      # Python library dependencies
├── .gitignore            # Production tracking filters
└── README.md             # Project documentation

```

---

##  Prerequisites

To deploy this pipeline using Docker with GPU acceleration, the host system requires:

1. **Docker Engine** (v20.10+)
2. **NVIDIA GPU Drivers** compatible with CUDA 12.1+
3. **NVIDIA Container Toolkit** installed to pass host GPUs to container runtime instances.

---

##  How to Run (Production Docker Environment)

### Step 1: Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

```

### Step 2: Prepare the Shared Volumetric Storage

Because datasets and model weights should persist outside the container life-cycle, set up a local host directory and place your data archive inside it:

1. Create a local folder on your host machine:
```bash
mkdir -p ~/minivlm_host_data

```


2. Move your `emidec_training.zip` dataset archive directly into that folder:
```text
~/minivlm_host_data/
└── emidec_training.zip

```



### Step 3: Compile the Docker Image

Build the container using the cached configurations:

```bash
docker build -t cmr-minivlm-pipeline .

```

### Step 4: Execute the Ablation & Training Pipeline

Launch the pipeline container while passing all available system GPUs and binding the data volumes:

```bash
docker run --gpus all \
  -e DOCKER_MODE=true \
  -v ~/minivlm_host_data:/workspace/data \
  cmr-minivlm-pipeline

```

#### Flag Explanations:

* `--gpus all`: Grants the container access to native hardware GPU acceleration.
* `-e DOCKER_MODE=true`: Explicitly forces the execution script to bypass cloud notebook mounting schemes and process automated relative local container directories.
* `-v ~/minivlm_host_data:/workspace/data`: Maps the data folder on your machine directly to `/workspace/data` inside the container.

---

##  Outputs & Results Tracking

All tracking information, serialization checkpoints, and analytics generated inside the container are automatically written back to your host folder (`~/minivlm_host_data/`).

Upon execution completion, the directory layout will look like this:

```text
~/minivlm_host_data/
├── emidec_training.zip
├── checkpoints/
│   ├── vision_only_v2.pth
│   ├── text_only_v2.pth
│   └── multimodal_v2.pth
└── results/
    ├── quantitative_ablation_results.csv   # Aggregated performance tables
    ├── ablation_performance_curves.png     # Validation metrics curves over epochs
    ├── performance_comparison_bars.png     # Comparative bar plots (Dice vs Acc)
    └── spatial_grounding_attention.png     # Attention map alignment visualizations

```

---

##  Fallback Mode: Running inside Google Colab

The underlying execution script contains fallback wrappers for development phases. If `DOCKER_MODE` is left unset or evaluated as `false`, the pipeline defaults to a notebook environment and expects your data archive to be present on Google Drive under:
`/content/drive/MyDrive/MiniVLM_EMIDEC/emidec_training.zip`

```# Set DOCKER_MODE to false to use the notebook environment
export DOCKER_MODE=false

# Expected dataset archive path in Google Drive
/content/drive/MyDrive/MiniVLM_EMIDEC/emidec_training.zip
```
