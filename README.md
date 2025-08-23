Of course. Here is the updated `README.md` with a "Getting Started" section that includes instructions to `git clone` the repository.

-----

# Automatic Audio FX Chain Generation using CLAP

This project provides a Python script that automatically discovers and tunes a chain of audio effects to make a source audio file match a given text prompt. It leverages the Contrastive Language-Audio Pre-training (CLAP) model for semantic understanding and uses Bayesian Optimization for intelligent, efficient parameter searching.

The system can determine which effects to use (e.g., Reverb, Distortion, EQ), in what combination, and with what specific parameters, to best match a creative goal like "a low fidelity radio effect on a voice."

## Features

  - **Text-to-Audio-FX:** Generates complex audio effect presets from simple text descriptions.
  - **Intelligent Search:** Uses Bayesian Optimization (`scikit-optimize`) to efficiently find optimal parameters without brute-forcing all possibilities.
  - **Dynamic FX Activation:** The optimizer intelligently decides which effects to turn ON or OFF for the best result.
  - **High-Quality Objective:** Utilizes a "negative prompt" strategy to refine the evaluation criteria, guiding the search away from harsh or undesirable sounds and toward more musically plausible results.
  - **Multiple Candidates:** The search process identifies and saves multiple high-scoring presets (Top-N), giving you a variety of creative options.

-----

## Getting Started

Follow these steps to set up and run the project.

### 1\. Clone the Repository

First, clone this repository to your local machine using Git.

```bash
git clone https://your_repository_url_here.git
cd your_repository_directory_name
```

### 2\. Create Environment & Install Dependencies

It's recommended to use a Conda environment to manage dependencies. Make sure you have Anaconda or Miniconda installed.

#### Option A: Using `environment.yml` (Recommended)

This file ensures you have the correct Python version and all dependencies.

1.  Create a file named `environment.yml` with the following content:

    ```yaml
    name: fx-search
    channels:
      - pytorch
      - nvidia
      - conda-forge
      - defaults
    dependencies:
      - python=3.10
      - pip
      - pytorch
      - torchvision
      - torchaudio
      - pytorch-cuda=11.8 # Or your CUDA version, e.g., 12.1
      - pip:
        - transformers
        - pedalboard
        - scikit-optimize
        - librosa
        - soundfile
        - numpy
        - tqdm
    ```

2.  Create and activate the Conda environment:

    ```bash
    conda env create -f environment.yml
    conda activate fx-search
    ```

#### Option B: Using `requirements.txt`

1.  Create a file named `requirements.txt` with the following content:

    ```
    torch
    transformers
    pedalboard
    scikit-optimize
    librosa
    soundfile
    numpy
    tqdm
    ```

2.  Create a new Conda environment and install the packages using pip:

    ```bash
    conda create -n fx-search python=3.10
    conda activate fx-search
    pip install -r requirements.txt
    ```

    **Note:** For GPU support, you may need to install PyTorch manually by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/) to match your CUDA version.

-----

## Usage

Run the script from your terminal. All arguments are required unless they have a default value.

### Arguments

| Argument          | Description                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------------- |
| `--audio`         | **Required.** Path to the source audio file (e.g., `clean_vocal.wav`).                                    |
| `--prompt`        | **Required.** The text prompt describing the desired sound (e.g., `"a clear vocal with subtle reverb"`). |
| `--outdir`        | **Required.** The directory where the results (audio files, JSON) will be saved.                         |
| `--model`         | The CLAP model to use from the Hugging Face Hub. (Default: `laion/clap-htsat-unfused`)                   |
| `--top_n`         | The number of top-ranking results to save. (Default: `5`)                                                |
| `--n_calls`       | The total number of iterations for the Bayesian optimization search. (Default: `100`)                      |
| `--use_negative`  | A flag to enable the negative prompt objective function for higher quality results. (Enabled by default) |

### Example

```bash
python main.py \
    --audio "./path/to/my_audio.wav" \
    --prompt "A distorted and aggressive guitar riff" \
    --outdir "./results/aggressive_guitar" \
    --n_calls 150 \
    --top_n 5
```

-----

## Output Structure

The script will create the specified output directory (`--outdir`), which will contain:

  - **`best.wav`**: The single best audio result, ranked by the benchmark (positive prompt) CLAP score.
  - **`rank_2.wav`**, **`rank_3.wav`**, etc.: Other high-scoring candidates from the search, also ranked by their benchmark score.
  - **`results.json`**: A detailed JSON file containing the full presets for each of the top N results, including their final parameters, composite scores, and benchmark CLAP scores.