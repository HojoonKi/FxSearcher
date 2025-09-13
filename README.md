# FxSearcher: Gradient-free Text-driven Audio Transformation

***Control any audio effect—from commercial VSTs to custom plugins—using only natural language.***

FxSearcher is a powerful Python framework for text-driven audio transformation. Unlike traditional deep learning methods that are limited to simple, differentiable effects, FxSearcher is **gradient-free**. This unique approach allows it to control **any audio FX plugin (e.g., VST, AU)**, including the complex, commercial-grade tools previously inaccessible to automated control.

By leveraging the semantic understanding of the CLAP model and the efficiency of Bayesian Optimization, FxSearcher intelligently discovers the optimal chain of effects and parameter settings to match your creative vision, described in plain text.

Simply provide a source audio file and a prompt like *"Make this voice sound like it's coming from an old, lo-fi radio."* FxSearcher will then search through your available plugins, determine the best combination, and tune their parameters to achieve the desired effect.

## Features

* **🎹 Universal FX Compatibility**
    * Works with your existing library of **indifferentiable, commercial-grade audio plugins**. No need for special "differentiable" versions. FxSearcher gives you automated control over the professional tools you already use.

* **🧠 Gradient-Free Intelligent Search**
    * Employs **Bayesian Optimization** (`scikit-optimize`) to efficiently find optimal parameters without the constraints and limitations of gradient-based methods.

* **🎯 Refined Audio Quality with Guiding Prompts**
    * Utilizes a sophisticated scoring function with both a positive prompt (e.g., *"lo-fi radio"*) and a **Guiding Prompt** (e.g., *"clear, not harsh"*) to steer the search away from undesirable artifacts and towards perceptually high-quality results.

* **🔗 Dynamic FX Chain Generation**
    * The optimizer intelligently activates or bypasses effects in a given chain, discovering the most effective combination for the target sound.

* **📁 Exportable & Reproducible Presets**
    * Provides multiple high-scoring candidates (Top-N) as both processed audio files and **exact parameter presets**. These presets can be saved and loaded into your Digital Audio Workstation (DAW) for full reproducibility.

-----

## Getting Started

Follow these steps to set up and run the project.

### 1\. Clone the Repository

First, clone this repository to your local machine using Git.

```bash
git clone https://github.com/HojoonKi/FxSearcher.git
cd FxSearcher
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
      - pytorch-cuda=12.8 # Or your CUDA version, e.g., 12.1
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
| `--use_guide`  | A flag to enable the guiding prompt objective function for higher quality results. (Enabled by default) |

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

-----

## Acknowledgements

Audio effect processing in this project is powered by [Spotify's Pedalboard](https://github.com/spotify/pedalboard/tree/master?tab=readme-ov-file).