# PEO: Training-Free Aesthetic Quality Enhancement in Pre-Trained Text-to-Image Diffusion Models with Prompt Embedding Optimization

[![arXiv](https://img.shields.io/badge/arXiv-Paper-B31B1B)](https://www.arxiv.org/pdf/2510.02599)


## 📄 Abstract

This paper introduces a novel approach to aesthetic quality improvement in pre-trained text-to-image diffusion models when given a simple prompt. Our method, dubbed Prompt Embedding Optimization (PEO), leverages a pre-trained text-to-image diffusion model as a backbone and optimizes the text embedding of a given simple and uncurated prompt to enhance the visual quality of the generated image. We achieve this by a tripartite objective function that improves the aesthetic fidelity of the generated image, ensures adherence to the optimized text embedding, and minimal divergence from the initial prompt. The latter is accomplished through a prompt preservation term. Additionally, PEO is training-free and backbone-independent. Quantitative and qualitative evaluations confirm the effectiveness of the proposed method, exceeding or equating the performance of state-of-the-art text-to-image and prompt adaptation methods.

![](assets/teaser.svg)

### 🛠️ Dependencies 

```
virtualenv peo_venv
source peo_venv/bin/activate
pip install -r requirements.txt
```

Initialize an Accelerate environment with:

```
accelerate config
```

Download `sac+logos+ava1-l14-linearMSE.pth` from [here](https://github.com/christophschuhmann/improved-aesthetic-predictor).
Then run

```
wget https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz
mv bpe_simple_vocab_16e6.txt.gz .conda/envs/peo-venv/lib/python3.x/site-packages/hpsv2/src/open_clip
```

### ⚡ Run PEO using:

```
python src/peo/__main__.py \
--prompt_path="datasets/prompts.txt" \
--model_name="stable-diffusion-v1-5/stable-diffusion-v1-5" \
--seed=7885661233 \
--logdir="logdir" \
--optimizer_type="Adam" \
--aes=1.0 \
--text_image_sim=0.5 \
--text_sim=0.5 \
--hpsv2=0.0 \
--optimizer_lr=0.01 \
--guidance_scale=7.5 \
--num_inference_steps=14
```

where prompts.txt should contain prompts to run PEO on. 

## 📐 Overview of Prompt Embedding Optimization (PEO)
![](assets/method.svg)

## 🖼️ Results

### Optimization process of PEO
![](assets/fig_9.svg)

### Comparison with baseline
![](assets/fig_4.svg)

### Complex prompts and PEO
![](assets/fig_10.svg)

### Ablation on the loss function
![](assets/fig_7.svg)

### Optimization algorithms and PEO
![](assets/fig_3_sup.svg)

### Hyper-parameter search for the learning rate
![](assets/fig_2_sup.svg)

### Failure Cases
![](assets/fig_14.svg)

## Citations

If PEO contributes to your research, please cite the paper:

```
@article{margaryan2025peo,
  title   = {PEO: Training-Free Aesthetic Quality Enhancement in Pre-Trained Text-to-Image Diffusion Models with Prompt Embedding Optimization},
  author  = {Hovhannes Margaryan and Bo Wan and Tinne Tuytelaars},
  journal = {arXiv preprint arXiv:2510.02599},
  year    = {2025},
}
```

## Acknowledgments

We thank the authors of [Promptist](https://huggingface.co/microsoft/Promptist) for providing the dataset subsets from DiffusionDB and COCO used in their evaluation.
We also acknowledge the broader open-source ecosystem (e.g., PyTorch, Diffusers, Hugging Face, etc.) that made our research possible.


## Contact

This repository is currently being refactored. For any issues and questions please open a GitHub issue or email us at [marg.hovo@gmail.com](mailto:marg.hovo@gmail.com).

