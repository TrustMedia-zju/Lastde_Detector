# Training-free LLM-generated Text Detection by Mining Token Probability Sequences

This project provides the core code for the two main methods, **Lastde and Lastde++** , as presented in our [paper](https://openreview.net/forum?id=vo4AHjowKi&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions)).

We follow the standard testing procedures outlined in [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt/tree/main) to evaluate each detection method.


<p align="center">
<img src="resources/Lastde_framework.png" width="100%">></a> <br>
</p>

## Environment

- Python3.8
- Pytorch2.0.0
- Other dependencies:
  ```python
  pip install -r requirements.txt
  ```
  (Note: Our experiments were conducted on two RTX 3090 GPUs with 24GB of memory each.)

## Source Models and Proxy Models

The `pretrain_models` directory is used to store open-source models, including those used as proxies or for generating text produced by LLMs. Here, we take `gpt-j-6b` and `Llama-3-8B` as examples, and the model weights can be downloaded from the following addresses:
- [gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b/tree/main)
- [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main)

## Datasets

The main dataset is divided into two parts:
- The `human_original_data` directory contains raw text in json format, with the Xsum dataset (i.e., **xsum.json**) as an example.
- The `human_llm_data_for_experiment` directory stores the complete data used for experiments, with **xsum_llama3_8b.raw_data.json** as an example. This dataset needs to be obtained by running 
    ```python
    python py_scripts/data_generations/data_generation_opensource.py
    ```
    (Note : We have already provided the data here, so there is no need to run) Each complete data entry contains two parts: 'original' (human-written text) and 'sampled' (LLM-generated text), with the content of the two types of text corresponding to each other. The 'sampled' text is generated by using the first 30 tokens of the corresponding 'original' text as prompt input to the source model (in this case, Llama-3-8B) for continuation, and all entries are truncated to the same length.
- The `perturbation_data_detectgpt_npr` and `regeneration_data_dnagpt` directories store the complete data of the DetectGPT/DetectNPR and DNA-GPT detection experiments respectively.
- The remaining directories correspond to the robustness section in our paper.



## Detection

Running **detection_white_box.sh** or **detection_black_box.sh** in `shell_scripts` will trigger white-box and black-box detection, respectively.
```shell
cd shell_scripts

# white-box setting
./detection_white_box.sh 

# black-box setting
./detection_black_box.sh
```

The detection methods include:
- Likelihood, LogRank, Entropy, DetectLRR, **Lastde(ours)**. Results will be saved in `experiment_results/statistic_detection_results`.
- DetectGPT. Results will be saved in `experiment_results/detectgpt_detection_results`.
- DetectNPR. Results will be saved in `experiment_results/npr_detection_results`.
- DNA-GPT. Results will be saved in `experiment_results/dna_gpt_detection_results`.
- Fast-DetectGPT. Results will be saved in `experiment_results/fast_detectgpt_detection_results`.
- **Lastde++(ours)**. Results will be saved in `experiment_results/lastde_doubleplus_detection_results`.

The code for the above detection methods is encapsulated in `py_scripts/baselines`.

Other experimental scripts are located in the `shell_scripts` directory.

## Baselines
We provide the following baseline implementations.

- **sample-based methods:** Likelihood, LogRank, Entropy, DetectLRR, **Lastde(ours)**, Binoculars
- **Distribution-based methods:** DetectGPT, DetectNPR, DNA-GPT, Fast-DetectGPT, **Lastde++(ours)**
- **Plug and Play versions:** Likelihood_tocsin, LogRank_tocsin, DetectLRR_tocsin, Lastde_tocsin, Fast-DetectGPT_tocsin, Lastde++_tocsin
- **supervised-based methods:** RoBERTa_Base, RoBERTa_Large, ReMoDetect

We thank the authors of open source projects and models such as [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt/tree/main), [TOCSIN](https://github.com/Shixuan-Ma/TOCSIN), [ReMoDetect](https://github.com/hyunseoklee-ai/ReMoDetect) and [Binoculars](https://github.com/ahans30/Binoculars).

## Citation
If you find this work useful, you can cite it with the following BibTex entry:
```markdown
@articles{
  xu2025trainingfree,
  title={Training-free {LLM}-generated Text Detection by Mining Token Probability Sequences},
  author={Yihuai Xu and Yongwei Wang and Yifei Bi and Huangsen Cao and Zhouhan Lin and Yu Zhao and Fei Wu},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
}
```
