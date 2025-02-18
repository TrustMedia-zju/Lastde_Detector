# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
xsum_gptj_6b_datasets_path=/datasets/human_llm_data_for_experiment/xsum_gptj_6b
xsum_opt_27b_datasets_path=/datasets/human_llm_data_for_experiment/xsum_opt_2.7b
xsum_gemma_7b_datasets_path=/datasets/human_llm_data_for_experiment/xsum_gemma_7b
xsum_gpt4turbo_datasets_path=/datasets/human_llm_data_for_experiment/xsum_gpt4turbo
# results folders
proxy_model_results=/experiment_results/proxy_model_selection_results


# statistic_detect
echo `date`, Preparing statistic_detect  ...
python ../py_scripts/baselines/statistic_detect.py --dataset_file ${xsum_gptj_6b_datasets_path} --output_file ${proxy_model_results}/xsum_gptj_6b_by_gptneo_2.7b --scoring_model_name gptneo_2.7b
python ../py_scripts/baselines/statistic_detect.py --dataset_file ${xsum_opt_27b_datasets_path} --output_file ${proxy_model_results}/xsum_opt_2.7b_by_opt_13b --scoring_model_name opt_13b
python ../py_scripts/baselines/statistic_detect.py --dataset_file ${xsum_gemma_7b_datasets_path} --output_file ${proxy_model_results}/xsum_gemma_7b_by_gpt2_xl --scoring_model_name gpt2_xl
python ../py_scripts/baselines/statistic_detect.py --dataset_file ${xsum_gpt4turbo_datasets_path} --output_file ${proxy_model_results}/xsum_gpt4turbo_by_llama1_13b --scoring_model_name llama1_13b
python ../py_scripts/baselines/statistic_detect.py --dataset_file ${xsum_gpt4turbo_datasets_path} --output_file ${proxy_model_results}/xsum_gpt4turbo_by_llama3_8b --scoring_model_name llama3_8b


# FastDetectGPT
echo `date`, Preparing FastDetectGPT ...
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${xsum_gptj_6b_datasets_path} --output_file ${proxy_model_results}/xsum_gptj_6b_by_gptneo_2.7b --scoring_model_name gptneo_2.7b --reference_model_name gptneo_2.7b
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${xsum_opt_27b_datasets_path} --output_file ${proxy_model_results}/xsum_opt_2.7b_by_opt_13b --scoring_model_name opt_13b --reference_model_name opt_13b
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${xsum_gemma_7b_datasets_path} --output_file ${proxy_model_results}/xsum_gemma_7b_by_gpt2_xl --scoring_model_name gpt2_xl --reference_model_name gpt2_xl
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${xsum_gpt4turbo_datasets_path} --output_file ${proxy_model_results}/xsum_gpt4turbo_by_llama1_13b --scoring_model_name llama1_13b --reference_model_name llama1_13b
python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${xsum_gpt4turbo_datasets_path} --output_file ${proxy_model_results}/xsum_gpt4turbo_by_llama3_8b --scoring_model_name llama3_8b --reference_model_name llama3_8b


# Lastde++
echo `date`, Preparing Lastde++ ...
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${xsum_gptj_6b_datasets_path} --output_file ${proxy_model_results}/xsum_gptj_6b_by_gptneo_2.7b --scoring_model_name gptneo_2.7b  --reference_model_name gptneo_2.7b --embed_size 4 --epsilon 8 --tau_prime 15
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${xsum_opt_27b_datasets_path} --output_file ${proxy_model_results}/xsum_opt_2.7b_by_opt_13b --scoring_model_name opt_13b --reference_model_name opt_13b --embed_size 4 --epsilon 8 --tau_prime 15
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${xsum_gemma_7b_datasets_path} --output_file ${proxy_model_results}/xsum_gemma_7b_by_gpt2_xl --scoring_model_name gpt2_xl --reference_model_name gpt2_xl --embed_size 4 --epsilon 8 --tau_prime 15
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${xsum_gpt4turbo_datasets_path} --output_file ${proxy_model_results}/xsum_gpt4turbo_by_llama1_13b --scoring_model_name llama1_13b --reference_model_name llama1_13b --embed_size 4 --epsilon 8 --tau_prime 15
python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${xsum_gpt4turbo_datasets_path} --output_file ${proxy_model_results}/xsum_gpt4turbo_by_llama3_8b --scoring_model_name llama3_8b --reference_model_name llama3_8b --embed_size 4 --epsilon 8 --tau_prime 15

# DNAGPT
# # generate data 
dnagpt_datasets=/datasets/regeneration_data_dnagpt
# echo `date`, Preparing regeneration data for DNAGPT ...
python ../py_scripts/data_generations/data_generation_perturbation.py --dataset_file ${xsum_gptj_6b_datasets_path} --output_file ${dnagpt_datasets}/xsum_gptj_6b_by_gptneo_2.7b --Generation_methods Rewrite --rewrite_model gptneo_2.7b --scenario black
python ../py_scripts/data_generations/data_generation_perturbation.py --dataset_file ${xsum_opt_27b_datasets_path} --output_file ${dnagpt_datasets}/xsum_opt_2.7b_by_opt_13b --Generation_methods Rewrite --rewrite_model opt_13b --scenario black
python ../py_scripts/data_generations/data_generation_perturbation.py --dataset_file ${xsum_gemma_7b_datasets_path} --output_file ${dnagpt_datasets}/xsum_gemma_7b_by_gpt2_xl --Generation_methods Rewrite --rewrite_model gpt2_xl --scenario black
python ../py_scripts/data_generations/data_generation_perturbation.py --dataset_file ${xsum_gpt4turbo_datasets_path} --output_file ${dnagpt_datasets}/xsum_gpt4turbo_by_llama1_13b --Generation_methods Rewrite --rewrite_model llama1_13b --scenario black
python ../py_scripts/data_generations/data_generation_perturbation.py --dataset_file ${xsum_gpt4turbo_datasets_path} --output_file ${dnagpt_datasets}/xsum_gpt4turbo_by_llama3_8b --Generation_methods Rewrite --rewrite_model llama3_8b --scenario black

# detect
python ../py_scripts/baselines/dna_gpt.py --dataset_file ${dnagpt_datasets}/xsum_gptj_6b_by_gptneo_2.7b --output_file ${proxy_model_results}/xsum_gptj_6b_by_gptneo_2.7b --scoring_model_name gptneo_2.7b --scenario black
python ../py_scripts/baselines/dna_gpt.py --dataset_file ${dnagpt_datasets}/xsum_opt_2.7b_by_opt_13b --output_file ${proxy_model_results}/xsum_opt_2.7b_by_opt_13b --scoring_model_name opt_13b --scenario black
python ../py_scripts/baselines/dna_gpt.py --dataset_file ${dnagpt_datasets}/xsum_gemma_7b_by_gpt2_xl --output_file ${proxy_model_results}/xsum_gemma_7b_by_gpt2_xl --scoring_model_name gpt2_xl --scenario black
python ../py_scripts/baselines/dna_gpt.py --dataset_file ${dnagpt_datasets}/xsum_gpt4turbo_by_llama1_13b --output_file ${proxy_model_results}/xsum_gpt4turbo_by_llama1_13b --scoring_model_name llama1_13b --scenario black
python ../py_scripts/baselines/dna_gpt.py --dataset_file ${dnagpt_datasets}/xsum_gpt4turbo_by_llama3_8b --output_file ${proxy_model_results}/xsum_gpt4turbo_by_llama3_8b --scoring_model_name llama3_8b --scenario black
