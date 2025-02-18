# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
original_dataset=/datasets/human_llm_data_for_experiment/xsum_gpt4turbo
differient_length_text_datasets=/datasets/response_length_dataset/xsum_gpt4turbo # (differient length xsum_gpt4turbo)

# detect results folders
detection_results_path=/experiment_results/response_length_results


# generate data
for length in 30 50 120 150 160; do
    echo `date`, Preparing cutoff response differient length dataset , length=${length}  ...
    python ../py_scripts/data_generations/data_generation_response_length.py --dataset_file ${original_dataset} --output_file ${differient_length_text_datasets} --response_length $length
done

# detection 
# statistis_methods
for length in 30 50 120 150 160; do
    echo `date`, detecting statistis methods , length=${length}  ...
    python ../py_scripts/baselines/statistic_detect.py --dataset_file ${differient_length_text_datasets}_length_${length} --output_file ${detection_results_path}/xsum_gpt4turbo_length_${length} --scoring_model_name gptj_6b
done

# detectgpt method
# perturbation data
detectgpt_datasets=/datasets/perturbation_data_detectgpt_npr
for length in 30 50 120 150 160; do
    echo `date`, Preparing detectgpt dataset , length=${length}  ...
    python ../py_scripts/data_generations/data_generation_perturbation.py --dataset_file ${differient_length_text_datasets}_length_${length} --output_file ${detectgpt_datasets}/xsum_gpt4turbo_length_${length} --Generation_methods Perturbation --scenario black  --min_length $length
done

# # detection
for length in 30 50 120 150 160; do
   echo `date`, detecting detectgpt method , length=${length}  ...
   python ../py_scripts/baselines/detect_gpt.py  --dataset_file ${detectgpt_datasets}/xsum_gpt4turbo_length_${length} --output_file ${detection_results_path}/xsum_gpt4turbo_length_${length}  --scoring_model_name gptj_6b --scenario black 
done

for length in 30 50 120 150 160; do
   echo `date`, detecting detectgpt method , length=${length}  ...
   python ../py_scripts/baselines/detect_npr.py  --dataset_file ${detectgpt_datasets}/xsum_gpt4turbo_length_${length} --output_file ${detection_results_path}/xsum_gpt4turbo_length_${length}  --scoring_model_name gptj_6b --scenario black 
done

# fastdetectgpt
for length in 30 50 120 150 160; do
    echo `date`, detecting fast_detect_gpt methods , length=${length}  ...
    python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${differient_length_text_datasets}_length_${length} --output_file ${detection_results_path}/xsum_gpt4turbo_length_${length} --scoring_model_name gptj_6b --reference_model_name gptj_6b
done

# lastde++
for length in 30 50 120 150 160; do
    echo `date`, detecting lastde_doubleplus methods , length=${length}  ...
    python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${differient_length_text_datasets}_length_${length} --output_file ${detection_results_path}/xsum_gpt4turbo_length_${length} --scoring_model_name gptj_6b --reference_model_name gptj_6b --embed_size 3 --epsilon 1  --tau_prime 10
done
