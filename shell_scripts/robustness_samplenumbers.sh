# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
datasets_path=/datasets/human_llm_data_for_experiment # original datasets
samplenumbers_generation_datasets=/datasets/sample_numbers_compare_dataset_detectgpt_npr_dnagpt # compare samples generation dataset (detectgpt、detectnpr、dnagpt)
# detect results folders
detection_results_path=/experiment_results/sample_numbers_compare_results

# open-source
datasets="xsum squad writing" #  
source_models="gemma_7b" #
samplenumbers=10,20,50,100
scenarios="white"


# generate data
# perturbation methods: detectgpt || npr || dna_gpt
for dataset in $datasets; do
  for number in 10 20 50 100; do
    echo `date`, Preparing detectgpt-detectnpr dataset , n_perturbations=${number}  ...
    python ../py_scripts/data_generations/data_generation_perturbation.py --dataset_file ${datasets_path}/${dataset}_${source_models} --output_file ${samplenumbers_generation_datasets}/${dataset}_${source_models} --Generation_methods Perturbation --n_perturbations $number 
  done
done

for dataset in $datasets; do
  for number in 10 20 50 100; do
    echo `date`, Preparing dnagpt dataset , n_regenerations=${number}  ...
    python ../py_scripts/data_generations/data_generation_perturbation.py --dataset_file ${datasets_path}/${dataset}_${source_models} --output_file ${samplenumbers_generation_datasets}/${dataset}_${source_models} --Generation_methods Rewrite --rewrite_model ${source_models} --n_regenerations $number --min_length 55 
  done
done


# detection
detectgpt
for dataset in $datasets; do
  for number in 10 20 50 100; do
    echo `date`, detectgpt detection , n_perturbations=${number}  ...
    python ../py_scripts/baselines/detect_gpt.py --dataset_file ${samplenumbers_generation_datasets}/${dataset}_${source_models} --output_file ${detection_results_path}/${dataset}_${source_models}_perturbate_$number-${scenarios}  --scoring_model_name ${source_models} --n_perturbations $number
  done
done

# detectnpr
for dataset in $datasets; do
  for number in 10 20 50 100; do
    echo `date`, detectnpr detection , n_perturbations=${number}  ...
    python ../py_scripts/baselines/detect_npr.py --dataset_file ${samplenumbers_generation_datasets}/${dataset}_${source_models} --output_file ${detection_results_path}/${dataset}_${source_models}_perturbate_$number-${scenarios}  --scoring_model_name ${source_models} --n_perturbations $number
  done
done

# dnagpt
for dataset in $datasets; do
  for number in 10 20 50 100; do
    echo `date`, dnagpt detection , n_regenerations=${number}  ...
    python ../py_scripts/baselines/dna_gpt.py  --dataset_file ${samplenumbers_generation_datasets}/${dataset}_${source_models} --output_file ${detection_results_path}/${dataset}_${source_models}_regenerate_$number-${scenarios}  --scoring_model_name ${source_models} --n_regenerations $number
  done
done

# fast_detectgpt
for dataset in $datasets; do
  for number in 10 20 50 100; do
    echo `date`, fastdetectgpt detection , n_samples=${number}  ...
    python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${datasets_path}/${dataset}_${source_models} --output_file ${detection_results_path}/${dataset}_${source_models}_sampling_$number-${scenarios}  --reference_model_name ${source_models} --scoring_model_name ${source_models} --n_samples $number
  done
done

# lastde_doubleplus
for dataset in $datasets; do
  for number in 10 20 50 100; do
    echo `date`, fastlmde detection , n_samples=${number}  ...
    python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${datasets_path}/${dataset}_${source_models} --output_file ${detection_results_path}/${dataset}_${source_models}_sampling_$number-${scenarios} --reference_model_name ${source_models} --scoring_model_name ${source_models}  --n_samples $number --embed_size 4 --epsilon 8 --tau_prime 15
  done
done