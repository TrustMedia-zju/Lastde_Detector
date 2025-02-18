# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
datasets_path=/datasets/human_llm_data_for_experiment
perturbation_datasets_path=/datasets/perturbation_data_detectgpt_npr
regeneration_datasets_path=/datasets/regeneration_data_dnagpt

DetectGPT_Generation_methods="Perturbation"
DNAGPT_Generation_methods="Rewrite"
dnagpt_black_rewrite_model="gptj_6b"

# opened-source and closed-source
datasets="xsum squad writing" #  
source_models="gpt2_xl gptneo_2.7b opt_2.7b gptj_6b llama1_13b llama2_13b llama3_8b opt_13b bloom_7b falcon_7b gemma_7b phi2"  

datasets_black="xsum writing reddit" 
source_models_black="gpt2_xl gptneo_2.7b opt_2.7b llama1_13b llama2_13b  llama3_8b opt_13b bloom_7b falcon_7b gemma_7b phi2 gpt4turbo"  


# DetectGPT、NPR white-box
for D in $datasets; do
 for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M} ...
   python ../py_scripts/data_generations/data_generation_perturbation.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${perturbation_datasets_path}/${D}_${M} --Generation_methods ${DetectGPT_Generation_methods} --main_results
 done
done

# DetectGPT、NPR black-box
for M in $source_models_black; do
  echo `date`, Preparing dataset reddit_${M} ...
  python ../py_scripts/data_generations/data_generation_perturbation.py --dataset_file ${datasets_path}/reddit_${M} --output_file ${perturbation_datasets_path}/reddit_${M} --Generation_methods ${DetectGPT_Generation_methods} --main_results
done


# DNA-GPT white-box
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Preparing dataset ${D}_${M} ...
    python ../py_scripts/data_generations/data_generation_perturbation.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${regeneration_datasets_path}/${D}_${M} --Generation_methods ${DNAGPT_Generation_methods} --rewrite_model ${M} --scenario white
  done
done

# DNA-GPT black-box
for D in $datasets_black; do
  for M in $source_models_black; do
    echo `date`, Preparing dataset ${D}_${M} ...
    python ../py_scripts/data_generations/data_generation_perturbation.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${regeneration_datasets_path}/${D}_${M} --Generation_methods ${DNAGPT_Generation_methods} --rewrite_model ${dnagpt_black_rewrite_model} --scenario black 
  done
done