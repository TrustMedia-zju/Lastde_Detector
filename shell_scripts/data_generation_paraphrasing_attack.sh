# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
dataset_file=/datasets/human_llm_data_for_experiment
datasets="xsum writing reddit"
source_models="llama1_13b opt_13b gpt4turbo gptj_6b"
# results folders
output_file=/datasets/paraphrasing_attack_data


for D in $datasets; do
  for M in $source_models; do
    echo `date`, Preparing dataset ${D}_${M} ...
    python ../py_scripts/data_generations/data_generation_paraphrasing.py --dataset_file ${dataset_file}/${D}_${M} --output_file ${output_file}/${D}_${M}_paws_paraphrasing_attack --paraphrase_model t5_paraphrase_paws
  done
done