# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
datasets_path=/datasets/human_original_data
results_path=/datasets/human_llm_data_for_experiment

# open-source
datasets="xsum squad writing reddit" # xsum squad writing 
source_models="gpt2_xl gptneo_2.7b opt_2.7b gptj_6b llama1_13b llama2_13b llama3_8b opt_13b bloom_7b falcon_7b gemma_7b phi2" 

# preparing dataset
for D in $datasets; do
 for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M} ...
   python ../py_scripts/data_generations/data_generation_opensource.py --output_file ${results_path}/${D}_${M} --dataset ${D} --model ${M}
 done
done
exit
