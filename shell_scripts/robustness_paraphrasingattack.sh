# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
paraphrasing_attack_data_path=/datasets/paraphrasing_attack_data
datasets="xsum writing reddit"
white_source_models="llama1_13b opt_13b gptj_6b"
black_source_models="llama1_13b opt_13b gpt4turbo"
# results folders
paraphrasing_attack_results=/experiment_results/paraphrasing_attack_results


# white
for D in $datasets; do
  for M in $white_source_models; do
    echo `date`, Preparing dataset ${D}_${M} ...
    python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${paraphrasing_attack_data_path}/${D}_${M}_paws_paraphrasing_attack --output_file ${paraphrasing_attack_results}/${D}_${M}_paws_paraphrasing_attack_white --reference_model_name ${M} --scoring_model_name ${M}
    python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${paraphrasing_attack_data_path}/${D}_${M}_paws_paraphrasing_attack --output_file ${paraphrasing_attack_results}/${D}_${M}_paws_paraphrasing_attack_white --reference_model_name ${M} --scoring_model_name ${M} --embed_size 4 --epsilon 8 --tau_prime 15
  done
done

# black
for D in $datasets; do
  for M in $black_source_models; do
    echo `date`, Preparing dataset ${D}_${M} ...
    python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${paraphrasing_attack_data_path}/${D}_${M}_paws_paraphrasing_attack --output_file ${paraphrasing_attack_results}/${D}_${M}_paws_paraphrasing_attack_black --reference_model_name gptj_6b --scoring_model_name gptj_6b
    python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${paraphrasing_attack_data_path}/${D}_${M}_paws_paraphrasing_attack --output_file ${paraphrasing_attack_results}/${D}_${M}_paws_paraphrasing_attack_black --reference_model_name gptj_6b --scoring_model_name gptj_6b --embed_size 4 --epsilon 8 --tau_prime 15
  done
done
