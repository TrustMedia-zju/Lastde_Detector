# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
datasets_path=/datasets/human_llm_data_for_experiment
# results folders
lastde_doubleplus_results_path=/experiment_results/lastde_doubleplus_hyperparameters_results

dataset_detail_path="xsum_gpt4turbo reddit_gpt4turbo squad_llama1_13b writing_gemma_7b"
scenarios="black"
proxy_model="gptj_6b"

# window_size(embed_szie)
embed_size=2,3,4,5,6
# epsilon
epsilon=1,4,6,8,10
# tau_prime
tau_prime=5,10,15,20,25


# test window_size(embed_size)
for D in $dataset_detail_path; 
  do
  for size in 2 3 4 5 6
    do
    echo `date`, Preparing dataset embed_size=${size} ${D} ...
    python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${datasets_path}/${D} --output_file ${lastde_doubleplus_results_path}/${D}_windowsize-${size} --reference_model_name ${proxy_model} --scoring_model_name ${proxy_model}  --embed_size $size --epsilon 8 --tau_prime 15
  done
done

# test epsilon
for D in $dataset_detail_path; 
  do
  for epsilon in 1 4 6 8 10
    do
    echo `date`, Preparing dataset epsilon=${epsilon} ${D} ...
    python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${datasets_path}/${D} --output_file ${lastde_doubleplus_results_path}/${D}_epsilon-${epsilon}  --reference_model_name ${proxy_model} --scoring_model_name ${proxy_model}  --embed_size 4 --epsilon $epsilon --tau_prime 15
  done
done

# test tau_prime
for D in $dataset_detail_path; 
  do
  for tau in 5 10 15 20 25
    do
    echo `date`, Preparing dataset tau_prime=${tau} ${D} ...
    python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${datasets_path}/${D} --output_file ${lastde_doubleplus_results_path}/${D}_tauprime-${tau}  --reference_model_name ${proxy_model} --scoring_model_name ${proxy_model}  --embed_size 4 --epsilon 8 --tau_prime $tau
  done
done
