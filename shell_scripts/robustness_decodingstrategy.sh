# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
original_dataset=/datasets/human_original_data
differient_strategy_datasets=/datasets/decoding_strategies_data

# detect results folders
detection_results_path=/experiment_results/decoding_strategy_results

# open-source
datasets="xsum squad writing"
source_models="gpt2_xl gptneo_2.7b opt_2.7b"
scenarios="black"
proxy_model=gptj_6b

# =========================================================================
# generating datasets
Top-p=0.96
for D in $datasets; do
 for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M}_Top-p=0.96 ...
   python ../py_scripts/data_generations/data_generation_opensource.py --output_file ${differient_strategy_datasets}/${D}_${M}_Topp --dataset ${D} --model ${M}  --do_top_p  --top_p 0.96
 done
done

# Top-k=40
for D in $datasets; do
 for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M}_Top-k=40 ...
   python ../py_scripts/data_generations/data_generation_opensource.py --output_file ${differient_strategy_datasets}/${D}_${M}_Topk --dataset ${D} --model ${M} --do_top_k --top_k 40
 done
done

# Temperature=0.8
for D in $datasets; do
 for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M}_Temperature=0.8 ...
   python ../py_scripts/data_generations/data_generation_opensource.py --output_file ${differient_strategy_datasets}/${D}_${M}_Temperature --dataset ${D} --model ${M} --do_temperature --temperature 0.8
 done
done
# =========================================================================

# detection
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# statistic methods：likelihood、logrank、entropy、lrr、lastde
for D in $datasets; do
for M in $source_models; do
   echo `date`, statistic methods ${D}_${M}_Top-p=0.96  ...
   python ../py_scripts/baselines/statistic_detect.py --dataset_file ${differient_strategy_datasets}/${D}_${M}_Topp --output_file ${detection_results_path}/${D}_${M}_Topp_${scenarios}  --scoring_model_name ${proxy_model}
 done
done

for D in $datasets; do
for M in $source_models; do
   echo `date`, statistic methods ${D}_${M}_Top-k=40  ...
   python ../py_scripts/baselines/statistic_detect.py --dataset_file ${differient_strategy_datasets}/${D}_${M}_Topk --output_file ${detection_results_path}/${D}_${M}_Topk_${scenarios}  --scoring_model_name ${proxy_model}
 done
done

for D in $datasets; do
for M in $source_models; do
   echo `date`, statistic methods ${D}_${M}_Temperature=0.8 ...
   python ../py_scripts/baselines/statistic_detect.py --dataset_file ${differient_strategy_datasets}/${D}_${M}_Temperature --output_file ${detection_results_path}/${D}_${M}_Temperature_${scenarios}  --scoring_model_name ${proxy_model}
 done
done
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# fast_detectgpt
for D in $datasets; do
 for M in $source_models; do
   echo `date`, fastdetectgpt method ${D}_${M}_Top-p=0.96 ...
   python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${differient_strategy_datasets}/${D}_${M}_Topp --output_file ${detection_results_path}/${D}_${M}_Topp_${scenarios} --reference_model_name ${proxy_model} --scoring_model_name ${proxy_model}
 done
done

for D in $datasets; do
 for M in $source_models; do
   echo `date`, fastdetectgpt method ${D}_${M}_Top-k=40 ...
   python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${differient_strategy_datasets}/${D}_${M}_Topk --output_file ${detection_results_path}/${D}_${M}_Topk_${scenarios} --reference_model_name ${proxy_model} --scoring_model_name ${proxy_model}
 done
done

for D in $datasets; do
 for M in $source_models; do
   echo `date`, fastdetectgpt method ${D}_${M}_Temperature=0.8 ...
   python ../py_scripts/baselines/fast_detect_gpt.py --dataset_file ${differient_strategy_datasets}/${D}_${M}_Temperature --output_file ${detection_results_path}/${D}_${M}_Temperature_${scenarios} --reference_model_name ${proxy_model} --scoring_model_name ${proxy_model}
 done
done
# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # lastde++
for D in $datasets; do
 for M in $source_models; do
   echo `date`, fastlmde method ${D}_${M}_Top-p=0.96 ...
   python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${differient_strategy_datasets}/${D}_${M}_Topp --output_file ${detection_results_path}/${D}_${M}_Topp_${scenarios} --reference_model_name ${proxy_model} --scoring_model_name ${proxy_model}  --embed_size 4 --epsilon 8 --tau_prime 15
 done
done

for D in $datasets; do
 for M in $source_models; do
   echo `date`, fastlmde method ${D}_${M}_Top-k=40 ...
   python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${differient_strategy_datasets}/${D}_${M}_Topk --output_file ${detection_results_path}/${D}_${M}_Topk_${scenarios} --reference_model_name ${proxy_model} --scoring_model_name ${proxy_model}  --embed_size 4 --epsilon 8 --tau_prime 15
 done
done

for D in $datasets; do
 for M in $source_models; do
   echo `date`, fastlmde method ${D}_${M}_Temperature=0.8 ...
   python ../py_scripts/baselines/lastde_doubleplus.py --dataset_file ${differient_strategy_datasets}/${D}_${M}_Temperature --output_file ${detection_results_path}/${D}_${M}_Temperature_${scenarios} --reference_model_name ${proxy_model} --scoring_model_name ${proxy_model}  --embed_size 4 --epsilon 8 --tau_prime 15
 done
done
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>