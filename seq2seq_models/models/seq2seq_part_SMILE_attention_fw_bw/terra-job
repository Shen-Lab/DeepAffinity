#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=seq2seq_SMILE_attention_fw_bw 
#SBATCH --time=10:00:00              
#SBATCH --ntasks=28                  
#SBATCH --mem=30G                  
#SBATCH --output=my_output     
#SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=mostafa_karimi@tamu.edu    #Send all emails to email_address 
#SBATCH --account=122788945864


#First Executable Line
module load Anaconda/3-5.0.0.1
source activate tensorflow-gpu-1.3.0
module load cuDNN/5.1-CUDA-8.0.44
python translate.py --data_dir ./data --train_dir ./check-point --en_vocab_size=100 --fr_vocab_size=100
source deactivate
