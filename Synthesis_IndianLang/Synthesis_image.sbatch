#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 6-00:00 # time (D-HH:MM)  setting it for 6 days to run
#SBATCH --mail-user=142102014@smail.iitpkd.ac.in

module load cuda11.2/toolkit/11.2.2
module load cudnn8.0-cuda11.0
module load ex3-modules

source /global/D1/projects/ZeroShot_Word_Recognition/SceneText_Recognition/Vasanthan_STR/bin/activate ###  Give exact path with respect to your env folder in your cluster account

## This .sbatch file should be in the same location that is within your STR***-main folder , where mytrain.py and other codes are located...  hope you understand why I put a ./ before each relevant folder in the argument list

python render_Indian_language_scenetext.py ./Mal_train.txt ./fontlists/Mal_Uniq_RS.txt /global/D1/projects/ZeroShot_Word_Recognition/SceneText_Recognition/STRforIndicLanguages-main/Mal_train Malayalam 1 ./val_256/

