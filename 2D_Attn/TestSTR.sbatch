#!/bin/bash
#SBATCH -p dgx2q
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00 # time (D-HH:MM)  setting it for 6 days to run
#SBATCH --mail-user=142102014@smail.iitpkd.ac.in

module load cuda11.2/toolkit/11.2.2
module load cudnn8.0-cuda11.0
module load ex3-modules


source /global/D1/projects/ZeroShot_Word_Recognition/SceneText_Recognition/Vasanthan_STR/bin/activate ###  Give exact path with respect to your env folder in your cluster account

## This .sbatch file should be in the same location that is within your STR***-main folder , where mytrain.py and other codes are located...  hope you understand why I put a ./ before each relevant folder in the argument list

python3.6 test_attn.py --model ./output/attn_model_withoutglyphs/best_model_malayalam_attn.pth --test_data ../StarNet_model/finetune --display_result ILST_attn_withoutglyphs.txt --lexicon ../Attn_model/malayalam_letters.txt


 echo "Testing complete"
