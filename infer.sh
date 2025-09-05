. ./path.sh # set the path

resume="Dongchao/almtokenizer2/ep5.checkpoint"
exp_dir="Dongchao/almtokenizer2"
output_dir="./"
python3 infer.py --resume $resume \
                 --exp_dir $exp_dir \
                 --rank 0 \
                 --output_dir $output_dir 

