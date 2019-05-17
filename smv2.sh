game=./rts/game_MC/game \
model_file1=./rts/game_MC/model \
model_file2=./rts/game_MC/oppo_pool \
model=actor_critic \
python3 smv2.py \
    --num_games 1024 \
    --batchsize 128 \
    --tqdm \
    --players "fs=50,type=AI_NN;fs=50,type=AI_NN" \
    --trainer_stats winrate \
    --additional_labels id,last_terminal,seq \
    --load1 ./model/sm/smv2/sminit.bin \
    --load2 ./model/oppo_pool/smv2/oppoinit.bin \
    --gpu 0 \
    --T 20 \
    --shuffle_player \
    --keys_in_reply V
