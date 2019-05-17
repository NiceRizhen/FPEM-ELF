MODEL=$1 \

game=./rts/game_MC/game \
model_file=./rts/game_MC/oppo_pool \
model=actor_critic \
python3 oppo_pool.py  \
    --num_games 1024 \
    --batchsize 128 \
    --tqdm --players "fs=50,type=AI_NN;fs=50,type=AI_NN" \
    --trainer_stats winrate \
    --additional_labels id,last_terminal,seq \
    --load ./oppo_pool/oppo_pool-9.bin \
    --gpu 0 \
    --T 20 \
    --shuffle_player \
    --keys_in_reply V \
    --save_dir ./oppo_pool \
    --save_prefix oppo_pool
