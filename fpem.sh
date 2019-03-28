MODEL=$1

game=./rts/game_MC/game model_file=./rts/game_MC/fpem_model model=actor_critic python3 train_fpem.py  --num_games 1024 --batchsize 128 --tqdm --players "fs=50,type=AI_NN;fs=50,type=AI_NN" --trainer_stats winrate --additional_labels id,last_terminal,seq --load ./fpem/fpem.bin --gpu 0 --T 20 --shuffle_player --keys_in_reply V --save_dir ./fpem --save_prefix fpem