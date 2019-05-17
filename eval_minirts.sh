MODEL=$1
NUM_EVAL=20000

game=./rts/game_MC/game model_file=./rts/game_MC/oppo_pool model=actor_critic python3 eval.py --num_games 128 --batchsize 128 --tqdm --load $1 --gpu 0 --players "fs=50,type=AI_NN;fs=50,type=AI_SIMPLE" --eval_stats winrate --num_eval $NUM_EVAL --additional_labels id,last_terminal,seq --shuffle_player --num_frames_in_state 1 --greedy --keys_in_reply V #--omit_keys Wt,Wt2,Wt3 # --save_replay_prefix ./replay/smv1replay
