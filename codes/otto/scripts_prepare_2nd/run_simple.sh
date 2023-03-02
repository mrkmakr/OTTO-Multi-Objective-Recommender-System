cd `dirname $0`
python prepare_cands.py 0
nohup python prepare_cands.py 14_p_chain_w_20_2 &
python prepare_cands.py 14_p_chain_all_20_2
nohup python prepare_cands.py 15_p_chain_all_20_2 &
python prepare_cands.py 16_p_chain_all_100_2
nohup python prepare_cands.py 19_p_chain_w_20_2 &
python prepare_cands.py emb_v42

nohup python -u prepare_feature.py --data_idx 0 --data_idx_end 6 --simple --with_label &
nohup python -u prepare_feature.py --data_idx 6 --data_idx_end 12 --simple --with_label &
python -u prepare_feature.py --data_idx 12 --data_idx_end 20 --simple --with_label
