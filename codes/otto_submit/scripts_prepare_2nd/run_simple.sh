cd `dirname $0`
path=../../otto/scripts_prepare_2nd/
python ${path}prepare_cands.py 0
nohup python ${path}prepare_cands.py 14_p_chain_w_20_2 &
python ${path}prepare_cands.py 14_p_chain_all_20_2
nohup python ${path}prepare_cands.py 15_p_chain_all_20_2 &
python ${path}prepare_cands.py 16_p_chain_all_100_2
nohup python ${path}prepare_cands.py 19_p_chain_w_20_2 &
python ${path}prepare_cands.py emb_v42
`
nohup python -u ${path}prepare_feature.py --data_idx 0 --data_idx_end 6 --simple &
nohup python -u ${path}prepare_feature.py --data_idx 6 --data_idx_end 12 --simple &
python -u ${path}prepare_feature.py --data_idx 12 --data_idx_end 18 --simple
`