cd `dirname $0`
path=../../otto/scripts_prepare_2nd/
nohup python ${path}prepare_cands.py 0 &
python ${path}prepare_cands.py 11_p_chain_full_20_5
nohup python ${path}prepare_cands.py 11_p_chain_full_100_5 &
python ${path}prepare_cands.py 11_p_chain_all_20_2
nohup python ${path}prepare_cands.py 12_p_chain_all_20_2 &
python ${path}prepare_cands.py 13_p_chain_all_20_2
nohup python ${path}prepare_cands.py 14_p_chain_all_20_2 &
python ${path}prepare_cands.py 15_p_chain_all_20_2
nohup python ${path}prepare_cands.py 16_p_chain_all_20_2 &
python ${path}prepare_cands.py 17_p_chain_all_20_2
nohup python ${path}prepare_cands.py 11_p_chain_all_100_2 &
python ${path}prepare_cands.py 12_p_chain_all_100_2
nohup python ${path}prepare_cands.py 13_p_chain_all_100_2 &
python ${path}prepare_cands.py 14_p_chain_all_100_2
nohup python ${path}prepare_cands.py 15_p_chain_all_100_2 &
python ${path}prepare_cands.py 16_p_chain_all_100_2
nohup python ${path}prepare_cands.py 17_p_chain_all_100_2 &
python ${path}prepare_cands.py 11_14_chain_1_2dic
nohup python ${path}prepare_cands.py 11_14_chain_2_2dic &
python ${path}prepare_cands.py 11_14_chain_3_2dic
nohup python ${path}prepare_cands.py 11_p_chain_w_20_2 &
python ${path}prepare_cands.py 11_p_chain_w_100_2
nohup python ${path}prepare_cands.py 15_14_chain_1_2dic &
python ${path}prepare_cands.py 15_14_chain_2_2dic
nohup python ${path}prepare_cands.py 15_14_chain_3_2dic &
python ${path}prepare_cands.py 15_13_chain_1_2dic
nohup python ${path}prepare_cands.py 15_13_chain_2_2dic &
python ${path}prepare_cands.py 15_13_chain_3_2dic
nohup python ${path}prepare_cands.py 11_13_chain_1_2dic &
python ${path}prepare_cands.py 11_13_chain_2_2dic
nohup python ${path}prepare_cands.py 11_13_chain_3_2dic &
python ${path}prepare_cands.py 14_p_chain_w_20_2
nohup python ${path}prepare_cands.py 14_p_chain_w_100_2 &
python ${path}prepare_cands.py 18_p_chain_w_20_2
nohup python ${path}prepare_cands.py 18_p_chain_w_100_2 &
python ${path}prepare_cands.py 15_18_chain_1_2dic
nohup python ${path}prepare_cands.py 15_18_chain_2_2dic &
python ${path}prepare_cands.py 15_18_chain_3_2dic
nohup python ${path}prepare_cands.py 11_18_chain_1_2dic &
python ${path}prepare_cands.py 11_18_chain_2_2dic
nohup python ${path}prepare_cands.py 11_18_chain_3_2dic &
python ${path}prepare_cands.py 19_p_chain_w_20_2
nohup python ${path}prepare_cands.py 20_p_chain_w_20_2 &
python ${path}prepare_cands.py emb_v15
nohup python ${path}prepare_cands.py emb_v18 &
python ${path}prepare_cands.py emb_v21
nohup python ${path}prepare_cands.py emb_v23 &
python ${path}prepare_cands.py emb_v27
nohup python ${path}prepare_cands.py emb_v29 &
python ${path}prepare_cands.py emb_v31
python ${path}prepare_cands.py emb_v42

nohup python -u ${path}prepare_feature.py --data_idx 0 --data_idx_end 6 --with_additinal_feats &
nohup python -u ${path}prepare_feature.py --data_idx 6 --data_idx_end 12 --with_additinal_feats &
python -u ${path}prepare_feature.py --data_idx 12 --data_idx_end 18 --with_additinal_feats
