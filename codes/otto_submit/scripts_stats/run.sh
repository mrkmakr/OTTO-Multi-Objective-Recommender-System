cd `dirname $0`
path=../../otto/scripts_stats/
python ${path}aid_feats.py 1
python ${path}aid_feats.py 2
python ${path}aid_feats.py 3
python ${path}h_aid_feats.py
python ${path}h_aid_feats_v2.py
python ${path}session_feats.py
python ${path}session2h.py
