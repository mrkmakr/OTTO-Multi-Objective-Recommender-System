cd `dirname $0`
path=../../otto/scripts_nn/
python ${path}emb_model_v42.py --pretrain at_test --is_inference_all > v42.log