cd `dirname $0`
path=../../otto/scripts_nn/
python ${path}emb_model_v15.py --pretrain at_test --is_inference_all > v15.log
python ${path}emb_model_v18.py --pretrain at_test --is_inference_all > v18.log
python ${path}emb_model_v21.py --pretrain at_test --is_inference_all > v21.log
python ${path}emb_model_v23.py --pretrain at_test --is_inference_all > v23.log
python ${path}emb_model_v27.py --pretrain at_test --is_inference_all > v27.log
python ${path}emb_model_v29.py --pretrain at_test --is_inference_all > v29.log
python ${path}emb_model_v31.py --pretrain at_test --is_inference_all > v31.log
python ${path}emb_model_v42.py --pretrain at_test --is_inference_all > v42.log