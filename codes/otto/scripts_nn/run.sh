cd `dirname $0`
python emb_model_v15.py --is_inference_all > v15.log
python emb_model_v18.py --is_inference_all > v48.log
python emb_model_v21.py --is_inference_all > v21.log
python emb_model_v23.py --is_inference_all > v23.log
python emb_model_v27.py --is_inference_all > v27.log
python emb_model_v29.py --is_inference_all > v29.log
python emb_model_v31.py --is_inference_all > v31.log
python emb_model_v42.py --is_inference_all > v42.log

# If training is successful but memory error occurs in inference, you can only infer with the following command
# python emb_model_v15.py --is_inference_all --is_skip_train --pretrain at_train