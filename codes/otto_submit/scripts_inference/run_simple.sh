cd `dirname $0`

run_ids=v1_simple
echo ${run_ids}
mkdir logs
for i in `seq 0 3`
do
echo ${i}
nohup python -u inference_ensemble.py ${run_ids} clicks ${i} > logs/clicks${i}.log &
sleep 5
nohup python -u inference_ensemble.py ${run_ids} carts ${i} > logs/carts${i}.log &
sleep 5
python -u inference_ensemble.py ${run_ids} orders ${i} > logs/orders${i}.log
sleep 5
done
sleep 60
python make_sub.py ${run_ids}
