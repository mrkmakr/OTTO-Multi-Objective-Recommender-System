cd `dirname $0`
echo "------ calculate covisitation"
sh scripts_covisitation/run_simple.sh
echo "------ train nn"
sh scripts_nn/run_simple.sh
echo "------ generate candidates and create data for reranker"
sh scripts_prepare_2nd/run_simple.sh
echo "------ train reranker"
sh scripts_ranker/run_simple.sh