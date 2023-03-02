cd `dirname $0`
echo "------ calculate covisitation"
sh scripts_covisitation/run.sh
echo "------ calculate aid and session stats"
sh scripts_stats/run.sh
echo "------ train nn"
sh scripts_nn/run.sh
echo "------ generate candidates and create data for reranker"
sh scripts_prepare_2nd/run.sh
echo "------ train reranker"
sh scripts_ranker/run.sh