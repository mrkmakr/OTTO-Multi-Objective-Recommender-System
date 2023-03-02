Hello!

Below you can find a outline of how to reproduce my solution for the OTTO – Multi-Objective Recommender System
 competition.  

#HARDWARE: (The following specs were used to create the original solution).  
Ubuntu 20.04 LTS (1500 GB boot disk). 
n1-standard-64 (64 vCPUs, 240 GB memory). 
4 x NVIDIA Tesla V100. 

#DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed). 
sh prepare_data.sh. 

#MODEL BUILD: There are two options to produce the solution.  
1) simple model. 
    a) expect this to run for 2 days. 
    b) reproduce the simple model result. 
2) retrain models. 
    a) expect this to run about 2 weeks. 
    b) follow this to produce entire solution from scratch. 

shell command to run each build is below
#1) simple model
sh run_all_simple.sh

#2) retrain models
sh run_all.sh

#MEMO
Use code from `otto/` to process data for training. Intermediate processing results are stored in `otto/inputs/`.
Data processing for inference is performed using the code `otto/` from `otto_submit/`. The intermediate processing result at this time is stored in `otto_submit/inputs/`.
To check whether the entire flow can be executed, execute `reduce_data_for_run_test.py` to reduce the number of data and then execute `run_all.sh` or `run_all_simple.sh`.
`otto/scripts_nn` contains codes for training NN. All of them are minor changes, so I commented only the most important code `emb_model_v42.py`.

