mkdir -p results/gridsearch

LOG_FILE="gridsearch.log"
echo "Starting grid search at $(date)" > $LOG_FILE

run_gridsearch() {
    local config=$1
    echo "Running grid search for $config at $(date)" | tee -a $LOG_FILE
    python src/gridsearch.py --config $config 2>&1 | tee -a $LOG_FILE
    echo "Completed grid search for $config at $(date)" | tee -a $LOG_FILE
    echo "----------------------------------------" | tee -a $LOG_FILE
}

# HOG 
run_gridsearch "configs/gridsearch_svc_hog_config.yaml"
run_gridsearch "configs/gridsearch_knn_hog_config.yaml"
run_gridsearch "configs/gridsearch_rf_hog_config.yaml"

# LBP 
run_gridsearch "configs/gridsearch_svc_lbp_config.yaml"
run_gridsearch "configs/gridsearch_knn_lbp_config.yaml"
run_gridsearch "configs/gridsearch_rf_lbp_config.yaml"

# GLCM
run_gridsearch "configs/gridsearch_svc_glcm_config.yaml"
run_gridsearch "configs/gridsearch_knn_glcm_config.yaml"
run_gridsearch "configs/gridsearch_rf_glcm_config.yaml"

# Gabor 
run_gridsearch "configs/gridsearch_svc_gabor_config.yaml"
run_gridsearch "configs/gridsearch_knn_gabor_config.yaml"
run_gridsearch "configs/gridsearch_rf_gabor_config.yaml"

# SIFT 
run_gridsearch "configs/gridsearch_svc_sift_config.yaml"
run_gridsearch "configs/gridsearch_knn_sift_config.yaml"
run_gridsearch "configs/gridsearch_rf_sift_config.yaml"

echo "All grid searches completed at $(date)" | tee -a $LOG_FILE 