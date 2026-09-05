import os

from data_processor import load_and_preprocess_data
from train_models import train_and_evaluate_models
from visualizer import generate_performance_visuals

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE_PATH = os.path.join(PROJECT_DIR, 'train.parquet.parquet')

# 1. Load data
(f_train, f_test, l_train, l_test), v_tool = load_and_preprocess_data(DATASET_FILE_PATH)

# 2. Train and get REAL metrics
lr_mod, dt_mod, real_lr_acc, real_dt_acc, real_lr_f1, real_dt_f1 = train_and_evaluate_models(f_train, f_test, l_train, l_test)

# 3. Pass REAL metrics to visuals
generate_performance_visuals(
    trained_decision_tree=dt_mod, 
    vectorizer_tool=v_tool, 
    logistic_accuracy=real_lr_acc,      
    decision_tree_accuracy=real_dt_acc, 
    logistic_f1_score=real_lr_f1,       
    decision_tree_f1_score=real_dt_f1    
)

print(f"Finished! Real LR Accuracy was: {real_lr_acc:.2f}")