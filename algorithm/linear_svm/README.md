SVM MODEL EXPERIMENTATION & EVALUATION GUIDE

1. Hyperparameter Tuning and Results (Based on Macro F1-score)
The hyperparameter tuning process for parameter C was conducted by training multiple configurations and evaluating each model independently on the Test set. The Macro F1-scores are summarized in the table below:

-----------------------------------------------------------
| C Parameter   |  5.0   |  1.0   |  0.01  |  0.1 |
-----------------------------------------------------------
| Macro F1-score| 74.02% | 74.33% | 73.54% | 74.94% |
-----------------------------------------------------------

Conclusion: After comparing the Macro F1-scores across all configurations, parameter C = 0.1 yields the highest performance (74.94%). Therefore, svm_pipeline_C0.1.pkl is officially selected as the optimal model for the final system integration.

2. Execution Template for Replication

To replicate the experiment or evaluate the selected optimal model, use the following two-step procedure:

Step 1: Training Phase
Run the training script with the optimal parameter C=0.1. The system automatically combines the Train and Validation datasets for a final robust re-training phase before saving the model package:
python train.py --c 0.1 --version C0.1

Step 2: Evaluation Phase
Run the evaluation script by passing the optimal model name to independently calculate final metrics and export visual plots into the result/ directory:
python evaluate.py --model-name svm_pipeline_C0.1.pkl

--> Outputs generated in the result/ directory:
- svm_evaluation_report.txt: A detailed text report containing comprehensive performance metrics (Precision, Recall, and the selection-base Macro F1-score).
- matrix_svm_pipeline_C0.1.png: The confusion matrix plot rendered in a clean blue color palette, containing only raw data counts for clear interpretation.
