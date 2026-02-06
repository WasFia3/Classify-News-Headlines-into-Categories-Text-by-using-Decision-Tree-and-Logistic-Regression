import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np

def generate_performance_visuals(trained_decision_tree, vectorizer_tool, 
                                 logistic_accuracy, decision_tree_accuracy, 
                                 logistic_f1_score, decision_tree_f1_score):
    
    # --- 1. Decision Tree Structure Visualization ---
    plt.figure(figsize=(20, 10), dpi=150) 
    plot_tree(trained_decision_tree, 
              max_depth=3,  # Showing top levels for report readability
              feature_names=list(vectorizer_tool.get_feature_names_out()), 
              filled=True, 
              rounded=True, 
              fontsize=10)
    plt.title("Decision Tree Visualization (Root and Top Nodes)")
    plt.savefig('decision_tree_graph.png', bbox_inches='tight')
    plt.close() 

    # --- 2. Model Performance Comparison Bar Chart ---
    model_labels = ['Logistic Regression', 'Decision Tree']
    accuracy_values = [logistic_accuracy, decision_tree_accuracy]
    f1_metric_values = [logistic_f1_score, decision_tree_f1_score]

    x_positions = np.arange(len(model_labels))
    group_width = 0.35 

    fig, axes = plt.subplots(figsize=(10, 6))
    acc_bars = axes.bar(x_positions - group_width/2, accuracy_values, group_width, label='Accuracy', color='skyblue')
    f1_bars = axes.bar(x_positions + group_width/2, f1_metric_values, group_width, label='F1-Score', color='salmon')

    # Formatting the chart
    axes.set_ylabel('Scores (0.0 - 1.0)')
    axes.set_title('Comparative Performance Analysis: Logistic vs Decision Tree')
    axes.set_xticks(x_positions)
    axes.set_xticklabels(model_labels)
    axes.set_ylim(0, 1.1) 
    axes.legend()

    # Helper function to display values on top of bars
    def attach_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            axes.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    attach_value_labels(acc_bars)
    attach_value_labels(f1_bars)

    plt.tight_layout()
    plt.savefig('performance_comparison_chart.png')
    plt.close()
    print("Picttures saved succesfully in the floder.")