

# Feedback Favors the Generalization of Neural ODEs

This project simulates the developed feedback neural networks on a spiral example.  

### 1. Files introduction

| Files                        | Introduction                                                 | Details                                                      |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Folder: **past_vision**      | Store intermediate versions of .py files.                    | ---                                                          |
| Folder: **png**              | The auto-storage directory in which the program runs. Store intermediate test results. | ---                                                          |
| Folder: **final_png**        | Store final test results, preparing for the paper.           | ---                                                          |
| Folder: **trained_model**    | Trained NN models.                                           | **Neural_ODE.pt** - modelof Neural_ODE; **FeedbackNN.pt** - model of feedback part；**iteration.txt** - evaluation of loss as trianing the feedback part; |
| **a_one_step_pre.py**        | One-step prediction program, to test the feedback mechanism. | ![a_one_step_pre0902](final_png\a_one_step_pre0902.png)      |
| **a_multi_steps_pre.py**     | Multi-steps prediction program.                              | ![a_multi_steps_pre0828](final_png\a_multi_steps_pre0828.png) |
| **b_ablation_L_data.py**     | Collect data with different degrees of uncertainty and different L levels. The different L needs to be set artificially. | ---                                                          |
| **b_ablation_L_heatmap.py**  | The plot program of b_ablation_L_data.py.                    | <img src="final_png\b_ablation_L_heatmap_sub0902.png" alt="b_ablation_L_heatmap_sub0902" style="zoom: 25%;" /><img src="D:\Code_workspace\GeneralNN\spiral_test\pythonProject1\final_png\b_ablation_L_heatmap_full0902.png" alt="b_ablation_L_heatmap_full0902" style="zoom:25%;" /> |
| **c_neural_ODE_nominal.py**  | Train Neural ODE on the nominal task and store the trained model (Neural_ODE.pt). | <img src="final_png\c_neural_ODE_nominal0902.png" alt="c_neural_ODE_nominal0902" style="zoom:50%;" /> |
| **c_neural_ODE_domran.py**   | Train Neural ODE through domain randomization and plot the degraded performance on the nominal task. | <img src="final_png\c_neural_ODE_domran0902.png" alt="c_neural_ODE_domran0902" style="zoom:50%;" /> |
| **d_FeedbackNN.py**          | Train feedback neurons through domain randomization, store the trained model (FeedbackNN.pt), and plot the mataining performance on the nominal task. | ![d_FeedbackNN0902](final_png\d_FeedbackNN0902.png)          |
| **d_FeedbackNN_converge.py** | The convergence procedure is revealed in training dataset    | ![d_FeedbackNN0903_ite477_conver](final_png\d_FeedbackNN0903_ite477_conver.png) |
| **d_FeedbackNN_test.py**     | Test trained model  FeedbackNN.pt on randomized tasks, to show its generalization. | ![d_FeedbackNN_test0902](final_png\d_FeedbackNN_test0902.png) |

