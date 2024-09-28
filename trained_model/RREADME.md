| File                                                    | Description                                                  |                                                              |
| ------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Neural_ODE_ite400_loss0.2.pt                            | Train Neural_DOE with 400 iterations, loss - 0.2             |                                                              |
| Neural_ODE_ite0400_loss0.301785.pt                      | Train Neural_DOE with 400 iterations, loss -0.301785         |                                                              |
| Neural_ODE_ite1000_loss0.008033.pt                      | Train Neural_DOE with 1000 iterations, loss -0.008033        |                                                              |
| Neural_ODE_ite1000_loss0.028490.pt                      | Train Neural_DOE with 1000 iterations, loss -0.028490        |                                                              |
|                                                         |                                                              |                                                              |
|                                                         |                                                              |                                                              |
|                                                         |                                                              |                                                              |
| FeedbackNN_ite180_loss0.497830.pt；iteration_ite180.txt | Train feedback part with Neural_ODE_ite400_loss0.2           | ![d_FeedbackNN0902](D:\Code_workspace\GeneralNN\spiral_test\pythonProject1\png\d_FeedbackNN0902.png) |
| FeedbackNN_ite511_loss0.495402.pt；iteration_ite511.txt | Train feedback part with Neural_ODE_ite400_loss0.2           | ![d_FeedbackNN0903_01](D:\Code_workspace\GeneralNN\spiral_test\pythonProject1\png\d_FeedbackNN0903_01.png) |
| FeedbackNN_ite1074_loss0.483191.pt                      | Train feedback part with Neural_ODE_ite0400_loss0.301785     | ![d_FeedbackNN0903_03_400_0.3](D:\Code_workspace\GeneralNN\spiral_test\pythonProject1\png\d_FeedbackNN0903_03_400_0.3.png) |
| FeedbackNN_ite492_loss0.496539_conver.pt                | Train feedback part with Neural_ODE_ite400_loss0.2； with data modification 0.2 | ![d_FeedbackNN0903_02_exp](D:\Code_workspace\GeneralNN\spiral_test\pythonProject1\png\d_FeedbackNN0903_02_exp.png) |
| FeedbackNN_ite477_loss0.494530_conver.pt                | Train feedback part with Neural_ODE_ite400_loss0.2； with data modification 0.5；no mini-batch; lager NN | ![d_FeedbackNN0903_exp](D:\Code_workspace\GeneralNN\spiral_test\pythonProject1\final_png\d_FeedbackNN0903_exp.png) |
|                                                         |                                                              |                                                              |
|                                                         |                                                              |                                                              |
|                                                         |                                                              |                                                              |
|                                                         |                                                              |                                                              |
|                                                         |                                                              |                                                              |
|                                                         |                                                              |                                                              |
|                                                         |                                                              |                                                              |



### Conclusion: 

1. More iterations  on Neural_ODE do not necessarily result in smaller training loss;
2. Less training loss on Neural_ODE.pt would not improve the training performance of FeedbackNN;