#How to run the files
We have used two models to predict the species of the leaf.

So, we have two trained models in two folders i.e. custom_model and renet. These folders contains the training script that has the implementation and the saved weights and graphs (.h5). We have two formats i.e python(.py) and python notebook(.ipynb).

We've also included the final_prediction scprit, which can be used for the final evaluation. In this script, only the path variables, i.e. test_path, path_res_model and path_custom_model have to be changed. 
test_path referes to the test dataset, 
path_res_model refers to the path of the reg_model.h5 in the resnet folder, 
and path_custom_model refers to the path of the leaf_reco.h5 in the custom_model. 

And the final line prints the final accuracy that our algorithm can achieve on the private test set.
