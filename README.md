## Machine learning class project: processing experimental figures using convolutional neural networks 

Done in collaboration with Zhaoyu Bai and Yaozhang Zhou.

## Goal: automate the plotting of Streda lines (associated with quantum Hall effect) from dirty experimental figures.

A typical experimental figure (original input) and its Streda plot (grund truth) look like (taken from https://www.nature.com/articles/s41586-021-04002-3)

![Alt text](/orginal_vs_ground_truth.png?raw=true "orginal_vs_ground_truth") 

and we aim to automate such plottings.

## Methodology
The main "dirt" of the experimental figures is the line-broadening side effect of the experimental technology,  so our main task is to thin the broadened lines/stripes.

Due to the scarcity of actual experimental figures,  we generate synthetic data of random straight lines (ground truth) and their broadened version (input)   

A synthetic input and its ground truth look like

![Alt text](/synthetic_input_vs_ground_truth.png?raw=true "synthetic_orginal_vs_ground_truth") 

To generate these synthetic data, run "run_script.m" in the synthetic_data_gen_matlab/ folder.  We should twenty such examples in thick_lines_synthetic/ and thin_lines_synthetic/ folders.

We use the UNET architecture (a variant of CNN) to achieve the line thinning task. Weight file is "unet_model.py", and we use "train.py" to train the model.  

## Result

After the model is trained,  we run "plot_and_save.py" to juxtapose and compare the inferenced output and the actual ground truth: 

![Alt text](/three_images_comparison_unfiltered.png?raw=true "synthetic_orginal_vs_ground_truth") 

we can sharpen the visuals by filtering the inferenced imaged to black and white: 

![Alt text](/three_images_comparison.png?raw=true "synthetic_orginal_vs_ground_truth") 

## Discussion
The model works well in the region where the lines are not too densely distributed,  and the color is not too light.  Otherwise the result is not accurate.  The main bottleneck for our project is the scarcity of actual experimental data and their ground truths,  and the crudeness of synthetic data.
