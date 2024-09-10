## Machine learning class project: processing experimental figures using convolutional neural networks 

Done in collaboration with Zhaoyu Bai and Yaozhang Zhou.

Goal: automate the plotting of Streda lines (associated with quantum Hall effect) from dirty experimental figures.

A typical experimental figure looks like (taken from https://www.nature.com/articles/s41586-021-04002-3)

![Alt text](/nature_imag_cropped.png?raw=true "original") 

 and its post-propcessed plot (ground truth) looks like 

![Alt text](/hand_drawn_cropped.png?raw=true "ground truth") 

and we aim to automate such plottings.

The main "dirt" of the experimental figures is in fact the line-brodening effect of the experimental technology,  so our main task is to thin the broadened lines/stripes.

Due to the scarcity of actual experimental figures,  we generate synthetic data of random straight lines (ground truth) and their broadened version (input) using MATLAB ("run_script.m" in /synthetic_data_gen_matlab/).  

A synthetic input looks like

![Alt text](/thick_lines_synthetic/run_001.png?raw=true "synthetic input") 
