READ ME! :)

Repository for my second year "Waves" lab cycle. Included files:

- waves_lab_script.pdf
-> document provided which details our experiment, split into three tasks
- fourier.py
-> a very basic script containing a function to calculate N terms of a Fourier series for a square wave and plot them. Task 1.2 in the lab script specifies a square wave with period τ = 240 and amplitude T(t) 0 to 100. The fourier series for this was solved analytically then transcribed into python.
- task_1.3_semicircle_low.txt and task_1.3_semicircle_high.txt
-> two data files for plotting a semicircle, one with low resolution (0.1 au) and one with high res (0.01 au) for the purpose of the following script
- num_integration.py
-> again, very basic script for task 1.3 to perform numerical integration under a curve (specifically the data sets above). This is done by plotting a series of rectangles under each curve and summing their areas. These estimations are then compared to the actual area.

########################################################

The following libraries are required for the script to function, 
all of which are included in the conda environment:\
numpy\
matplotlib\
seaborn
