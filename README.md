# Image Inpainting Project
This project uses a simple CNN trained to predict a full image from an image that is obscured with a black grid.


### Example usage
To train a new model run
```
python main.py working_config.json
```
To run the Streamlit application run
```
streamlit run streamlit_application.py
```
### Structure
The folder structure of the project
```
Image-Inpainting-Project
|- architectures.py
|    Classes and functions for network architectures
|- datasets.py
|    Dataset classes and dataset helper functions
|- main.py
|    Main file. In this case also includes training and evaluation routines.
|- README.md
|    A readme file containing info on project, example usage and dependencies.
|- utils.py
|    Plotting functions for the training procedure.
|- working_config.json
|     Configuration file for specifying the network architecture and training parameters. Can also be done via command line arguments to main.py.
|- streamlit_application.py
|     Streamlit script. Used for project (inference) demonstration.
```

### Dependencies
To install the project environmen, run
```
conda env create -f environemnt.yml
```
