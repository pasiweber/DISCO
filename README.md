# DISCO: DISCO: Internal Evaluation of Density-Based Clustering with Noise Labels

This repository is the official implementation of DISCO: Internal Evaluation of Density-Based Clustering with Noise Labels, submitted to [IEEE ICDM 2025](https://www3.cs.stonybrook.edu/~icdm2025/index.html).




## Motivational Figure

![Motivation](imgs/Motivation.png)

## Structure of the repository

```bash
.
├── clusterings_                            # saved kmeans clustering
│
├── data                                    # dataset infos
│
├── datasets  
│   ├── DENSIRED                            # data generator
│   ├── synth                               # synthetic data
│   └── ...                                 # file to provide access to data
│
├── imgs                                    # image files (plots, figures)
│   └── ...         
│              
├── src
│   ├── Clusterer                           # implementations for clustering methods
│   ├── Evaluation                          # implementations of CVIs
│   ├── Experiments                         # experiment scripts
│   │   ├── DatasetsJupyterNotebooks                    # datasets
│   │   ├── JupyterNotebooks_Analysis                   # notebooks to analyse datasets
│   │   ├── JupyterNotebooks_SyntheticExperiments       # notebooks to generate experiment results
│   │   ├── scripts                                     # additional experiments
│   │   └── ...
│   ├── utils                               # colors, metrics, utility functions 
│   ├── __init__.py                         # init file
│   └── __setup.ipynb                       # notebook for setup
│ 
├── .gitignore                              # ignore files should not commit to Git
└── README.md                               # project description  
```


## Experimental Setup
| Method  | Hyperparameter                  | Value | 
|---------|---------------------------------|-------|
| CDBW    | number of representative points | 10    | 
| CVDD    | number of neighborhoods         | 7     |
| CVNN    | number of nearest neighbors     | 10    | 
| DCSI    | corepoints                      | 5     | 
|---------|---------------------------------|-------|
| DISCO   | min pts                         | 5     |
|---------|---------------------------------|-------|