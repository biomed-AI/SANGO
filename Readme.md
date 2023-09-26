![](figures/model.png)

# SANGO

The official implementation for "SANGO".

**Table of Contents**

* [Datasets](#Datasets)
* [Installation](#Installation)
* [Usage](#Usage)
* [Tutorial](#Tutorial)
* [Citation](#Citation)

## Datasets

To Do.

## Installation

To reproduce SANGO, we suggest first create a conda environment by:

~~~shell
conda create -n SANGO python=3.8
conda activate SANGO
~~~

and then run the following code to install the required package:

~~~shell
pip install -r requirements.txt
~~~

and then install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) according to the CUDA version, take torch-1.13.1+cu117 as an example:

~~~shell
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
~~~

## Usage

~~~shell
# data preprocessing
# As needed

# Stage 1: embeddings extraction
cd SANGO/CACNN

python main.py -i reference_query_example.h5ad \ # input data
               -g mm9 \ # genome
               -o ../output/reference_query_example \ # output path

# Stage 2: cell type prediction
cd ../GraphTransformer

python main.py --data_dir ../output/reference_query_example/CACNN_output.h5ad \ # input data
               --train_name_list reference --test_name query \
               --save_path ../output \
               --save_name reference_query_example \
~~~

## Tutorial

1. Install the required environment according to [Installation](#Installation)

2. Create a folder `data` and download the datasets according to [Datasets](#Datasets)

3. For more detailed information, run the tutorial [reference_query_example.ipynb](reference_query_example.ipynb) for how to do data preprocessing and training 

## Citation

If you find our codes useful, please consider citing our work:

~~~bibtex

~~~