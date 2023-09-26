## SANGO

The official implementation for "SANGO".

**Table of Contents**

* [Datasets](#Datasets)
* [Installation](#Installation)
* [Usage](#Usage)
* [Tutorial](#Tutorial)

### Datasets

To Do.

### Installation

To reproduce SANGO, we suggest first create a conda environment by:

~~~shell
conda create -n SANGO python=3.8
conda activate SANGO
~~~

and then run the following code to install the required package:

~~~shell
pip install -r requirements.txt
~~~

### Usage

~~~shell
# data preprocessing
# As needed

# Stage 1: embeddings extraction
cd CACNN

python main.py -i ../preprocessed_data/BoneMarrowB_liver.h5ad \
               -z 64 \
               -g mm9 \
               -o ../output/BoneMarrowB_liver \
               --max_epoch 300 \
               --device 0

# Stage 2: cell type prediction
cd ../GraphTransformer

python main.py --use_bn \
               --use_residual \
               --use_gumbel \
               --data_dir ../output/BoneMarrowB_liver/CACNN_output.h5ad \
               --train_name_list BoneMarrow_62216 --test_name Liver_62016 \
               --save_path ../output \
               --save_name BoneMarrowB_liver \
               --device 0
~~~

### Tutorial

* BoneMarrowB_Liver.ipynb
* MosP1_Cerebellum.ipynb

