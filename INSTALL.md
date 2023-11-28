## Installation

First, clone the repository locally:

```bash
git clone https://github.com/mkimhi/RISE.git
cd RISE
```

Install dependencies and pycocotools:

```bash
pip install -r requirements.txt
pip install -e .
pip install shapely==1.7.1
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```

Compiling Deformable DETR CUDA operators:

```bash
cd projects/IDOL/idol/models/ops/
sh make.sh
```


## Data Preparation

Go to [ArmBench](http://armbench.s3-website-us-east-1.amazonaws.com/) and requst acess to the data.
Link it to 'datasets' directory:

```bash
cd datasets/
ln -s /path_to_armbench
```



we expect the directory structure to be the following:

```
RISE
├── datasets
│   ├──armbench
│   ├──OCID
│   ├──OSD 
...
armbench
├── images
├── annotations
├── train.json
├── 1_100_train.json
├── ...
├── test.json

```


