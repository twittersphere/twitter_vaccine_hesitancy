# Exploring vaccine hesitancy in digital public discourse: From  tribal polarization to socio-economic disparities

Official Repository of the paper.

## Installation

Conda environment creation:
```bash
conda create env -n vaccine_hesitancy --python=3.9 -y
```

Activate conda environment
```bash
conda activate vaccine_hesitancy
```

Install [Snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)

CTM Installation:
```bash
git clone https://github.com/HuzeyfeAyaz/contextualized-topic-models.git
cd contextualized-topic-models
git checkout h5py_support
python setup.py install
```

Install all other dependencies
```bash
pip install -r requirements.txt
```

## Usage

```bash
snakemake --cores=8
```

> Note: This is the refactored version of the main code and not fully tested. Please use cautiously.

- If you used any code or want to refer the paper, please cite:
```
Ayaz H, Celik MH, Koytak HZ, Yanik IE (2024) Exploring vaccine hesitancy in digital public discourse: From tribal polarization to socio-economic disparities. PLOS ONE 19(11): e0308122. https://doi.org/10.1371/journal.pone.0308122
```
- [BibTex Link](https://journals.plos.org/plosone/article/citation/bibtex?id=10.1371/journal.pone.0308122)
