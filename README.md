# Text Generation for Chinese Online Fantasy Novels.

*Author: Horace He*

This work collect 12 representative chinese online fantasy novels from 3 famous authors, and form a medium scale dataset of 768K sentences paired with book title and author information.
	
This work apply the RNNLM and VAE models on the proposed chinese online fantasy dataset. In order to alleviate the KL vanishing problem, various approaches have been studied to aid VAE optimization.
	
This work evaluate the performance of unconditional fantasy generation, and analyze the learned VAE latent space with t-SNE visualization.

## Prerequests

### Dependencies

* anaconda==4.8.3
* torch==1.5.0
* torchtext==0.6.0
* tensorflow==2.1.0
* tensorboard==2.1.1
* HanLP==2.0.0a44
* transfomer==2.10.0

### Add project directory to PYTHONPATH

```bash
export PYTHONPATH=/path/to/project/:$PYTHONPATH
```

## Data Preparation

### Data Cleaning

The program will download and clean the text files autormatically.

```bash
python prepare_data.py
```

### Dataset building

Build torchtext dataset after preparation.

```bash
python dataset.py
```

## Train Model


### Train Language Model

```bash
python src/train/train_lm.py
```

Run with "-h" option to see argument details.

### Train VAE

```bash
python src/train/train_vae.py
```

Run with "-h" option to see argument details.

## Visualization

### Training curve

The training process can be visualized by running tensorboard in the results directory.

```bash
tensorboard --logdir .
```

And then open the port 6006 on browser.

### Latent space visualization

Run the following command to write embeddings into results directory, then restart the tensorboard to see visualizations.

```bash
python src/visualize/latent_space_visualize.py
```

Run with "-h" option to see argument details.