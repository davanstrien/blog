---
layout: post
title: Selected projects
permalink: /projects/
toc: true
---

This page collects selected projects that I have worked on.

## Machine learning projects

### flyswot: using computer vision to detect 'fake flysheets'
[**Related Posts**](https://danielvanstrien.xyz/categories/#flyswot)

An increasing challenge for libraries is managing the scale of digitised material resulting from digitisation projects and 'born digital' materials. This project aims to detect mislabelled digitised manuscript pages. 

<figure>
    <img src="https://api.bl.uk/image/iiif/ark:/81055/vdc_100056093448.0x000179/full/400,/0/default.jpg"/>
    <figcaption>A manuscript page with the correct label "fse.ivr"</figcaption>
</figure>

As a result of the limitations of a previous system for hosting digitised manuscript images, many images have incorrect page metadata associated with the image. An image has been correctly labelled as an 'end flysheet' in the example above. This label is represented by the `fse` label, which is included in the filename for the image. However other types of manuscript pages also have this label incorrectly assigned, i.e. a 'cover' has `fse` in the filename. There is around a petabyte of images to review before ingesting a new library system. This project uses computer vision to support library staff in processing this collection. At the moment, this project does the following:

- pulls in an updated dataset of training examples
- trains a model on these images
- the model architecture has multiple heads to allow the model to make both a 'crude' prediction for whether the image is incorrectly labelled and a 'full' prediction for the true label.
- once a new version of the model has been trained, it is pushed to the 🤗 model hub.
- the end-user uses the model through a command-line tool which is pointed at a directory of images to be checked.

The code for the command-line tool is available here: [github.com/davanstrien/flyswot/]()

**Some of the tools used:** [fastai](https://docs.fast.ai/), [DVC](https://dvc.org/), [Weights and Biases](https://wandb.ai/), [🤗 model hub](huggingface.co/models), [pytest](https://docs.pytest.org/en/7.0.x/), [nox](https://nox.thea.codes/en/stable/), [poetry](https://python-poetry.org/)

### Book Genre Detection

![](../images/genre-gradio.png)

This project created machine learning models which would predict whether a book was 'fiction' or 'non- fiction' based on the book title:

- The project was developed to address a gap in metadata in a large scale digitised book collection.
- The project used weak supervision to generate a more extensive training set beyond the initial human-generated annotations.
- Currently, two models are publicly available, one via the 🤗 [Model Hub](https://huggingface.co/BritishLibraryLabs/bl-books-genre) and one via [Zenodo](https://doi.org/10.5281/zenodo.5245175).
- The process of creating the models is documented in a [Jupyter Book](https://living-with-machines.github.io/genre-classification/intro.html). This documentation aims to communicate the critical steps in the machine learning pipeline to aid other people in the sector develop similar models.
https://huggingface.co/spaces https://huggingface.co/spaces/BritishLibraryLabs/British-Library-books-genre-classifier-v2

**Some of the tools used:** [fastai](https://docs.fast.ai/), [transformers](https://huggingface.co/docs/transformers/), [blurr](https://github.com/ohmeow/blurr), [Hugging face model hub](https://huggingface.co/models), [Jupyter Book](https://jupyterbook.org/), [Snorkel](https://github.com/snorkel-team/snorkel), [Gradio](https://gradio.app/)


## Datasets


### British Library books

- Extracted plain text and other metadata files from ALTO XML [https://github.com/davanstrien/digitised-books-ocr-and-metadata]()
- Added the dataset to the [🤗 datasets hub](https://huggingface.co/datasets/blbooks)

### British Library Books Genre data
- Created a datasets loading script and prepared a Dataset card for a dataset supporting book genre detection using machine learning: [https://huggingface.co/datasets/blbooksgenre]()

### Datasets to support programming historian lessons
I think having more realistic datasets is important for teaching machine learning effectively. As a result, I created two datasets for two under review [Programming Historian](https://programminghistorian.org/) lessons.

- [19th Century United States Newspaper Advert images with 'illustrated' or 'non illustrated' labels](https://doi.org/10.5281/zenodo.5838410)

- [19th Century United States Newspaper images predicted as Photographs with labels for "human", "animal", "human-structure" and "landscape"](https://doi.org/10.5281/zenodo.4487141)

### Workshop datasets 

- [Images from Newspaper Navigator predicted as maps, with human corrected labels](https://doi.org/10.5281/zenodo.4156510)

## Workshop materials

- [Computer Vision for the Humanities workshop](https://github.com/Living-with-machines/Computer-Vision-for-the-Humanities-workshop)
- [Working with maps at scale using Computer Vision and Jupyter notebooks](https://github.com/Living-with-machines/maps-at-scale-using-computer-vision-and-jupyter-notebooks)
- [Introduction to Jupyter Notebooks: the weird and the wonderful](https://github.com/Living-with-machines/Jupyter-Notebooks-The-Weird-and-Wonderful)

## Tutorials

- [Jupyter book showing how to build an ML powered book genre classifier](https://living-with-machines.github.io/genre-classification/intro.html)
- [A (brief) history of advertising in US Newspapers using computer vision](https://living-with-machines.github.io/nnanno/intro.html)
- (**Under review**)  [Computer Vision for the Humanities: An Introduction to Deep Learning for Image Classification, Programming Historian lessons](https://github.com/programminghistorian/ph-submissions/issues/343)
- (**Under development**) [Intro to AI for GLAM, Carpentries Lesson](https://carpentries-incubator.github.io/machine-learning-librarians-archivists/index.html)

# Code

You can view much of my code related activity on [GitHub](https://github.com/davanstrien)

# Publications

- [Semantic Scholar page](https://www.semanticscholar.org/author/Daniel-Alexander-van-Strien/71075073)
- [OCRCID page](https://orcid.org/0000-0003-1684-6556)
