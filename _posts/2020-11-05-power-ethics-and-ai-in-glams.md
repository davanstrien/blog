---
layout: post
descrption: Presenting ai generated labels
categories: [fastai course, ethics]
title: "Power, Ethics and AI for GLAMs"
---

This is a very quick rambling post on ethics/power and ai in a GLAM setting. 

On a recent(ish) call for the [fastai4fglams study group](https://github.com/AI4LAM/fastai4GLAMS) we discussed [Lecture 5](https://course.fast.ai/videos/?lesson=5) of the fast.ai course which focuses on Ethics.

A question we discussed in the call was about the more specific ethical issues that could emerge when using AI in a library context.

Some of the topics we touched on in the discussion included:

- The Provenance of collection items
- How this provenance is represented (or not) in metadata
- The use of Black Box commercial solutions for working with GLAM data
- The alternatives (or lack of) to using machine learning for some tasks
- How much to document the provenance of a label produced by ml

# How to present labels produced via machine learning

I should preface this discussion by saying I am not an expert in cataloging or the associated literature.

Instead of trying to cover all of these issues I will focus on one particular question for GLAMs using machine learning: how to display the labels produced by these models in library systems i.e. catalogues?

One of the main uses cases of Machine Learning models is to produce labels for collections items. These labels could either seek to 'augment' or 'replace' existing metadata records fields. A question for either of these is how to display, store and manage labels produced via machine learning models in library systems. For example a model which is trained to predict the `year` field for items in a particular collection.

## The model 

Knowing the model which produced a label including:

- the architecture
- the year in which the model was trained/inference made 
- version of the model
- ...

There is a danger of replicated the granularity of tools like [Weights and Biases](https://wandb.ai/). Whilst this information is useful for training and provenance for people familiar with machine learning methods, some of the information in these systems is going to be less relevant for a typical library catalog user e.g. GPU memory consumption during training. There are potentially some other fields which will be more relevant to a broader number of people.

## Possible labels

The first question for this type of task might be to know how the original model task was defined. Predicting labels for 'year' could be treated as a classification task. For example if you have a collection where you a certain that all items in that collection will have been produced between certain dates the task may be to predict classify whether the item belong to the 80s or 50s decade. In this case the model has a restricted range of possible outputs i.e. each decade in the original training data. 

Another approach would be to instead make this a 'regression' task, where the model predicts a continuous value. In this case the model is not bound by a particular set of possible years.

The distinction between a classification and a regression model, and the possible bound of values for a label might give the end-user of that label a better sense of how it should be 'read'. Using this information will also provide some way of contextualizing the confidence of the label, and what this might mean. An F1 score will mean different things if a model had to choose between one five possible decades for a year label or choose one of a 100 possible years.

Alongside knowing the potential labels, it may be useful for a user of a catalog to know something about the distribution of these labels i.e. how many times does a certain label appear both during the inference and training steps.

## Training data

Having at least a minimal way of interrogating the original training data could allow for more interrogation of models even by those who aren't experts in machine learning and in fact domain experts might pick up on important features of a training dataset that might have been missed during it's construction. Including some information about:

- number of training examples
- label distribution 
- min, max values 
- etc.

Ideally this dataset would be available for open use by others but this might not always be possible for collections which aren't fully open.

## How to present this information?

To be continued...