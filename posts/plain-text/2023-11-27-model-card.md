---
description: "What do people talk about in their model cards?"
categories: [Hugging Face]
title: "Extracting Insights from Model Cards Using Open Large Language Models"
date: "2023-11-27"
---

[Model Cards](https://huggingface.co/docs/hub/model-cards) are a vital tool for documenting machine learning models. Model Cards are stored in `README.md` files on the Hugging Face Hub.

There are currently over 400,000 models openly shared on the Hugging Face Hub. How can we better understand what information is shared in these model cards?

<p align="center"> 
 <img src="https://cdn-uploads.huggingface.co/production/uploads/60107b385ac3e86b3ea4fc34/of0NdtzeiXm6JEN2HCCAE.png" alt="Wordcloud image with words like training, model, information. "><br> <em>Some of the concepts we'll see emerge from Model Card READMEs</em> 
 </p>

## What do people talk about in their model README.md?

Various organisations, groups and individuals develop models on the Hugging Face Hub; they cover a broad range of tasks and have a wide variety of audiences in mind. In turn, READMEs for models are also diverse. Some READMEs will follow a Model Card [template](https://huggingface.co/docs/hub/model-card-annotated), whilst others will use a very different format and focus on describing very different attributes of a model. How can we better understand what people discuss in model cards?

## Can we extract metadata from model READMEs?

One of the things I want to understand better is what information people are talking about in their READMEs. Are they mostly talking about the training? How often do they mention the dataset? Do they discuss evaluations in detail? Partly, I want to understand this purely out of curiosity, but I am also interested in knowing if there are features that regularly appear in model cards that could potentially be extracted into more structured metadata for a model.

As an example of this kind of work, recently, the Hub added a metadata field for `base_model`. This metadata makes it easier to know the model used as a starting point for fine-tuning a new model. You can, for example, find models fine-tuned from [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) using this filter [https://huggingface.co/models?other=base_model:mistralai/Mistral-7B-v0.1](). However, for this to be possible, the `base_model` field has to be stored as metadata. In the run to adding this `base_model` filtering to the Hub via [Librarian-Bots](https://huggingface.co/librarian-bots), I made a bunch of automated pull requests adding this metadata using the information available in the model `README.md`.

<p align="center"> 
 <img src="https://cdn-uploads.huggingface.co/production/uploads/60107b385ac3e86b3ea4fc34/VHDwba0v0YB4IZCbMBtnh.png" alt="Screenshot of a Pull Request"><br> 
<em>An example of a pull request made to add metadata to a model</em> 
 </p>

Potentially, other data of this kind could also be drawn out of model cards and exposed in a more structured way, which makes filtering and searching for models on the Hub easier.

## Annotating with Large Language Models?

As part of my work as Hugging Face's Machine Learning Librarian, I have created a [dataset](https://huggingface.co/datasets/librarian-bots/model_cards_with_metadata) of model cards from the Hugging Face Hub. This dataset is updated daily. This dataset currently has over 400,000 rows. This makes analysing this data by hand difficult.

A recent [blog post](https://www.numind.ai/blog/a-foundation-model-for-entity-recognition) from NuMind discusses their approach to creating a foundation model for Named Entity Recognition. As part of this work, they created a large dataset using an LLM to annotate concepts &#8212; the term they use for entities &#8212; in a large dataset derived from the Pile. They do this by prompting the model to annotate in an open-ended way, i.e. instead of prompting the model to label specific types of entities; they prompt the model to label "as many entities, concepts, and ideas as possible in the input text."

Whilst we sometimes want to have an LLM help annotate a specific type of entity, this open approach allows us to use an LLM to _help_ us explore a dataset.

In the NuMind work, they used GPT-4. I wanted to use an open LLM instead. After some exploring I landed on [teknium/OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B):

> OpenHermes 2.5 Mistral 7B is a state of the art Mistral Fine-tune, a continuation of OpenHermes 2 model, which trained on additional code datasets.

I found that the model responded well to an adapted version of the original prompt used by NuMind, and since the model is a 7 Billion parameter model, it's a little expensive to run both financially and in terms of environmental impact compared to other larger models which could also be used for this task.

I hosted the model on [Inference Endpoints](https://huggingface.co/inference-endpoints) and ran inference using the `huggingface_hub` Python library. The code for getting an annotation looked roughly like this:

```python
def get_annotations(input):
    message = f"""
    The goal is to create a dataset for entity recognition.
    Label as many entities, concepts, and ideas as possible in the input text.
    Invent new entity types that may not exist in traditional NER Tasks such as more abstract concepts and ideas.
    Make sure the entity concept is not part of speech but something more meaningful.
    Avoid finding meaningless entities.
    Output format (separate entities with new lines, everything, including description, and entity concept is written in English): entity from the text -|- entity concept -|- description of entity group/concept.

    Example:

    Input: "Fine-tuned XLSR-53 large model for speech recognition in English"

    Output:

    XLSR-53 -|- model -|- a large pre-trained language model specifically designed for speech recognition in English.

    English -|- language -|- the language of the text and the model's target language.

    Fine-tuned -|- model modification -|- the process of adapting the pre-trained model to a specific task, in this case, speech recognition.

    Input: "{input}"

    Output:"""
    messages = [
        {
            "role": "system",
            "content": "You are Hermes 2. A system designed to annotate textual data",
        },
        {"role": "user", "content": message},
    ]
    gen_input = tokenizer.apply_chat_template(messages, tokenize=False)

    return client.text_generation(
        gen_input,
        max_new_tokens=450,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
```

Some examples of output I got from this:

```
Input: Fine-tuned XLSR-53 large model for speech recognition in English

Output:

XLSR-53 -|- model -|- a large pre-trained language model specifically designed for speech recognition in English.

English -|- language -|- the language of the text and the model's target language.

Fine-tuned -|- model modification -|- the process of adapting the pre-trained model to a specific task, in this case, speech recognition.
```

As you can see, the model did a pretty good job of labelling concepts. Let's take a deeper dive into the results.

## What concepts did we find in Model Cards?

The dataset we've produced via this approach contains annotations for the original entity/concept, i.e. the word that the model annotated, a "category", which is the type of that concept (as labelled by the model), as well as a description produced by the LLM about that category.

To start, here is some high-level information about our dataset:

- 146,800 total annotations, i.e. concepts
- 46,240 unique subjects
- 16,581 unique categories

We can see the number of unique subjects and even unique subjects and categories. Whilst this wouldn't be desirable if we had a fixed set of labels we wanted to annotate, for this more open-ended exploration, this is less of an issue and more of a challenge for us in how best to understand this data!

### The most frequently appearing subjects

To start with let's take a look at the top 20 most frequently appearing subjects in our model cards:

| subject                 | proportion (%) |
| ----------------------- | -------------- |
| Training                | 1.00272        |
| Entry                   | 0.807221       |
| More                    | 0.651226       |
| Model                   | 0.612398       |
| model                   | 0.54564        |
| information             | 0.504087       |
| needed                  | 0.501362       |
| Limitations             | 0.472071       |
| More Information Needed | 0.433243       |
| learning_rate           | 0.398501       |
| Fine-tuned              | 0.387602       |
| Transformers            | 0.378747       |
| Tokenizers              | 0.376703       |
| Intended uses           | 0.370572       |
| hyperparameters         | 0.361717       |
| Evaluation              | 0.360354       |
| Training procedure      | 0.358992       |
| Versions                | 0.352861       |
| Adam                    | 0.34673        |
| Hyperparameters         | 0.343324       |

We may see some of these terms like "More", "information", "needed" are artefacts from placeholder text put into the model card templates. It's reassuring to see "Evaluation" and "Intended uses" appearing this frequently. Since the "subjects" are quite diverse, let's also take a look at the 20 most common categories:

| category           | proportion |
| ------------------ | ---------- |
| model              | 3.97752    |
| model modification | 2.29837    |
| numerical value    | 2.01226    |
| action             | 1.68869    |
| dataset            | 1.53951    |
| metric             | 1.25409    |
| process            | 1.23229    |
| software           | 1.22684    |
| entity             | 1.15736    |
| software version   | 1.14237    |
| concept            | 1.09741    |
| data               | 0.936649   |
| data type          | 0.867166   |
| person             | 0.787466   |
| quantity           | 0.773842   |
| organization       | 0.730926   |
| language           | 0.729564   |
| library            | 0.647139   |
| numeric value      | 0.626022   |
| version            | 0.613079   |

We would expect to see many of these categories, i.e. 'model' and 'model modification', 'numerical value'. Some of these categories are a little more abstract, i.e. 'action'. Let's look at the `description` field for some of these:

```
['the process of adding new software to a system.',
 'the action of preserving or retaining something.',
 'an invitation to interact with the content, usually by clicking on a link or button.',
 'the action of visiting or viewing the webpage.',
 'the interaction between the user and the software.']
```

and at the actual "subjects" where this has been applied

```
['install', 'Kept', 'click', 'accessed', 'experience']
```

We can also view these categories as a wordcloud (the subject wordcloud is at the start of this post)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/60107b385ac3e86b3ea4fc34/Pf7-6Kx3O_O13rZL03HgF.png)

## What can we extract?

Coming back to one of the motivations of this work, trying to find 'concepts' in model cards that could be extracted as metadata, what might we consider interesting to extract? From the categories, we can see datasets appear frequently. While I have already done some work to extract those, there is more to be done on extracting all dataset mentions from model cards.

Whilst likely more challenging, we can see that the 'metric' category appears pretty often. Let's filter the dataset to examples where the category has been labelled 'metric' and take the top 30 most frequent examples.

| subject              | proportion |
| :------------------- | ---------: |
| Validation Loss      |    11.6212 |
| Training Loss        |    7.63271 |
| Accuracy             |    6.97274 |
| Loss                 |    6.74319 |
| f1                   |    5.39455 |
| accuracy             |    3.18508 |
| results              |    2.38164 |
| recall               |      2.066 |
| Recall               |      2.066 |
| precision            |    1.95122 |
| Validation Accuracy  |    1.66428 |
| Results              |     1.5495 |
| 'f1'                 |    1.46341 |
| 'precision'          |    1.43472 |
| 'recall'             |    1.43472 |
| Train Loss           |    1.34864 |
| training_precision   |    1.34864 |
| Precision            |    1.29125 |
| Rouge1               |   0.774749 |
| "accuracy"           |   0.774749 |
| Validation           |   0.659971 |
| Performance          |   0.631277 |
| Rouge2               |   0.573888 |
| Train Accuracy       |   0.573888 |
| F1                   |   0.516499 |
| Bleu                 |   0.487805 |
| Iou                  |    0.45911 |
| num_epochs           |    0.45911 |
| Micro F1 score       |   0.373027 |
| Matthews Correlation |   0.344333 |

While the results here are a little noisy, with a bit of work, we can potentially begin to think about how to extract mentions of metrics from model cards and merge duplicated metrics that have been expressed differently. This sort of data could start to give us very interesting 'on-the-ground' insights into how people are evaluating their models.

## Conclusion

If you want to play with the results yourself, you can find the full dataset here: [librarian-bots/model-card-sentences-annotated](https://huggingface.co/datasets/librarian-bots/model-card-sentences-annotated).

You may also want to check out the [Model Card GuideBook](https://huggingface.co/docs/hub/model-card-guidebook)

If you have other ideas about working with this kind of data, I'd love to hear from you! You can follow me on the [Hub](https://huggingface.co/davanstrien) (you should also follow [Librarian bot!](https://huggingface.co/librarian-bot)).
