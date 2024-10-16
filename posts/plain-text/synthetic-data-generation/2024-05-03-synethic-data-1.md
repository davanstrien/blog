---
description: _This post is part of a series on synthetic data generation techniques. You may also want to check out [Awesome Synthetic (text) datasets](https://github.com/davanstrien/awesome-synthetic-datasets), where I will be collecting these posts._
categories: [data, synthetic-data]
title: "Synthetic dataset generation techniques\\: generating custom sentence similarity data"
date: "2024-05-23"
---

_This post is part of a series on synthetic data generation techniques. You may also want to check out [Awesome Synthetic (text) datasets](https://github.com/davanstrien/awesome-synthetic-datasets), where I will be collecting these posts._

One of the most exciting use cases for LLMs is generating synthetic datasets that can be used to train non-LLM models. In the past, gathering enough data was one of the most significant barriers to training task-specific models, such as text classification models. LLMs can potentially help in this area.

## Creating data for training and fine-tuning embedding models using LLMs?

One area where synthetic data can be compelling is generating data for training sentence similarity models.

> Sentence Similarity is the task of determining how similar two texts are. Sentence similarity models convert input texts into vectors (embeddings) that capture semantic information and calculate how close (similar) they are between them. This task is particularly useful for information retrieval and clustering/grouping. [source](https://huggingface.co/tasks/sentence-similarity)

Whilst some strong open embedding models can be used for sentence similarity tasks, there are times when additional data for fine-tuning a model might be helpful:

- When working in a domain where generic models don't work well.
- When you want to optimize a model for a particular use, i.e. retrieval vs classification
- Scaling, scaling, scaling: you want to train a general embedding model and need more data.

For the latter example, LLMs are useful not only because they allow you to scale the amount of data but also because they allow you to control what data you have in your training data. Many embedding models use some weak supervision of data found "in the wild". While using this data allows the model to learn how to model similarity, there is also quite a lot of noise in this data. A recent paper [_Improving Text Embeddings with Large Language Models_](https://huggingface.co/papers/2401.00368) showed generating data that aimed to be diverse to the kinds of data an embedding model would work with reducing the amount of data needed compared to using a much larger but noisier weakly labeled dataset.

## What is similar

One frustration I've sometimes had when people discuss sentence similarity as a task is that what "similarity" means is usually pretty poorly defined (sorry, this is my humanities training rearing its head). This is one of the reasons why I really like the paper [Description-Based Text Similarity](https://arxiv.org/html/2305.12517v3). In this papers the authors describe one of the issues with existing approaches:

> The notion of similarity...is not explicitly defined but rather learned from vast datasets containing pairs of texts labeled as similar, often mixing various different kinds of similarity (Kaster et al., 2021; Opitz & Frank, 2022). This makes them sub-optimal for information seeking queries, as it is hard to control or predict the results of a given similarity-based query. What is a good query representation and similarity definition for a semantic-search use case?

The approach they take in their paper is to use an LLM to generate new query sentences which aim to be "abstract descriptions of sentences" which can be trained alongside their instantiations. To make it more clear here are some examples they produce in the paper:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/60107b385ac3e86b3ea4fc34/hf8ZT_hJfKoy8TyUEX-vw.png)

## Generating custom sentence similarity data

While this paper is focused on the task of generating 'abstract' queries for sentences the approach can be adapted to other more targeted similarity datasets. In the rest of this post I will briefly give some examples of how you can generate this kind of data (the full notebook in the Awesome Synthetic Datasets repo has the full code).

### Using Inference Endpoints via the huggingface_hub library.

In the paper the authors use GPT3.5 from OpenAI. In this post we'll switch that out with an open model [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) which we'll call via the `huggingface_hub` library.

First we can import the required libraries:

```python
from huggingface_hub import get_token
from huggingface_hub import InferenceClient
```

We can then use `InferenceClient` to specify which model we want to use.

```python
client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct", token=get_token())

```

### The prompts

For generating the descriptions of Wikipedia this prompt is used:

```python
wiki_prompt = f"""
Let's write abstract descriptions of sentences. Example:
Sentence: Pilate's role in the events leading to the crucifixion lent themselves to melodrama , even tragedy , and Pilate often has a role in medieval mystery plays .
Description: A description of a historical religious figure's involvement in a significant event and its later portrayal in art.
Note: Descriptions can differ in the level of abstraction, granularity and the part of the sentence they focus on. Some descriptions need to be abstract, while others should be concrete and detailed.
For the following sentence, write up 5 good and stand-alone, independent descriptions and 5 bad descriptions (which may be related, but are clearly wrong). Output a json file with keys 'good', 'bad'.
Sentence: {sentence}
Start your answer with a curly bracket.
"""
```

Let's generate some sentences using this prompt. We'll use this sentence as an example:

> "In Greek mythology, Achilles ( ) or Achilleus () was a hero of the Trojan War who was known as being the greatest of all the Greek warriors. A central character in Homer's Iliad, he was the son of the Nereid Thetis and Peleus, king of Phthia and famous Argonaut. Achilles was raised in Phthia along his childhood companion Patroclus and received his education by the centaur Chiron. In the Iliad, he is presented as the commander of the mythical tribe of the Myrmidons."

```python
resp = client.text_generation(wiki_prompt.format(sentence=sentence))
print(resp)
```

    {
    "good": [
    "A description of a mythological figure's background and characteristics",
    "A summary of a legendary hero's life and exploits",
    "A passage about a character from ancient Greek literature",
    "A biographical sketch of a famous warrior from mythology",
    "A description of a central character in a famous epic poem"
    ],
    "bad": [
    "A description of a real person's life",
    "A summary of a historical event",
    "A passage about a character from a novel",
    "A biographical sketch of a king",
    "A

We can see that we have roughly what's requests in the prompt but let's try and load this as JSON:

```python
import json

json.loads(resp)
```

    ---------------------------------------------------------------------------

    JSONDecodeError                           Traceback (most recent call last)

    Cell In[82], line 3
          1 import json
    ----> 3 json.loads(resp)


    File ~/.pyenv/versions/3.11.1/lib/python3.11/json/__init__.py:346, in loads(s, cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, **kw)
        341     s = s.decode(detect_encoding(s), 'surrogatepass')
        343 if (cls is None and object_hook is None and
        344         parse_int is None and parse_float is None and
        345         parse_constant is None and object_pairs_hook is None and not kw):
    --> 346     return _default_decoder.decode(s)
        347 if cls is None:
        348     cls = JSONDecoder


    File ~/.pyenv/versions/3.11.1/lib/python3.11/json/decoder.py:337, in JSONDecoder.decode(self, s, _w)
        332 def decode(self, s, _w=WHITESPACE.match):
        333     """Return the Python representation of ``s`` (a ``str`` instance
        334     containing a JSON document).
        335
        336     """
    --> 337     obj, end = self.raw_decode(s, idx=_w(s, 0).end())
        338     end = _w(s, end).end()
        339     if end != len(s):


    File ~/.pyenv/versions/3.11.1/lib/python3.11/json/decoder.py:353, in JSONDecoder.raw_decode(self, s, idx)
        344 """Decode a JSON document from ``s`` (a ``str`` beginning with
        345 a JSON document) and return a 2-tuple of the Python
        346 representation and the index in ``s`` where the document ended.
       (...)
        350
        351 """
        352 try:
    --> 353     obj, end = self.scan_once(s, idx)
        354 except StopIteration as err:
        355     raise JSONDecodeError("Expecting value", s, err.value) from None


    JSONDecodeError: Unterminated string starting at: line 14 column 1 (char 489)

## Structured Generation

One way we could help the model generate valid JSON is to increase the number of tokens. However, we can also use another approach, Structured Text Generation. This can be used to constrain the model's output to a more specific format.

We can use Structured Text Generation via Inference API models which are hosted using [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index). We won't discuss how this works under the hood in this post (see https://huggingface.co/docs/text-generation-inference/conceptual/guidance for a nice guide on this). We'll instead focus on how we can use this to improve the results we're getting from our open LLM.

When doing structured text generation we use something known as a "grammar" to specify what we want our output to look like. There are various ways of creating these but one way is to use a [Pydantic](https://docs.pydantic.dev/latest/) model. [Pydantic](https://docs.pydantic.dev/latest/) is a very heavily used data validation library for Python which can be used to validate data fits a certain format. This library was originally designed more for validating data coming via APIs etc but can also be very useful in the context of LLMs.

A simple way to define our data is to create a model called `Sentences` and specify that we want two attributes, `good` and `bad`. Each attribute should be a list of strings. You'll notice that in this example, these are specified via standard Python types.

```python
from pydantic import BaseModel

class Sentences(BaseModel):
    good: list[str]
    bad: list[str]
```

To use this model via the `huggingface_hub` library we need to pass it as a JSON Schema. Let's see what the schema for this model looks like:

```python
schema = Sentences.model_json_schema()
schema
```

    {'properties': {'good': {'items': {'type': 'string'},
       'title': 'Good',
       'type': 'array'},
      'bad': {'items': {'type': 'string'}, 'title': 'Bad', 'type': 'array'}},
     'required': ['good', 'bad'],
     'title': 'Sentences',
     'type': 'object'}

We can pass this schema to the `text_generation` method for our client.

```python
resp = client.text_generation(
    wiki_prompt.format(sentence=sentence),
    grammar={"type": "json", "value": Sentences.model_json_schema()},
    max_new_tokens=2000,
)
```

We can see that we can now load our response into a valid JSON object

```python
json.loads(resp)
```

    {'bad': ["Achilles' biography",
      'A description of a person',
      'A passage about a book',
      'A story about a king',
      'A summary of a myth'],
     'good': ["A description of a mythological figure's background and character in ancient Greek literature",
      'A characterization of a legendary warrior in Greek mythology',
      'A summary of the early life and education of a hero in ancient Greek mythology',
      'A description of a central character in a famous epic poem',
      "A portrayal of a mythological hero's family and upbringing"]}

### More control

The focus here isn't to dig into structured generation in great detail but you can add more control to your generations. For example, we might think that the `bad` examples generated above are too short. We can use `StringConstraints` to specify a minimum length for these strings.

```python
from pydantic.types import Annotated, StringConstraints

class Sentences(BaseModel):
    good: Annotated[list[str], StringConstraints(min_length=30)]
    bad: Annotated[list[str], StringConstraints(min_length=30)]
```

Finally, we could also specify how many generations we want. The original prompt specified a max of five examples for good and bad. We can add this constraint to our model.

```python
from pydantic.types import Annotated, StringConstraints
from pydantic import Field


class Sentences(BaseModel):
    good: Annotated[
        list[str],
        Field(
            max_items=5,
            item_type=StringConstraints(
                min_length=30,
            ),
        ),
    ]
    bad: Annotated[
        list[str],
        Field(
            max_items=5,
            item_type=StringConstraints(
                min_length=30,
            ),
        ),
    ]

```

## Abstract descriptions

One extra step the authors of the paper take is to use the descriptions generated by the first prompt along with a second prompt focused on generating a more abstract representation of the sentence. We'll quickly see one example of what this looks like using one of our examples:

```python
prompt_abstract = "Sentence: in spite of excellent pediatric health care , several educational problems could be noted in this tertiary pediatric center .\nDescription: Despite having advanced healthcare resources, certain deficiencies in education were identified at a medical center that serves children.\nA very abstract description: The provision of care at a specialized medical center was not optimal in one particular area, despite the presence of advanced resources.\nSentence: {sentence}\nDescription: {description}\nA very abstract description:"
```

```python
def generate_abstract_description(sentence, description):
    return client.text_generation(
        prompt_abstract.format(sentence=sentence, description=description),
    )

```

```python
description =json.loads(resp).get('good')[1]
```

Our original sentence and description look like this

```python
print(f"Sentence: {sentence}\nDescription: {description}\n")
```

    Sentence: In Greek mythology, Achilles ( ) or Achilleus () was a hero of the Trojan War who was known as being the greatest of all the Greek warriors. A central character in Homer's Iliad, he was the son of the Nereid Thetis and Peleus, king of Phthia and famous Argonaut. Achilles was raised in Phthia along his childhood companion Patroclus and received his education by the centaur Chiron. In the Iliad, he is presented as the commander of the mythical tribe of the Myrmidons.
    Description: A characterization of a legendary hero in a famous epic poem

```python
print(f"Abstract version: {generate_abstract_description(sentence, description)}")
```

    Abstract version:  A figure from ancient mythology is described in terms of their family, upbringing, and role in a famous story.

## Conclusion

Whilst there would still be some details to take care of if you want to scale up the generation of this kind of data, the above example hopefully shows how an open LLM can be used to generate more tailored data for training similarity models. Whilst the prompts here are borrowed from the paper, they could, of course, be adapted to focus on generating other kinds of similarity data.
