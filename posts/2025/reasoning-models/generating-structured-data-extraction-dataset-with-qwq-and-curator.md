---
title: "Using QwQ to generate a reasoning dataset for structured data extraction"
description: "Learn how to use QwQ-32B to generate synthetic reasoning datasets for training smaller models on structured data extraction tasks"
author: "Daniel van Strien"
date: "2025-03-11"
categories: ["huggingface", "datasets", "synthetic-data", "qwq", "reasoning"]
image: "https://github.com/davanstrien/blog/raw/refs/heads/main/posts/2025/reasoning-models/curator.webp"
twitter-card:
  title: "Using QwQ to generate a reasoning dataset for structured data extraction"
  description: "Learn how to use QwQ-32B to generate synthetic reasoning datasets for training smaller models on structured data extraction tasks"
  image: "https://github.com/davanstrien/blog/raw/refs/heads/main/posts/2025/reasoning-models/curator.webp"
  card-style: summary_large_image
open-graph:
  title: "Using QwQ to generate a reasoning dataset for structured data extraction"
  description: "Learn how to use QwQ-32B to generate synthetic reasoning datasets for training smaller models on structured data extraction tasks"
  image: "https://github.com/davanstrien/blog/raw/refs/heads/main/posts/2025/reasoning-models/curator.webp"
toc-depth: 3
toc: true
---

# Why reasoning models?

The past year has seen a lot of excitement about reasoning models. There are lots of nuances to what a reasoning model is, and how they are trained, but in practice, reasoning models contrast mostly from "traditional" LLMs by using some extra tokens during inference to "reason".

Where a traditional LLM might answer:

```
> User: what is 2+2?
> Assistant: 4
```

A reasoning model might answer:

```
> User: what is 2+2?
> Assistant: <thinking> The user is asking for the sum of 2 and 2.
The answer is 4 if we take this at face value but maybe...
Let's check my answer...
We should answer 4</thinking>
4
```

Allowing a model some extra time/tokens to "think" turns out to help it solve harder problems (at the cost of increased inference time and cost).

The release of the [DeepSeek R1](https://huggingface.co/DeepSeek/DeepSeek-R1) model caused a lot of excitement which reached outside of the AI community. As a result, there has been a lot of work on reasoning models in the open source AI community. This includes, efforts at fully open repliations of R1, as well as an increasining amount of datasets being released focused on reasoning --- more on this below.

## Reasoning outside of math and science?

To date, the majority of the focus of reasoning models has been on maths, coding, and science. One reason for this is that it is easier in these domains to have a verifiable answer or output to compare the LLM against.

Group Relative Policy Optimization (GRPO), introduced by the DeepSeek team in [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), is a variant of "Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO".

When training a reasoning model using GRPO, we can use a reward function to guide the model to reward completions that are correct. For example, we could reward completions that are correct. A pseudocode example of this might look like:

```python
def check_answer(completions, **kwargs):
    return ["some reward" if completion == "ground truth answer"
    else "no reward" for completion in completions]
```

In this case we have some ground truth answer, and we want to reward completions that match this answer. When we have a single correct answer, this is straight forward. In other cases we might need some other methods of "verification", for example for a code model we can check if code has valid syntax, complies, or passes a test.

However, there is nothing to prevent us from using GRPO for other tasks. For example, the [trl](https://github.com/huggingface/trl) docs for the [GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) have an example of using a reward function to reward completions that are more concise.

```python
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]
```

One of the key findings of the Deepseek paper is that during training, the model learns to allocate more thinking time to a problem by reavaulating it's initial answer. This allows the model to self-correct it's answer. This property can obviously be useful for tasks related to maths. If the model is able to "think" through a solution in steps, it can avoid the pitfalls of a single step reasoning i.e. imediately jumping to the answer. However, it is likely that this "reasoning" step can also be useful for other tasks.

We have some nice community examples of this. For, example [Ihor Stepanov](https://huggingface.co/blog/Ihor/replicating-deepseek-r1-for-information-extraction) has replicated the DeepSeek R1 approach for information extraction, something I'll also cover in a series of posts starting with this one.

## Distiling reasoning models for information extraction (and other tasks)

The DeepSeek team also showed in another paper [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) that it is possible to distill reasoning into smaller models. They showed this works for distiling the general reasoning abilities of larger models into smaller ones but it's likely that we can also distill reasoning for specific tasks (and into even smaller models).

The steps to do this are roughly:

- Generate a dataset of reasoning traces
- Do supervised finetuning (SFT) on this dataset

Alternatively, we may directly want to use GRPO to train a model for a particular task. In this case we usually still need a SFT step to train the model to understand the task and output we are looking for (although it may be possible to skip this step if the task is simple or you are starting from an instruction tuned model that already understands the task).

One of the challenges of both of these approaches has been the cost of generating the reasoning traces. Potentially a new reasoning model from Qwen could help with this.

## QwQ-32B: A reasoning model that can democratize reasoning dataset generation?

Qwen recently released [QwQ-32B](https://huggingface.co/QwQ-32B):

> QwQ is the reasoning model of the Qwen series. Compared with conventional instruction-tuned models, QwQ, which is capable of thinking and reasoning, can achieve significantly enhanced performance in downstream tasks, especially hard problems. QwQ-32B is the medium-sized reasoning model, which is capable of achieving competitive performance against state-of-the-art reasoning models, e.g., DeepSeek-R1, o1-mini. [source](https://huggingface.co/Qwen/QwQ-32B)

Whilst we already had access to models that could produce reasoning traces, QwQ-32B is much smaller than many of these models, making the cost and time involved in generating reasoning traces much more manageable. This may be especially important for more niche tasks where we will want to iterate on the dataset generation process. In the rest of this post, I'll show how we can use QwQ-32B to generate a dataset for the task of model card data extraction.

### Generating a reasoning dataset for model card data extraction

Models hosted on the Hugging Face Hub often have some form of model card. These are a form of documentation that describe the model, it's capabilities, and how to use it.
The model cards on the Hub contains two parts:

- Structred metadata about the model stored in a YAML section of the `README.md` file
- A free text description of the model, which may follow a model card template but increasingly often does not

We can easily extract infromation from the YAML section of the `README.md` file. For example, we can use the the ` pipeline_tag` information to filter for object detection models on the Hub (https://huggingface.co/models?pipeline_tag=object-detection&sort=trending).

However, there is often information stored in the free text description that could also be useful to know. For example, we may want to know the number of parameters in the model or it's context length. As an example we may want to extract something that looks a bit like this:

```python
example_output = {
    "base_model_mentions": None,
    "model_summary": "An open-weight text-to-speech model with 82 million parameters that delivers quality comparable to larger models while being faster and more cost-efficient.",
    "model_details": {
        "name": "Kokoro",
        "demo_url": "https://hf.co/spaces/hexgrad/Kokoro-TTS",
        "context_length": None,
        "parameter_count": 82000000,
    },
    "dataset_information": [
        {"name": "Koniwa", "url": "https://github.com/koniwa/koniwa", "type": "GitHub"}
    ],
    "technical_specifications": {
        "knowledge_cutoff": "2023-06",
        "inference_hardware": "Requires at least 4GB VRAM, runs on consumer GPUs like RTX 3060",
        "inference_libraries": ["Transformers"],
    },
    "card_quality": "good",
}
```

whilst it is possible to directly use a larger LLM (particularly with structured output) to extract this information, we may often benefit from a smaller lighter model for this kind of task. This becomes especially important if we want to use this model on large amounts of data (something which will often be the case when we want to extract structured data from a large number of documents).

### Generating a dataset using vLLM, Curator, and QwQ-32B

[Curator](https://docs.bespokelabs.ai/) is a relatively new synethic data generation libary for generating and curating synthetic data. It can be used with both API models and via vLLM which is how I used it in this case.

I already have a dataset of model cards [here](https://huggingface.co/datasets/librarian-bots/model_cards_with_metadata) which can be used as input to Curator. You can find the full script [here](https://huggingface.co/datasets/davanstrien/parsed-model-cards/blob/main/data_gen.py) but roughly the steps are:

1. Load the model card dataset and do some basic filering i.e. remove very short and long model cards
2. Define a prompt that asks the model to extract the structured information from the model card
3. Define a way in which Curator should parse the output of the model.
4. Generate a dataset using Curator

Here is the resulting dataset

<iframe
  src="https://huggingface.co/datasets/davanstrien/parsed-model-cards/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

The dataset generation was done on a Hugging Face Jupyter Space.

## Next steps

Once we have a dataset like this we can use it to train a model to extract structured information from model cards. There are various ways we could do this.

### Supervised finetuning to distill reasoning

For example, we could use a supervised finetuning on both the reasoning traces and the structured output.

I will cover this in a future post but as an example, here is a example of the output from a model trained on this dataset:

```python
{'base_model_mentions': [{'model_name': 'Qwen2-VL-7B-Instruct',
   'link': 'https://huggingface.co/datasets/allenai/olmOCR-mix-0225'}],
 'model_summary': 'A fine-tuned version of Qwen2-VL-7B-Instruct designed for generating images in a specific format using the olmOCR-mix-0225 dataset.',
 'model_details': {'name': 'olmOCR-7B-0225-preview',
  'demo_url': 'https://olmocr.allenai.org/',
  'context_length': None,
  'parameter_count': None},
 'dataset_information': [{'name': 'olmOCR-mix-0225',
   'url': 'https://huggingface.co/datasets/allenai/olmOCR-mix-0225',
   'type': 'Hugging Face'}],
 'technical_specifications': {'knowledge_cutoff': None,
  'inference_hardware': None,
  'inference_libraries': ['Transformers']},
 'card_quality': 'good'}
```

Here is a sample of some of the reasoning traces from this model

```python
Okay, let's tackle this query step by step. The user provided a model card for QwQ-32B and wants specific structured information extracted. Let me go through each part of the query and the model card to ensure I don't miss anything.

First, looking at the **base_model_mentions** section. The task is to list models explicitly stated as base/foundation models used for fine-tuning. The card mentions that QwQ is based on Qwen2.5, whose code is from the latest Hugging Face transformers. The link is provided as https://huggingface.co/Qwen/QwQ-32B. So that should be in the list.

Next, the **model_summary** needs a single-sentence summary. The primary purpose is reasoning capacity, built on Qwen2.5, with 32B parameters. It has 32.5B parameters before post-training, and its performance on downstream tasks like math problems and hard questions. The summary should include the main function (reasoning), base model, parameters, and unique features like enhanced performance and context handling.

For **model_details**, the name is "Qwen/QwQ-32B" as per the repo name. Demo URL is https://huggingface.co/spaces/Qwen/QwQ-32B-Demo. They mention using the transformers library, so the Python code snippet's code variable is from transformers.context_length is Full 131,072, parameter count 32.5 billion (32.5B), and non-hidden parameters count 31.0B. But wait, the model has 32.5B before post-training, so parameter_count should be 32.5B. The context_length is mentioned in the requirements as Full 131,072 tokens. So for the parameters, the non-hidden part might need to be 31.0B.

**dataset_information** requires datasets used. The card mentions training stages pre-training with DeepSeek-R1 and o1-mini, but the main training used Qwen2.5's technical report. The dataset for post-training is not specified here, so maybe the datasets are already covered under the base model or other parts of the card. The training data isn't explicitly listed except mentioning post-training and the reports of those using their platforms. Since the exact training datasets aren't named here, perhaps there's no direct dataset info provided. Wait, the requirements mention enabling YaRN and using vLLM, which might involve specific datasets, but without explicit dataset names, maybe there's no direct dataset info here. So dataset_information would be null? Or do I look into the references? The card says "Dealing with their [fellow researchers'] work, they mention their own results and not those of others on the training data. " So it might not have explicit dataset names, so maybe leave it as an empty array?

Technical specifications: knowledge_cutoff is not explicitly stated. The training data's latest date isn't mentioned, so it's null. Inference hardware isn't specified except mentioning vLLM and the author's mention of using vLLM's setup. The requirements say to use the transformers library, but the example code uses PyTorch, but the requirements apply to the model's architecture. The card says "transformers<4.37.0", so probably uses the current version. The inference_libraries are mentioned as "transformers" since it's part of the code snippet. So the libraries are transformers.

**card_quality** assessment. Let's look at the criteria:
```

While this doesn't look terrible, it does seem quite verbose. This is potentially where using GRPO could help.

### GRPO for structured output

Whilst SFT can help us distill reasoning into a model, for a specific task like structured output extraction, we may want to use GRPO to more carefully control the output using reward functions. This may also allow us to encourage the reasoning to be more concise (whilst ensuring the output is still correct according to some criteria we define). We'll cover this in a future post!
