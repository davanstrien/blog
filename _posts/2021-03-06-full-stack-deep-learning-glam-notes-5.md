---
layout: post
descrption: full stack deep learning notes
categories: [deployment, ethics, glam]
title: "Full Stack Deep Learning notes on machine learning projects" 
---


# Introduction to my notes

These are my notes from the [full-stack deep learning course](https://fullstackdeeplearning.com/).

They have a  focus on the GLAM setting, which might have different requirements, incentives and resources compared to business deployments of ML projects. These notes are for my use primarily. I am posting them to make myself more accountable for making semi-coherent notes.

**These are my notes, written in a personal capacity - my employer denounces all my views**



# Lecture 5: Machine learning projects

This is the first &rsquo;proper&rsquo; lesson of the course with the previous ones focused on a review of deep learning approaches. This lecture focused on an overview of machine learning projects (beyond training a model)



## 85% of machine learning projects fail

This stat might be a bit questionable in general but likely has some truth to it. How does this differ in a GLAM setting? It might be more evident in a business setting if a project is successful or not based on a business need. In a GLAM setting, a project that is never "deployed"  might still by deemed successful. It is also my impression that there is a much closer overlap between academic/research projects and "business" projects in a GLAM setting. This might mean that the successful outcome is a published paper rather than a deployed model solving a "business" problem. 

## Lifecycle of a machine learning project

Covered in this lecture:

-  How to think about all of the activities in an ML project
- Prioritising projects: what is feasible?
-   Archetypes: categories of projects and the implications for project management
-   Metrics: how to pick a single number to optimise
-   Baselines: how to know if a model is performing well

Stages in the lifecycle are not linear but loop back and forth. i.e. you go back to project planning once you've started collecting data/labels because you realise the labels you collected aren't feasible to collect. This might be more important in a GLAM setting? Data collection usually takes place in a different way; for example, I don't know of any GLAMS that have extensively used paid crowdsource workers to collect training labels. This means that the label collection options might have to factor in more considerations around what is easy enough/fun enough.

Managing trade-offs between, e.g. accuracy and model latency



## Cross-project infrastructure and staffing

Should there be a separate team? Where should they sit in an organisation? 

Domain expertise is particularly important in a GLAM setting, so there might be risks with creating a team that does ML without this subject/domain expertise. On this point, I'm persuaded a lot by what fastai suggest about making deep learning accessible to domain experts. GLAMs should probably have some people who can support these efforts, but having a bunch of people hired as "the data science people" might be a bad idea if they don't work closely with others across an institution. Maybe I'm naive here that GLAMs would be willing to invest in broad training of many staff rather than hiring a few people. 🤷‍♂️



## Prioritising machine learning projects

How to decide what a suitable candidate for an ML project is?

### A (general) framework for prioritising projects

One way to identify projects is to try and find things that have a high impact and relatively low cost.


### Mental models for high-impact ML projects

Four mental models introduced for thinking about impact:

1.  Where can you take advantage of cheap prediction
2.  Where is there friction in your product?
3.  Where can you automate complicated manual processes?
4.  What are other people doing?

1.  Cheap predictions (1)

 For number 1, the idea of "cheap predictions" is useful even if the economics are different. Cheap predictions might be more about unlocking tasks that wouldn't otherwise be possible to do. Rather than using ML to replace something, it might make it feasible to do something which otherwise would be too expensive for the relative benefit.

2.  Where is there friction in your product? (2)

This objective seems important, but my examples of where GLAMS could make things easier for users would often not be simple projects to implement. They may also sit outside of what would be simple for a single institution to do on their own, or they may have already outsourced these systems. For example, cataloguing/discovery systems.

3.  Where can you automate complicated manual processes? (3)

For number 3, I would adjust this slightly to "where can you automate simple/boring manual processes" I think this is an area where there is a bunch of low hanging fruit for GLAMs. It is tempting to jump to more jazzy end-to-end models that will fully revolutionise cataloguing etc., but there are loads of tasks that GLAM staff have to do that:
- might be boring
- are probably difficult to solve using "traditional" programming i.e. writing a rules-based system would be difficult
- which has enough &rsquo;volume&rsquo; that it&rsquo;s worth automating.
- these are often also "staff facing" I think this reduced the potential risks of things going wrong versus something user-facing, for example, showing users a bunch of automatically generated tags for collection items?

4.  What is ML good at?

    I think this is a lovely consideration. I also believe this is something where energy shoud be spent in spreading the gospel of what ML is good at to staff working in GLAMS. This will make it much easier for people to self diagnose that something they are working with could be a good candidate for ml to help with and also helps demystify.
    
    The mental model of software 2.0 is a bit of a meme, but I think there is a lot of truth in there. The rules you would need to write to solve, for example, a relatively simple computer vision classification task, could be super complicated. Going back to my domain experts should use ml point:  I think that there are places where ml could be used as a "tool" in a glam setting in a much more mundane way for much more modest things. The actual coding ability level is potentially lower for this than creating complex rules-based approaches to the same task.

5.  What are other people doing?

  This should be an obvious one and ideally this will become more feasible as GLAMS get better at collaborating on sharing code/data (particularly labelled training data).
    
An interesting point here about how to find out what others are doing. I think GLAMs are good at shouting about ML stuff at the moment because it's super trendy. Still, there might be a lack of shouting about unsuccessful projects that others could learn from and/or shouting about more boring applications of ml that might be high impact but are not as obviously glamorous (pardon the pun).




### Feasability

What drives the feasibility of a project?

1.  Data availability

    -   how much data do you need?
    -   how stable is the data - if data updates in the real world all the time this will need to be replicated in the training data. This might be an argument for starting with more contained &rsquo;collections&rsquo; even if these collections are slightly arbitrary.
    -   how secure does the data need to be. Copyright might be a challenge. GLAMs should push Text and Data Mining Exemptions more forcefully for machine learning projects

2.  Accuracy requirements

    -   how costly are wrong predictions? This obviously depends on the problem, but it might be an argument for starting with "staff facing" or "stand alone" projects where there is either more human checking involved or the system's limitations can be made more explicit.
    -   how frequently does the system need to be correct to be useful?
    -   ethical implications of the model's accuracy - what is the harm of a model getting things wrong. Particularly important if the accuracy is not evenly distributed. For example, using Named Entity Recognition across a multi-language dataset with accuracy much higher for English compared to other results.
    -   pushing up accuracy tends to push up project costs because of more training data required, better label quality.

3.  Feasibility of the method

    -   What is still hard for ML? One rule of thumb suggested for what is possible with ml given:
    
    > "Pretty much anything that a normal (sic) person can do in <1 sec, we can now automate with ai"
    
    Leaving aside the loaded "normal person" this rule doesn't seem to work in many cases, as is pointed out in the lecture. There are a bunch of things that are easy for many people to do but aren't easy at all for an ML system. Jokes, sarcasm etc. are all challenigng for ml. 
    
    Still challenging:
    
    -   Pure unsupervised learning  difficult
    -   Reinforcement learning (outside of academia)


### How to run a feasibility assessment?

-   Are you sure you need ML at all? Many problems can still be solved with regex...
-   Put in work up-front to define success criteria with all of the stakeholders: think this one is super important. It could be tempting to be a bit hazy on this and make the success criteria "we did a cool ml project". This might be okay if the output is a paper/blog post but probably not if you're going to deploy something into &rsquo;production&rsquo;
-   Consider the ethics: consider what weird/bad results commercial black box APIs might give with heritage data. Consider whether you are pilling onto existing biases in your collection. Some of the labels/metadata GALMS use/have used is shitty, so maybe don&rsquo;t aim to get ml to produce even more of these&#x2026;
-   Do a literature review
-   Try to rapidly build a labelled benchmark dataset
-   build a minimal product to see if the problem seems viable (not using ml)


## Machine learning product archetypes



### Software 2.0

examples: code completion, recomender systems etc. In a GLAM setting I gues this would incldue catlogues, disvoery systems, document summerization etc.


<a id="org9a5b4e9"></a>

### Human-in-the-loop

Examples: turn sketches into slides, email auto-completion. In a GLAM setting, there are many potential examples, ML suggestions for cataloguing, document processing/classification/segmentation with some human review.



### Autonomous systems

Examples: self driving cars etc. My 2 c is that most (maybe all) GLAM institutions don&rsquo;t have sufficient experience with ML to really consider this. Exceptions may include robotics systems for some processes, but these are unlikely to be done "in house". Maybe I am missing something here?

## Data flywheel

sometimes you can get a positive feedback loop where you get more data -> better model -> more users - more data...

Things to consider. Where can this loop fail? The "more users" criteria might not translate in a glam setting, but perhaps building useful models/datasets that can then be extended on by other institutions is a similar positive feedback loop you can see. Transkribus is an example of this; they built a platform that got more data, which improved models that drove more people to consider it an option.


## Product design


### Principles from apple

How can you learn from your users? <- This is super relevant!

-   Explicit feedback (suggest less pop music)
-   implicit feedback (like this song)+ calibration during setup (scan your face for FaceID)
-   corrections (fix model mistakes) <- I think this is particularly important for successfully integrating into GLAM workflows and building trust that ML isn&rsquo;t going to be making decisions without possibilities for correction

### How should your app handle mistakes?

-   limitations (let users) know where to expect the model to perform well/badly
-   corrections (let users succeed even if the model fails), i.e. don&rsquo;t suggest labels that can&rsquo;t be ignored
-   attribution (help users understand where suggestions come from)
-   confidence (help users gauge the quality of results)


## Metrics

How to pick a single number to optimise?

The real world is messy, and you often care about lots of metrics. If you need to compare models/approaches, its very helpful to have a single number to optimise.
This can be done by combining different metrics.
the optimisation can change as things improve.
This single metric mindset is good for the model training part but should be balanced against other considerations as you move beyond this stage.


### Common metrics to pick from

- accuracy
- precision
- recall

Don't think we can generalise to saying one metric is more critical for glams, it really depends on the application.

Possible combinations weighted average between precision/recall.

More common in real-world is to use threshold n-1 metrics, evaluate to the nth. The idea here is that you pick one metric like accuracy to focus on but then try and improve that without making other things like the latency of the model much worse. This will be discussed more in future lectures.
An example: you need a model that will fit into the memory of a particular edge device, so you start with something that will do this and optimise from there.

### Choosing which metrics to the threshold

What should you optimise vs what should threshold?

Choosing metrics to  threshold:

- domain judgement: what can you engineer around? For example, maybe you can deal with a lower latency model by caching to get away with worse performance. In a glam setting, this will depend on the application, but if you are &rsquo;batch processing&rsquo; an existing collection you might not care about latency as much as if you are trying to use something which interacts with people in real-time, i.e. a cataloguing system.
- which metrics are least sensitive to model choice?
- which metrics are closest to desirable values.

Choosing threshold values:

- domain judgement - what is an acceptable tolerance downstream?
- how well does a baseline model do?
- how well does the model do already - if your model barely works, then its not going to be worth spending time trying to make it go faster&#x2026;

Once you have this baseline, you can eliminate models by successive metrics thresholds. i.e. first drop models with recall below n, then choose models which do well on precision given this minimal recall value.

It is important to try and be principled about which metric/combo you are going to use upfront as it will lead to different decisions being made about model choice etc. This might change, but its important to have something to drive decisions.


### Baselines

Why?

- Give you a lower bound for expected model performance
- the tighter the lower bound, the more useful

Having this baseline gives you some sense of how well/or crappy you are doing—otherwise its hard to make a judgement about next steps when training a model.

1.  Where to look for baselines?

- published results, make sure these are fair comparisons.
- Scripted baselines e.g. rule-based approaches
- could also use a simpler machine learning model as a baseline
-  or even take some kind of average of the dataset. sklearn has a bunch of dummy estimators useful for getting a baseline <https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation>

2.  How to create good human baselines
- random people quality might be low
- might require domain expertise
- trainign people on the task can improve quality
- ensembling of labels

# Conclusion

- ML lifecycle not linear
- projects that are high impact and wrong predictions not so bad
- secret sauce is to try and create automated data flywheels getting feedback from users of the model. much of this is about good product design
- metrics: having a central metric is important as a starting point
- baselines: help direct effort to know how the model is doing and where to direct energy


# suggested resources

-   <https://stanford-cs329s.github.io/syllabus.html>
-   <https://developer.apple.com/design/human-interface-guidelines/machine-learning/overview/introduction/>
-   <https://developers.google.com/machine-learning/guides/rules-of-ml/>

