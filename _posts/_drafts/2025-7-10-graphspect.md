---
layout: post
title: GraphSpect
date: 2025-07-10 21:01:00
description: placeholder
tags: graphspect, neuro-symbolic, GNNs, umr, compling, LLMs, aspect
categories: research
thumbnail: assets/img/blog/graphspect/profile.png
pretty_table: true

---

**Why care?**

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog/graphspect/intro.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    UMRs aren't real. They can't hurt you.
</div>

Converting tons of AMR graphs for the [UMR Project](https://umr4nlp.github.io/web/index.html) showed me the immense effort needed to annotate UMRs, let alone creating graphs from scratch. Despite releasing [a colossal dataset](https://lindat.mff.cuni.cz/repository/items/239427de-bcaa-401d-a0ae-2c69602daa67), we still have a long way to go to create a solid corpus of these graphs, which is why I'm trying to automate the aspect annotation process. My first thought was to test LLMs, since everyone knows that GPT can annotate whatever you want, and anyway, semantic frameworks are so last century... right?

### TL;DR
1. UMRs improve on the [AMR framework](https://en.wikipedia.org/wiki/Abstract_Meaning_Representation), which has broad applications ranging from biomedical parsing to human-robot interfaces.
2. LLMs still have trouble capturing linguistic aspect, a key property of language.
3. 

### WhoMRs?
For the uninitiated, [Abstract Meaning Representation](https://en.wikipedia.org/wiki/Abstract_Meaning_Representation) is a popular framework for encoding the semantics of natural language. While LLMs are extremely powerful and can flexibly handle a broad array of tasks, structured representations are still important when specificity and standardization is required. Consider the following sentences:

> The boy desires the girl to believe him.
>
> The boy desires to be believed by the girl.
>
> The boy has a desire to be believed by the girl.
>
> The boy’s desire is for the girl to believe him.
>
> The boy is desirous of the girl believing him.

As humans we understand that these sentences mean roughly the same thing, but 

### 