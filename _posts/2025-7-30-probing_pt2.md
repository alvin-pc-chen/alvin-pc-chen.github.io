---
layout: distill
title: Probing Part 2
description: Further investigation into LLM linguistic knowledge.
tags: probing, bertology, mechinterp, graphspect, UMR, compling, LLMs, aspect
giscus_comments: true
date: 2025-07-19
featured: true
# mermaid:
#   enabled: true
#   zoomable: true
# code_diff: true
# map: true
# chart:
#   chartjs: true
#   echarts: true
#   vega_lite: true
# tikzjax: true
# typograms: true
categories: research

toc:
    - name: 
---

_This is part 2 of my probing investigation where I try to identify if/how LLMs encode linguistic aspect. Part 1 contains relevant background and can be found [here](./2025-7-19-probing.md)._

My first investigation was an MVP using a simple dataset with binary labels to show that probing works as a strategy. Let's now take a look at a more complicated corpus and see what we can make of it. For the NeuroSpect project, our 8-person team manually annotated some 2000 predicates using the [UMR framework](https://github.com/umr4nlp/umr-guidelines/blob/master/guidelines.md#part-3-3-1-Aspect) to use for modeling. English predicates in UMR generally fall into one of 6 aspect categories, but crucially, not all identified predicates are events for which we assign the _None_ label.

Sentences often have multiple predicates, some really close to each other.

## Covering the basics
Let's run the same experiment again to make sure that the probes can pick up the same things with this new dataset. We already saw that only the closest token positions showed any significant performance, so let's narrow our search to ±5 positions:

F1 HERE

Since there are 7 labels with drastically different occurrences in the data (stats), F1 is a much better metric than accuracy. A classifier that predicts _None_ (or state?) every time will give an accuracy of 0.35 but the input embeddings may as well be random.

## Depending on Dependencies