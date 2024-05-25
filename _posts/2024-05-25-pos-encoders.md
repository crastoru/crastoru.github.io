---
layout: post
title: Geospatial Position Encoders: A Deep Dive
---

An inductive bias in machine learning is a constraint on a model or its input features given some prior knowledge of the target task. As humans, we can recognize a bird whether it’s flying in the sky or perched in a tree. Moreover, we don’t need to examine every cloud in the sky or take in the entirety of the tree to know that we are looking at a bird and not something else. These biases in the vision process are encoded in convolution layers via two properties: 

- *Weight-sharing*: the same kernel weights are used across each part of the input  
- *Locality*: the kernel has a much smaller width and height than the input 

We can also encode inductive biases in our choices of input features to the model. For example, if we are training a regression model to predict average house prices in a neighborhood, we may prefer to use median income or school district as inputs to the regression model rather than, say, the number of letters in the name of the neighborhood. 

Designing input features for a neural network involves a trade-off between expressiveness and inductive bias. On one hand, we want to allow our model the flexibility to learn patterns beyond what we humans can detect and encode. On the other hand, a model without any inductive biases will struggle to learn anything meaningful at all. 

 In this article, we will explore the inductive biases that go into designing effective position encoders for geographic coordinates. Position on Earth can be a useful input to a wide range of prediction tasks, including average house price prediction. As we will see, using latitude and longitude directly as input features is under-constraining and ultimately will make it harder for our model to learn anything meaningful. Instead, it is more common to encode prior knowledge about latitude and longitude in a nonparametric re-mapping that we call a positional encoder. 