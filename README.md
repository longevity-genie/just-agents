# just-agents
LLM agents done right, no over-engineering and redundant complexity!

# Motivation

Most of the existing agentic libraries are extremely over-engineered either directly or by using over-engineered libraries under the hood, like langchain and llamaindex.
In reality, interactions with LLMs are mostly about strings, and you can write your own template by just using f-strings and python native string templates. 
There is no need in complicated chains and other abstractions, in fact popular libraries create complexity just to sell you their paid services for LLM calls monitoring because it is extremely hard to understand what exactly is sent to LLMs.

We wrote this libraries while being pissed of by high complexity and wanted something controlled and simple.
Of course, you might comment that we do not have the ecosystem like, for example, tools and loaders. In reality, most of langchain tools are just very simple functions wrapped in their classes, you can always quickly look at them and re-implement them easier.

# Warning, the library is work in progress



# How it works

We use litellm library to interact with LLMs. 
All prompts that we use are stored in yaml files that you can easily overload.
It is assumed that a typical agent has role, goal and the background story.

```

```