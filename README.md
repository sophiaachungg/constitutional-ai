# Constitutional AI: Harmlessness from AI Feedback
*Bai et al., 2022 (Anthropic)*

Paper presentation repo for DS 5690

## Overview
### Core problem:
LLMs are trained on data from the best **and worst** of the Internet. As the saying goes, *"Garbage in, garbage out."* This leads to the problem of LLMs generating plenty of responses that are unhelpful, factually incorrect, or actively harmful. 

### Current solution:
Reinforcement Learning from Human Feedback (RLHF)

### Basics of RLHF:
1. **Collect human data.** Humans write prompts and rank several of the model's responses from best to worst.
2. **Train preference model.** A "judge" model is trained on human ranking data (above). Its job is to predict which response a human would prefer.
3. **Finetune with reinforcement learning.** The main language model is finetuned using reinforcement learning, generating responses while the judge model scores them. The main model's weights are adjusted to maximize the reward given by the preference model.

### Issues that arise:
* RLHF encourages evasiveness
* Alignment problem - How do you steer a model to be helpful, honest, and harmless?
* Helpful-Harmless spectrum - How do you strike a balance between helpfulness vs harmlessness?
* Scalability - Human labelers are slow and expensive. As LLMs continue to grow in scale, how can the ethics of the AI scale in tandem?
* Transparency - How do you know what values are taught to the model if they're hidden within the human judgments?

## Proposed solution
### Overview of Constitutional AI Approach
- Human supervision from a set of principles governing AI behavior + few-shot prompting examples = constitution
- Leverages chain-of-thought reasoning to make decision making in the scalable supervision stage more legible and auditable

[insert photo of Fig 1]

### Training Phase 1: Supervised Learning (SL-CAI) Stage 
Goal: Maximize model's ability to be harmless
1. Start with a model that is already good at following instructions
2. Model is given "red team" prompts designed to elicit harmful responses
3. For each harmful response, model has to 1) critique its own output based on a constitutional principle, and 2) revise it to be harmless
4. New model is finetuned on the final, harmless "revised" responses.

In summary, Response → Critique → Revision.

### Training Phase 2:  Reinforcement Learning (RL) Stage 
Goal: Further improve model's harmlessness and reliability using AI-generated feedback
1. Use the SL-CAI model to generate 2 responses for each harmful prompt.
2. Judge/preference model is shown the prompt and 2 responses and chooses the better one (less harmful) according to a constitutional principle.
3. Generate massive dataset of AI preferences from step 3.
4. New "reward model" is trained on the dataset of AI preferences, learning to predict which response the judge model would prefer.
5. SL-CAI model finetuned using reinforcement learning. In other words, the preference model provides a reward for generating responses that the judge model scores highly.

In summary, AI Comparison Evaluations → Preference Model → Reinforcement Learning

## The Result
The final model should be a highly harmless and non-evasive assistant.
