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

## Pseudocode
### Stage 1: Supervised Learning
```
initial_model = load_model("helpful-only-rlhf-model")
constitution = load_principles("harmlessness_principles.txt")
red_team_prompts = load_prompts("harmful_prompts.txt")

# Create a dataset of harmless revisions
harmless_finetuning_data = []
for prompt in red_team_prompts:
    # 1. Generate an initial (likely harmful) response
    initial_response = initial_model.generate(prompt)

    # 2. Use the constitution to critique the response
    principle = constitution.sample()
    critique_prompt = f"{prompt}\nResponse: {initial_response}\n\nCritique this response based on: {principle}"
    critique = initial_model.generate(critique_prompt)

    # 3. Use the critique to revise the response
    revision_prompt = f"{critique_prompt}\nCritique: {critique}\n\nNow, revise the initial response."
    revised_response = initial_model.generate(revision_prompt)

    # 4. Save the prompt and the final harmless revision for training
    harmless_finetuning_data.append({"prompt": prompt, "completion": revised_response})

# 5. Finetune a new model on the self-corrected data
sl_cai_model = finetune(base_model="pretrained_lm", data=harmless_finetuning_data)
```

### Stage 2: 
```
# Part A: Generate AI Preference Data
sl_cai_model = load_model_from_stage_1()
feedback_model = load_model("pretrained_lm") # The AI Judge
ai_preference_dataset = []

for prompt in red_team_prompts:
    # 1. Generate two different responses
    response_A = sl_cai_model.generate(prompt)
    response_B = sl_cai_model.generate(prompt)

    # 2. Ask the AI judge to choose the better one based on the constitution
    principle = constitution.sample()
    choice_prompt = f"Prompt: {prompt}\n\nWhich response is less harmful according to '{principle}'?\n\n(A) {response_A}\n(B) {response_B}"
    
    # The feedback model outputs its preference, e.g., "(A)"
    ai_choice = feedback_model.generate(choice_prompt)
    
    # 3. Store the AI's preference
    if ai_choice == "(A)":
        ai_preference_dataset.append({"chosen": response_A, "rejected": response_B})
    else:
        ai_preference_dataset.append({"chosen": response_B, "rejected": response_A})

# Part B: Train PM and use for RL
# 4. Train a Preference Model (PM) on the AI-generated labels
preference_model = train_preference_model(ai_preference_dataset)

# 5. Use the PM as a reward signal to finetune the SL model with RL
final_rl_cai_model = finetune_with_rl(
    policy_model=sl_cai_model,
    reward_function=preference_model
)
```

