# LLM Peer Scrutiny

This directory contains the results of the "Adversarial Peer Review" performed by frontier LLMs on the EML project.

## Scrutiny Methodology
To ensure the highest standard of technical honesty, we subjected the blog post and code to "appropriate brutal" feedback from **GPT-5.4**, **Claude Opus 4.6**, and **Gemini 3.1** via the OpenRouter API.

## Contents
- **`1_Narrative_Code_Review...`**: Feedback on the engagement, hook, and "marketing speak" of the launch.
- **`2_Formal_Methods_Review...`**: Expert-level audit of the Lean 4, Coq, and TLA+ specifications.
- **`3_Systems_Algorithmic_Review...`**: Audit of the Z3 scripts and JML contracts.
- **`GPT_5.4_Image_Prompts...`**: High-quality editorial prompts designed for `gpt-image-2`.

## Lessons Learned
The models correctly flagged several "dangling claims" and hardware anti-patterns, which we have addressed in the final production blog post.
