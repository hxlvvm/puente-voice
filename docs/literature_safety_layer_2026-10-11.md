# Literature: model-based emergency / self-harm detection (Consensus search, 2026-10-11)

Query: detecting suicidal ideation or medical emergency in short multilingual Spanish/English utterances with
zero-shot or small language-model classifiers. Only 3 results on the free tier; none covers Spanish phone speech or
physical emergencies (chest pain, stroke, overdose), so these bound the self-harm category only.

1. Deng et al. 2025, IEEE JBHI, "Evaluating LLMs in Crisis Detection: a real-world benchmark from psychological support
   hotlines" (PsyCrisisBench, 540 hotline transcripts, 64 LLMs): suicidal-ideation F1 0.880; few-shot and fine-tuning
   help; **a fine-tuned Qwen2.5-1.5B beat larger models** on suicidal-ideation detection.
   https://consensus.app/papers/details/e2fff13a23765c369038fa855872cd6c/
2. Nikmehr et al. 2025: zero-shot LLM prompting matches or exceeds fine-tuned RoBERTa on social-media suicidal
   ideation. https://consensus.app/papers/details/a51ac191bfaa5a0c9e9a2de8012235c6/
3. Abusaqer et al. 2026, ACM Southeast: fine-tuned RoBERTa 93.2 % accuracy beats zero-shot GPT-4o / Claude 3.5 /
   DeepSeek-R1 (90.4-90.5 %). https://consensus.app/papers/details/6e907365dea3548180318bee3f33a201/

Takeaway: a small local model (about 1.5B parameters) fine-tuned on in-domain examples is competitive with large zero-shot models.
That fits the constraints here: local, low latency, no PHI to hosted APIs. All three are text-only, English or Chinese,
and accuracy/F1-based; none reports recall at a fixed false-alarm rate, which is the number this product needs.
