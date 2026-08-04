# Related Work & Metrics — рабочие заметки команды

Тезисы по прочитанным статьям и таблица соответствия метрик. Черновики секций статьи вынесены в
[`paper/introduction.tex`](../paper/introduction.tex) и [`paper/related_work.tex`](../paper/related_work.tex).
Номера статей — по [papers/README.md](../papers/README.md). Ведёт M3.

This file contains the literature takeaways, the scientific metrics mapping table, and the LaTeX draft for the paper's Related Work section.

---

## 1. Literature Takeaways (M3 conveyor)

### 01 — AQUA-Bench (Kuan & Lee, arXiv 2601.12248, Jan 2026)
- **Core task:** Benchmark for unanswerability in audio QA across three scenarios: absent answer options, incompatible answer sets, and incompatible audio–question pairs.
- **Key findings:** Evaluation on animal sounds, instruments, and vocal sounds shows that while ALLMs excel on standard answerable tasks (90-95% accuracy), their performance drops significantly on unanswerable tasks due to a strong "forced-choice bias".
- **What we adopt / contrast:** AQUA-Bench uses a standard multiple-choice (MCQ) format on general sound events. Our study focuses on free-form spoken-content QA (speech comprehension), providing a head-to-head comparison of abstention strategies rather than just measurement.

### 02 — Towards Reliable LALM (Ma et al., arXiv 2505.19294, May 2025)
- **Core task:** Systematically investigates training-free (IDK prompting, MCoT prompting, Task Agent) and training-based (SFT/LoRA) methods to make audio-LLMs reject what they do not know.
- **Key findings:** Introduces the Reliability Gain Index (RGI) to evaluate how well a method balances conservativeness (rejecting correct answers) versus humbleness (rejecting wrong answers). Finds that reliability awareness is a cross-modal "meta-ability" that can transfer from speech to music/sound.
- **What we adopt / contrast:** We use their IDK-prompting style as our baseline S1 strategy and adopt their RGI metric. We contrast their work by proposing a pre-generation attention probing mitigation technique (detecting uncertainty at the representation level before token generation).

### 03 — HalluAudio (Zhao et al., arXiv 2604.19300, Apr 2026)
- **Core task:** Introduces a large-scale, human-verified diagnostic benchmark (>5,000 QA pairs) across speech, environmental sound, and music to systematically evaluate LALM hallucinations using adversarial and contrastive prompts.
- **Key findings:** Evaluation of 12 LALMs on 5,720 QA pairs reveals severe vulnerability to hallucinations despite high accuracy on standard tasks. Notably, models like Qwen2-Audio exhibit a strong affirmative bias and a high False Refusal Rate (FRR) — defaulting to overly conservative refusal behavior even when sufficient acoustic evidence is present.
- **What we adopt / contrast:** We adopt their mathematical definition of the False Refusal Rate (FRR), which directly corresponds to our "over-refusal" metric. We contrast by moving beyond diagnostic benchmarking: while HalluAudio measures these failures across general audio, we focus specifically on speech epistemic reasoning and actively *mitigate* the hallucination/over-refusal trade-off using our pre-generation probing method.

### 26 — LISTEN (Kuan & Lee, arXiv 2505.14518, 2025)
- **Core task:** Proposes a contrastive-like training method to mitigate hallucinations in ALLMs by explicitly teaching them to identify which sounds are *absent* from the audio.
- **Key findings:** By training a lightweight modality adapter using synthesized negative samples (descriptions of sounds not present in the audio), the model significantly improves at rejecting absent sound events. It achieves this while using only 3% of the training data required by massive models like Qwen-Audio. Additionally, the authors note that speech-related hallucinations remain largely underexplored.
- **What we adopt / contrast:** LISTEN represents a *training-based* mitigation approach (using adapter tuning and synthetic negative samples). We heavily contrast with this: we tackle the same problem (teaching models what they don't hear) but via a completely *training-free* method (pre-generation attention probing). Furthermore, while they focus on environmental sounds, our dataset explicitly addresses their stated gap regarding speech-content hallucinations.

### 27 — BALSa (Kuan & Lee, arXiv 2505.20166, Jan 2026)
- **Core task:** Proposes BALSa, a highly data-efficient framework that bootstraps audio-language alignment by leveraging a frozen backbone LLM to synthesize general-purpose contrastive training data (including descriptions of absent sounds).
- **Key findings:** Adding synthetic negative samples to the training data improves the weighted F1 score on hallucination detection by approximately 20% compared to positive-only training. The method maintains strong instruction-following capabilities (avoiding catastrophic forgetting) and achieves competitive QA performance using only ~300 hours of audio data, outperforming models trained on millions of hours.
- **What we adopt / contrast:** Like its predecessor *LISTEN* (Article 26), BALSa relies entirely on a *training-based* alignment strategy (tuning a modality adapter with contrastive synthetic data) to mitigate hallucinations and over-generation. We firmly contrast with this paradigm: our work aims to solve the same epistemic reasoning bottleneck (detecting absent evidence) using a purely *training-free* approach via pre-generation attention probing. We bypass the need for synthetic data curation and adapter fine-tuning altogether.

### 25 — Walking Through Uncertainty (Kuan et al., arXiv 2604.25591, Apr 2026)
- **Core task:** Systematically benchmarks five uncertainty estimation methods (predictive entropy, length-normalized entropy, semantic entropy, discrete semantic entropy, and P(True) self-verification) across multiple audio-aware LLMs (Qwen2.5-Omni-3B/7B, Audio Flamingo 3) on both general audio reasoning and trustworthiness-oriented benchmarks.
- **Key findings:** Semantic-level and verification-based uncertainty estimation methods consistently outperform token-level likelihood baselines on general reasoning tasks (MMAU AUROC up to 0.85 vs. 0.64). However, performance on trustworthiness benchmarks (AQUA-Bench and Hallucination) is highly model- and task-dependent. Using uncertainty as a routing signal for adaptive inference (caption-then-reason CoT) reduces token costs by 36–76%, but its benefit is tightly bound to whether the fallback reasoning strategy itself improves direct answer accuracy.
- **What we adopt / contrast:** Kuan et al. focus entirely on *post-generation / sampling-based* uncertainty (computing entropy over $K$ generated output samples or running post-hoc self-verification). We contrast with this by proposing a *pre-generation* attention probing mitigation technique—detecting uncertainty directly at the representation level from internal hidden states before token generation begins, completely bypassing the computational cost and latency of post-hoc sampling.

### 09 — Know Your Limits: Survey of Abstention in LLMs (Wen et al., arXiv 2407.18418, Feb 2025)
- **Core task:** Provides a comprehensive survey of abstention (refusal to answer) in large language models, categorizing literature across the model lifecycle (pretraining, alignment, and inference) and unifying existing work under three core perspectives: query answerability $a(x)$, model knowledge $c(x,y)$, and human value alignment $h(x,y)$.
- **Key findings:** Formalizes mathematical definitions for full and partial abstention $r(x,y)$, categorizes inference-stage mechanisms (including inner-state probing, uncertainty estimation, calibration, and consistency), and standardizes evaluation metrics (such as Abstention Recall/Prudence Score, Abstention Precision, and Over-conservativeness/ARSP). Highlights critical open bottlenecks, notably the over-abstention trade-off caused by alignment tuning and the need to extend abstention to multimodal settings.
- **What we adopt / contrast:** We adopt their unified taxonomy of abstention metrics (mapping Abstention Recall to our Correct Refusal Rate and Over-conservativeness/ARSP to our Over-Refusal Rate) to ground our evaluation framework. We contrast by directly filling the key gap identified in their survey (§6 Future Directions): while their survey focuses almost exclusively on text-only LLMs, our work extends abstention mechanisms to multimodal speech-language models (ALLMs) via pre-generation attention probing.
---

## 2. Metrics Mapping Table (Target for §4 of the Paper)

This table maps our project-specific metrics to classical binary classification and selective prediction terms, treating "Abstention" (Refusal) as the positive class ($P = \text{Abstain}$, $N = \text{Answer}$).

| Project-specific Metric | Project Definition | Binary Classification Term | Selective Prediction Term | Math Formula |
| :--- | :--- | :--- | :--- | :--- |
| **Correct Refusal Rate** | Ratio of Category C (unanswerable) questions where the model correctly abstained | True Positive Rate (TPR) / Sensitivity / Recall on "Refusal" | **Abstention Recall** (Recall of the refusal decision) | $\frac{TP}{TP + FN} = \frac{N_{r\|C}}{N_C}$ |
| **Hallucination Rate** | Ratio of Category C questions where the model erroneously attempted to answer | False Negative Rate (FNR) / Type II Error | **Selective Risk** on unanswerable subset | $\frac{FN}{TP + FN} = \frac{N_{w\|C}}{N_C}$ |
| **Over-Refusal Rate** | Ratio of Category A & B (answerable) questions where the model erroneously abstained | False Positive Rate (FPR) / Fallout / Type I Error | **Rejection Error** / Usability Loss | $\frac{FP}{FP + TN} = \frac{N_{r\|A \cup B}}{N_{A \cup B}}$ |
| **Accuracy on A & B** | Ratio of correct textual responses given on answerable questions | True Negative Rate (TNR) / Specificity (for correctly answered) | **Selective Accuracy** (Accuracy on the covered subset) | $\frac{TN_{correct}}{FP + TN} = \frac{N_{c\|A \cup B}}{N_{A \cup B}}$ |
| **Refusal Precision** | Ratio of correct refusals among all abstention actions taken by the model | Positive Predictive Value (PPV) / Precision on "Refusal" | **Abstention Precision** (Precision of the refusal decision) | $\frac{TP}{TP + FP} = \frac{N_{r\|C}}{N_{r\|C} + N_{r\|A \cup B}}$ |

---

