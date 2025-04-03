# Context and Layers in Harmony: A Unified Strategy for Mitigating LLM Hallucinations

## Abstract
![overvall](https://github.com/user-attachments/assets/5f831f9d-99e5-419e-a3fb-6cbc2c4837c0)

Large language models, despite their strong performance, often overlook newly provided prompts and produce hallucinated content due to excessive reliance on pre-trained knowledge. We propose LACD, a technique that dynamically selects tokens by analyzing probability distributions across layers. By restoring neglected context, LACD directs more attention to prompt information and suppresses the overshadowing influence of prior knowledge from lower to final layers. On the HotPot QA dataset, LACD delivers up to a 2.2% improvement over simple context augmentation and performs similarly to or better than existing methods (e.g., DoLa, CAD). These findings indicate that LACD effectively mitigates hallucination while enhancing factual reliability.

## Results

Below is the performance of LACD compared to various baselines and methods on the HotPotQA and SQuAD datasets:

| **Model**                                 | **EM (HotPotQA)** | **F1 (HotPotQA)** | **EM (SQuAD)** | **F1 (SQuAD)** |
|-------------------------------------------|-------------------:|------------------:|---------------:|---------------:|
| **Baseline (w/o context)**               | 2.23              | 4.33              | 2.69           | 4.62           |
| **Baseline (w. context)**                | 38.84             | 52.91             | 15.61          | 25.31          |
| **DoLa **  | 38.93             | 56.50             | 16.62          | 28.30          |
| **CAD **  | 39.08             | 55.70             | 30.12          | 45.29          |
| **LACD (Ours)**                          | **41.01**         | **56.84**         | **30.35**      | **46.13**      |

---

## Setup

### Install Dependencies
Install all required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software in accordance with the terms of the license.
