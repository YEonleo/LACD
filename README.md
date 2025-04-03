# Context and Layers in Harmony: A Unified Strategy for Mitigating LLM Hallucinations

## Abstract
Large language models, despite their strong performance, often overlook newly provided prompts and produce hallucinated content due to excessive reliance on pre-trained knowledge. We propose LACD, a technique that dynamically selects tokens by analyzing probability distributions across layers. By restoring neglected context, LACD directs more attention to prompt information and suppresses the overshadowing influence of prior knowledge from lower to final layers. On the HotPot QA dataset, LACD delivers up to a 2.2\% improvement over simple context augmentation and performs similarly to or better than existing methods (e.g., DoLa, CAD). These findings indicate that LACD effectively mitigates hallucination while enhancing factual reliability.

## Results

Below is the performance of LACD compared to various baselines and methods on the HotPotQA and SQuAD datasets:

```latex
\begin{table}[ht]
\centering
\caption{Exact Match(EM) and F1-score on two QA datasets (HotPotQA and SQuAD).}
\label{tab:decoding-results}
\begin{tabular}{l|cc|cc}
\hline
\multirow{2}{*}{\textbf{Model}} 
 & \multicolumn{2}{c|}{\textbf{HotPotQA}} 
 & \multicolumn{2}{c}{\textbf{SQuAD}} \\
\cline{2-5}
 & \textbf{EM} & \textbf{F1} & \textbf{EM} & \textbf{F1} \\
\hline

\textbf{Baseline (w/o context)} 
 & 2.23 & 4.33 & 2.69 & 4.62 \\
\textbf{Baseline (w. context)} 
 & 38.84 & 52.91 & 15.61 & 25.31 \\
\cline{1-5}

\textbf{DoLa \citeyearpar{chuang2023dola}} 
 & 38.93 & 56.50 & 16.62 & 28.30 \\
\textbf{CAD \citeyearpar{shi2024trusting}}
 & 39.08 & 55.70 & 30.12 & 45.29 \\
\cline{1-5}

\textbf{LACD (Ours)} 
 & \textbf{41.01} & \textbf{56.84} & \textbf{30.35} & \textbf{46.13} \\
\hline
\end{tabular}
\end{table}
