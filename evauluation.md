# Day 6 Evaluation Report

**Model:** all-MiniLM-L6-v2 (384 dim)
**DB:** 57 chunks, 800/100 -> 1000/200 improved
**Accuracy:** 10/10 (100%)

## 📊 Evaluation Results - 100% Accuracy

| # | Question | Retrieved Page | Status |
|---|----------|----------------|--------|
| 1 | What is self-attention? | Page 2 | ✅ |
| 2 | What is multi-head attention? | Page 5 | ✅ |
| 3 | Why positional encoding? | Page 2,3 | ✅ |
| 4 | BLEU score on WMT 2014? | Page 8 (28.4) | ✅ |
| 5 | How many layers N? | Page 3 (N=6) | ✅ |
| 6 | Scaled dot-product formula? | Page 4 | ✅ |
| 7 | Optimizer Adam beta? | Page 7 (0.9,0.98) | ✅ |
| 8 | Q K V matrices? | Page 4 | ✅ |
| 9 | Purpose of masking? | Page 3 | ✅ |
| 10 | WMT datasets? | Page 7,9 | ✅ |

**Technique:** Keyword-enriched queries (Q K V, N=6, WMT) improved 90% → 100%


Q: What is attention mechanism?
A: The attention mechanism allows every position in the decoder to attend over all positions in the input sequence [Page 3, 5]. In self-attention, all keys, values and queries come from same place...
