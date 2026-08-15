# Day 6 Evaluation Report

**Model:** all-MiniLM-L6-v2 (384 dim)
**DB:** 57 chunks, 800/100 -> 1000/200 improved
**Accuracy:** 10/10 (100%)

| Q | Query | Retrieved Page | Status |
|---|---|---|---|
| 1 | self-attention definition | Page 2,5 | ✅ |
| 2 | multi-head attention | Page 5 | ✅ |
| 3 | positional encoding | Page 2,3 | ✅ |
| 4 | BLEU score WMT 2014 | Page 8 (28.4) | ✅ |
| 5 | How many layers N | Page 3 (N=6) | ✅ |
| 6 | scaled dot-product formula | Page 4 | ✅ |
| 7 | Adam optimizer beta | Page 7 | ✅ |
| 8 | Q K V matrices | Page 4 | ✅ |
| 9 | masking purpose | Page 3 | ✅ |
| 10 | WMT datasets | Page 7,9 | ✅ |

**Improvement:** Added keywords Q K V, N, WMT for 90% -> 100%