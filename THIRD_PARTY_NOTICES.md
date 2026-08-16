# Third-Party Notices

This project uses the AGILE (AI-Guided Ionizable Lipid Engineering) dataset —
1,200 ionizable lipid structures (SMILES) with measured HeLa transfection 
efficiency. No code from the AGILE repository is used here; this project's 
pipeline (`data_processing.py`, `train.py`, `evaluate.py`, `predict.py`) 
is independent, original work by the author. The dataset's source repository 
is released under the MIT license, reproduced below.

**A note on scope:** MIT is a software license that grants permission to use, 
modify, and redistribute *code*. Whether it was intended by the AGILE authors 
to also cover the dataset files bundled in that repository is not stated 
explicitly in the license text itself. Treating the dataset as MIT-licensed 
here is a reasonable reading given it ships in an MIT-licensed repo with no 
separate data license attached.

---

## AGILE

Source: https://github.com/bowang-lab/AGILE

MIT License

Copyright (c) 2023 Haotian Cui, Shihao Ma, WangLab @ U of T

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

**Note on the trained model artifact (`models/xgb_transfection.pkl`):** this
model was trained on the AGILE dataset (1,200 ionizable lipids, HeLa
transfection readouts). It is a derivative work of that data, though not of
any AGILE code.
