This repository contains the implementation of the experiments from the paper
**"Generating, Reconstructing, and Representing Discrete and Continuous Data:
Generalized Encoding-Decoding Diffusion Probabilistic Models"** (published at
ICML 2024 as *"Unified Generation, Reconstruction, and Representation: Generalized
Diffusion with Adaptive Latent Encoding-Decoding"*), introducing **EDDPMs**
(Encoding-Decoding Diffusion Probabilistic Models) — a generalization of standard
diffusion models that adds parameterized encoding-decoding, unifying generation,
reconstruction, and representation learning across discrete and continuous data
(text, protein sequences, and images).
 
## 📄 Paper
- [ICML 2024 Proceedings (PMLR)](https://proceedings.mlr.press/v235/liu24bh.html)

## 📁 Repository Structure
 
```
.
├── Image/     # Experiments on continuous (image) data
├── Protein/   # Experiments on protein sequence data
├── Text/      # Experiments on discrete (text) data
└── asset/     # Supporting assets
```
 
Please refer to the README inside each folder for setup and usage details specific
to that modality.
 
## 📚 Citation
 
If you find this work useful, please cite:
 
```bibtex
@InProceedings{pmlr-v235-liu24bh,
  title     = {Unified Generation, Reconstruction, and Representation: Generalized Diffusion with Adaptive Latent Encoding-Decoding},
  author    = {Liu, Guangyi and Wang, Yu and Feng, Zeyu and Wu, Qiyu and Tang, Liping and Gao, Yuan and Li, Zhen and Cui, Shuguang and Mcauley, Julian and Yang, Zichao and Xing, Eric P. and Hu, Zhiting},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages     = {31964--31993},
  year      = {2024},
  volume    = {235},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v235/liu24bh.html}
}
```
