# Instructions for running embedding clustering 

1. **Create the API Configuration File**

   In the root directory of the project, create a file named `emb_cluster_config.json` with the following content:

   ```json
   {
       "OPENAI_API_KEY": "[YOUR_KEY]"
   }
   ```


2. **add ProblemBodies_23.csv to the follwoing path**
    ```bash
    /clustering/data_subsets/
    ```

3. **Create a raw_data folder inside /clustering and place raw datasets in these locations**

    ```bash
      /clustering/raw_data/assist2009/assist2009/skill_builder_data_corrected_collapsed.csv 
      /clustering/raw_data/assist2012/assist2012/2012-2013-data-with-predictions-4-final.csv
      /clustering/raw_data/assist2017/assist2017/anonymized_full_release_competition_dataset.csv
    ```



4. **run the following c**
    ```bash
    python preprocess.py
    python generate_embeddings.py
    python generate_clustered_datasets.py
    ```



# pyKT

[![Downloads](https://pepy.tech/badge/pykt-toolkit)](https://pepy.tech/project/pykt-toolkit)
[![GitHub Issues](https://img.shields.io/github/issues/pykt-team/pykt-toolkit.svg)](https://github.com/pykt-team/pykt-toolkit/issues)
[![Documentation](https://img.shields.io/website/http/pykt-team.github.io/index.html?down_color=red&down_message=offline&up_message=online)](https://pykt.org/)

pyKT is a python library build upon PyTorch to train deep learning based knowledge tracing models. The library consists of a standardized set of integrated data preprocessing procedures on more than 7 popular datasets across different domains, 5 detailed prediction scenarios, more than 10 frequently compared DLKT approaches for transparent and extensive experiments. More details about pyKT can see our [website](https://pykt.org/) and [docs](https://pykt-toolkit.readthedocs.io/en/latest/quick_start.html).




## Installation
Use the following command to install pyKT:

Create conda envirment.

```
conda create --name=pykt python=3.7.5
source activate pykt
```


```
pip install -U pykt-toolkit -i  https://pypi.python.org/simple 

```

## Hyper parameter tunning results
The hyper parameter tunning results of our experiments about all the DLKT models on various datasets can be found at https://drive.google.com/drive/folders/1MWYXj73Ke3zC6bm3enu1gxQQKAHb37hz?usp=drive_link.

## References
### Projects

1. https://github.com/hcnoh/knowledge-tracing-collection-pytorch 
2. https://github.com/arshadshk/SAKT-pytorch 
3. https://github.com/shalini1194/SAKT 
4. https://github.com/arshadshk/SAINT-pytorch 
5. https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing- 
6. https://github.com/arghosh/AKT 
7. https://github.com/JSLBen/Knowledge-Query-Network-for-Knowledge-Tracing 
8. https://github.com/xiaopengguo/ATKT 
9. https://github.com/jhljx/GKT 
10. https://github.com/THUwangcy/HawkesKT
11. https://github.com/ApexEDM/iekt
12. https://github.com/Badstu/CAKT_othermodels/blob/0c28d870c0d5cf52cc2da79225e372be47b5ea83/SKVMN/model.py
13. https://github.com/bigdata-ustc/EduKTM
14. https://github.com/shalini1194/RKT
15. https://github.com/shshen-closer/DIMKT
16. https://github.com/skewondr/FoLiBi
17. https://github.com/yxonic/DTransformer
18. https://github.com/lilstrawberry/ReKT

### Papers

1. DKT: Deep knowledge tracing 
2. DKT+: Addressing two problems in deep knowledge tracing via prediction-consistent regularization 
3. DKT-Forget: Augmenting knowledge tracing by considering forgetting behavior 
4. KQN: Knowledge query network for knowledge tracing: How knowledge interacts with skills 
5. DKVMN: Dynamic key-value memory networks for knowledge tracing 
6. ATKT: Enhancing Knowledge Tracing via Adversarial Training 
7. GKT: Graph-based knowledge tracing: modeling student proficiency using graph neural network 
8. SAKT: A self-attentive model for knowledge tracing 
9. SAINT: Towards an appropriate query, key, and value computation for knowledge tracing 
10. AKT: Context-aware attentive knowledge tracing 
11. HawkesKT: Temporal Cross-Effects in Knowledge Tracing
12. IEKT: Tracing Knowledge State with Individual Cognition and Acquisition Estimation
13. SKVMN: Knowledge Tracing with Sequential Key-Value Memory Networks
14. LPKT: Learning Process-consistent Knowledge Tracing
15. QIKT: Improving Interpretability of Deep Sequential Knowledge Tracing Models with Question-centric Cognitive Representations
16. RKT: Relation-aware Self-attention for Knowledge Tracing
17. DIMKT: Assessing Student's Dynamic Knowledge State by Exploring the Question Difficulty Effect
18. ATDKT: Enhancing Deep Knowledge Tracing with Auxiliary Tasks
19. simpleKT: A Simple but Tough-to-beat Baseline for Knowledge Tracing
20. SparseKT: Towards Robust Knowledge Tracing Models via K-sparse Attention
21. FoLiBiKT: Forgetting-aware Linear Bias for Attentive Knowledge Tracing
22. DTransformer: Tracing Knowledge Instead of Patterns: Stable Knowledge Tracing with Diagnostic Transformer
23. stableKT: Enhancing Length Generalization for Attention Based Knowledge Tracing Models with Linear Biases
24. extraKT: Extending Context Window of Attention Based Knowledge Tracing Models via Length Extrapolation
25. csKT: Addressing Cold-start Problem in Knowledge Tracing via Kernel Bias and Cone Attention
26. LefoKT: Rethinking and Improving Student Learning and Forgetting Processes for Attention Based Knowledge Tracing Models
27. FlucKT: Cognitive Fluctuations Enhanced Attention Network for Knowledge Tracing
28. UKT: Uncertainty-aware Knowledge Tracing
29. HCGKT: Hierarchical Contrastive Graph Knowledge Tracing with Multi-level Feature Learning
30. RobustKT: Enhancing Knowledge Tracing through Decoupling Cognitive Pattern from Error-Prone Data

## Citation

We now have a [paper](https://arxiv.org/abs/2206.11460?context=cs.CY) you can cite for the our pyKT library:

```bibtex
@inproceedings{liupykt2022,
  title={pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models},
  author={Liu, Zitao and Liu, Qiongqiong and Chen, Jiahao and Huang, Shuyan and Tang, Jiliang and Luo, Weiqi},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```
