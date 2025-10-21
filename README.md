
## GKGL-PE: a GNN-based Knowledge Graph Learning framework for Pathway Embedding

This repository contains the code for our paper, "GKGL-PE: A GNN-based Knowledge Graph Learning Framework for Pathway Embedding," published at the International Conference on Intelligent Biology and Medicine (ICIBM) 2024, held from October 10-12, 2024 in Houston, Texas, United States.

![Alt text](images/gkgl-pe_framework.png)

## Data resources
The different dataset and KG used in this project are located in data directory. These files include:

-) The data about pathways from https://reactome.org/download/current/ReactomePathways.txt, relationships between pathways from https://reactome.org/download/current/ReactomePathwaysRelation.txt and pathway-protein relations from https://reactome.org/download/current/NCBI2Reactome.txt on 24 March 2024.

-) The built knowledge graph including pathway-pathway and pathway-protein relationships.

## Setup

-) conda create -n gnn python=3.11 -y

-) conda activate gnn 

-) conda install pytorch::pytorch torchvision torchaudio -c pytorch

-) pip install pandas

-) pip install py2neo pandas matplotlib scikit-learn

-) pip install tqdm

-) conda install -c dglteam dgl

-) pip install seaborn

## Get start
## creating embedding
python GKGL-PE/embedding_clustering/gat_embedding.py --in_feats 128 --out_feats 128 --num_layers 4 --num_heads 1 --batch_size 1 --lr 0.01 --num_epochs 200
## link prediction
python GKGL-PE/embedding_clustering/main.py --out-feats 128 --num-heads 4 --num-layers 6 --lr 0.02 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 200

gene_pathway_embedding % python embedding/gat_embedding.py --in_feats 2 --hidden_feats 2 --out_feats 128 --num_layers 2  --lr 0.0001 --num_epochs 2002
✅ Created network 'emb_train' with 6193 nodes and 2278939 edges
Number of positives in network: 3251======================================++++++++++++++
✅ Created network 'emb_test' with 2655 nodes and 440119 edges
✅ Saved train/test DGL graphs to embedding/data/emb/raw
Checking training graphs:==========================================================================================
Graph 0 (unknown) has 3251 positives out of 6193 nodes
Checking test graphs:
Graph 0 (unknown) has 1388 positives out of 2655 nodes
nx_graph.graph_nx.nodes===================== 6193
cluster_labels_initial===================== 6193
first_node_stId_in_cluster_initial-------------------------------
 {19: 'ACIN1', 15: 'AKT1', 2: 'AKT2', 16: 'AKT3', 1: 'CARD8', 0: 'CYCS', 11: 'DYNLL2', 3: 'MAPK1', 12: 'PPP3R1', 18: 'TP53BP2', 4: 'XIAP', 17: 'ADRM1', 5: 'CASP6', 8: 'DSG1', 7: 'DSG3', 21: 'MAGED1', 24: 'PLEC', 14: 'PSMC3', 13: 'PSMC5', 10: 'TICAM1', 20: 'PF4', 6: 'VWF', 9: 'ATP1B2', 23: 'ITGAX', 22: 'NOS1'}

