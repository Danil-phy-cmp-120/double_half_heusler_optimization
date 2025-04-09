# GNN for Thermoelectric Property Prediction

## 📌 Description

This project implements a **Graph Neural Network (GNN)** to predict thermoelectric properties of materials from crystal structures and compositional data. It is inspired by the work:

- [*Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties* by Tian Xie and Jeffrey C. Grossman](https://doi.org/10.1103/PhysRevLett.120.145301)
- [CGCNN reference implementation and resources](https://alexheilman.com/res/ml/cgcnn.html)

The goal is to accelerate the discovery of efficient thermoelectric materials.

## 🔬 Model

- GNN trained on graphs constructed from crystal structures
- Node features based on elemental properties
- Edge features include bond lengths and coordination
- Target properties may include:
  - Seebeck coefficient
  - Electrical/thermal conductivity (in progress)

## 📚 Dataset

The model is trained on data derived from the following open-access DFT database:

> Zhang, Y. et al.  
> "High-throughput screening of Heusler alloys for spintronics: a database of Heusler compounds."  
> *Scientific Data*, 4, 170085 (2017).  
> [https://www.nature.com/articles/sdata201785](https://www.nature.com/articles/sdata201785)

## ⚙️ Dependencies

Recommended:

```bash
pip install torch torch-geometric pymatgen matminer scikit-learn
```

## 🚀 Usage

Open and run the notebook:

```bash
jupyter notebook GNN_thermoelectrics.ipynb
```

## 📄 License

Distributed under the terms of your chosen license.