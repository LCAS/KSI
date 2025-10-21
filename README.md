<h1 align="center">
  Keypoint Semantic Integration for Improved Feature Matching in Outdoor Agricultural Environments
</h1>

<h3 align="center">
  Rajitha de Silva · Jacob Swindell · Jonathan Cox · Marija Popović · Cesar Cadena · Cyrill Stachniss · Riccardo Polvara
</h3>

<h4 align="center">
  IEEE Robotics and Automation Letters (RA-L)
</h4>

<h4 align="center">
  <a href="https://arxiv.org/pdf/2503.08843" target="_blank">📄 Paper</a> &nbsp; | &nbsp;
  <a href="https://github.com/LCAS/KSI" target="_blank">💻 Code</a> &nbsp; | &nbsp;
  <a href="https://universe.roboflow.com/gaia-hse8w/semanticblt/dataset/1" target="_blank">🌱 Dataset</a>
</h4>

---

## TL;DR

We present **Keypoint Semantic Integration (KSI)** — a lightweight method that enhances keypoint descriptors with semantic context to reduce **perceptual aliasing** in visually repetitive outdoor environments such as vineyards.  
By embedding instance-level semantic information (e.g., trunks, poles, buildings) into keypoint descriptors, KSI significantly improves **feature matching**, **pose estimation**, and **visual localisation** across months and seasons.  
It integrates seamlessly with classical (SIFT, ORB) and learned (SuperPoint, R2D2, SFD2) descriptors, using existing matchers like SuperGlue or LightGlue without retraining.

---

## Abstract

Robust robot navigation in outdoor environments requires accurate perception systems capable of handling visual challenges such as repetitive structures and changing appearances. Visual feature matching is crucial to vision-based pipelines but remains particularly challenging in natural outdoor settings due to perceptual aliasing. We address this issue in vineyards, where repetitive vine trunks and other natural elements generate ambiguous descriptors that hinder reliable feature matching. We hypothesise that semantic information tied to keypoint positions can alleviate perceptual aliasing by enhancing keypoint descriptor distinctiveness. To this end, we introduce a keypoint semantic integration technique that improves the descriptors in semantically meaningful regions within the image, enabling more accurate differentiation even among visually similar local features. We validate this approach in two vineyard perception tasks: (i) relative pose estimation and (ii) visual localisation. Our method improves matching accuracy across all tested keypoint types and descriptors, demonstrating its effectiveness over multiple months in challenging vineyard conditions.

---

## Method Overview

KSI operates as a **plug-and-play enhancement** over existing keypoint pipelines:

1. **Panoptic Segmentation** – Uses YOLOv9 to segment vineyard-relevant classes (trunks, poles, buildings, etc.) from RGB images.  
2. **Semantic Encoding** – Each instance mask is encoded via a lightweight autoencoder to produce a compact semantic embedding.  
3. **Descriptor Fusion** – The semantic embedding is added to the corresponding keypoint descriptor and L2-normalised.  
4. **Matching** – Enhanced descriptors are matched together using existing matchers such as SuperGlue or LightGlue.

The result is a **semantics-aware matching pipeline** that maintains compatibility with standard SLAM and localisation systems.

---

<p align="center">
  <img src="static/horizontal.png" alt="KSI overview" width="900"><br>
  <em>Figure 1. Overview of the KSI pipeline integrating semantic embeddings into keypoint descriptors.</em>
</p>

<p align="center">
  <img src="static/cm.png" alt="Semantic matching results" width="700"><br>
  <em>Figure 2. KSI enhances descriptor distinctiveness in repetitive vineyard scenes across seasons.</em>
</p>

<p align="center">
  <img src="static/vineyard.png" alt="Vineyard dataset map" width="900"><br>
  <em>Figure 3. Vineyard loop used for evaluation — trunks and buildings provide stable semantics year-round.</em>
</p>

<p align="center">
  <img src="static/forest.png" alt="Woodland generalisation" width="700"><br>
  <em>Figure 4. KSI generalises to woodland environments, improving tree-based feature matching.</em>
</p>

---

## Dataset: **SemanticBLT**

We introduce [**Semantic Bacchus Long-Term (SemanticBLT)**](https://universe.roboflow.com/gaia-hse8w/semanticblt/dataset/1) — a multi-season dataset of vineyard images with **panoptic segmentation** for six classes (buildings, pipes, poles, robots, trunks, vehicles).  
It extends the [Bacchus Long-Term (BLT)](https://doi.org/10.1002/rob.22209) dataset with semantic annotations, enabling perception research in repetitive natural scenes.

---

## Citation

If you use this work, please cite:

```bibtex
@article{de2025keypoint,
  title={Keypoint Semantic Integration for Improved Feature Matching in Outdoor Agricultural Environments},
  author={de Silva, Rajitha and Cox, Jonathan and Popovic, Marija and Cadena, Cesar and Stachniss, Cyrill and Polvara, Riccardo},
  journal={arXiv preprint arXiv:2503.08843},
  year={2025}
}
```
