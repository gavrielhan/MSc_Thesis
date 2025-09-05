# MSc_Thesis

This project is the codebase for my MSc thesis, which investigated **multimodal machine learning approaches for disease risk prediction** using data from the **Human Phenotype Project (HPP)** and external retinal imaging datasets.  

Two main directions were explored:  

- **Temporal Graph Neural Networks (GNNs)**:  
  Patient trajectories were modeled as temporal heterogeneous graphs built from continuous glucose monitoring (CGM), sleep embeddings, and medical history. We tested survival analysis (Cox loss) and link prediction (BCE loss) heads to predict the onset of seven chronic conditions (e.g., diabetes, hypertension, breast cancer). Despite extensive experimentation, predictive performance was limited due to the sparsity and irregularity of the data, highlighting important methodological challenges for future research.  

- **Self-supervised vision transformers (I-JEPA)**:  
  We adapted the Image Joint Embedding Predictive Architecture (I-JEPA), pretrained on ImageNet, to high-resolution retinal images. Using LoRA fine-tuning, we evaluated transfer performance on Messidor and IDRiD, showing that domain adaptation improved alignment with clinically relevant retinal structures, though absolute performance still fell short of state-of-the-art diabetic retinopathy models.  

Overall, the project demonstrates both the promise and limitations of applying advanced machine learning to heterogeneous clinical data. The GNN experiments exposed the challenges of modeling sparse temporal signals, while the imaging experiments showed encouraging adaptation potential of foundation models in ophthalmology. Together, they represent steps toward a **unified multimodal framework for preventive healthcare**.

---

## Key Features
- LoRA fine-tuning of vision transformers on retina datasets  
- Stratified 20% validation split from training set per dataset  
- Best-epoch selection by validation **macro ROC AUC**; corresponding test metrics recorded  
- Optional image preprocessing: black border cropping, autocontrast, equalization  
- Frozen feature KNN evaluation for baseline comparisons  
- Per-epoch lightweight logs and metric tracking  
- Default `batch_size=1` for GPU memory efficiency  

---

## Repository Layout (selected)

- **JEPA/**
  - `ijepa_finetune_crossdatasets.py`: main fine-tune + evaluation entrypoint  
  - `ijepa_retina_pretrain.py`, `ijepa_retina_ft.py`: retina-specific pretraining and fine-tuning flows  
  - `test_retina_pretrain.py`: evaluation helpers  
  - `prepare_retina_dataset.py`: dataset preparation utilities  
  - `ijepa/src/`: core model, training, masking, transforms, and utilities  

- **Graph-based pipeline**
  - `graph_gen_split.py`: temporal heterogeneous graph construction  
  - `mk_graphs.py`: helper for patient/sleep/diagnosis integration  
  - `model.py`, `train.py`, `eval.py`, `link_prediction_head.py`: heterogeneous GNN encoder and predictive heads  

