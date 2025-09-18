Project Title

TCN-5mC: A Predictor of 5-methylcytosine Sites Based on Multi-Feature Fusion and TCN Networks

   Project Introduction
   
   TCN-5mC is a novel deep learning framework designed for accurate and efficient prediction of 5-methylcytosine (5mC) sites in DNA promoter regions. 5mC is a key epigenetic modification involved in gene regulation, genome stability, and disease progression (e.g., cancer, Alzheimer’s disease). Traditional experimental methods (e.g., bisulfite sequencing) are costly and time-consuming, making computational predictors essential for large-scale studies.
   
   This project integrates:
   
   Multi-feature fusion: Combines One-hot encoding and Nucleotide Chemical Property (NCP) encoding to capture comprehensive sequence information.
   
   Hybrid deep learning architecture: Temporal Convolutional Network (TCN) for long-range dependency capture + Bidirectional Gated Recurrent Unit (BiGRU) for sequential pattern learning + improved Convolutional Block Attention Module (CBAM) for feature refinement.
   
   Imbalanced data handling: Uses SMOTE (Synthetic Minority Oversampling Technique) and Focal Loss to address the natural imbalance of 5mC datasets.

Environment Requirements：

numpy==1.24.3

python==3.6

matplotlib==3.7.1

tensorflow==2.12.0

keras==2.12.0

keras-tcn==3.2.2  

imbalanced-learn==0.10.1  

scikit-learn==1.2.2

Data Preparation:

Prepare your data as FASTA files (no 'N' bases allowed) with the following naming convention:

data/

├── train_positive_data.fasta  # Training set: Positive samples

├── train_negative_data.fasta  # Training set: Negative samples

├── test_positive_data.fasta   # Test set: Positive samples

└── test_negative_data.fasta   # Test set: Negative samples
