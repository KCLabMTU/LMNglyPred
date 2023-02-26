# LMNglyPred
LMNglyPred: Prediction of Human N-Linked Glycosylation Site using embeddings from pre-trained protein language model


Evaluate Model
Programs were executed using anaconda version: 2020.07, recommended to install the same

The programs were developed in the following environment. python : 3.8.3.final.0, python-bits : 64, OS : Linux, OS-release : 5.8.0-38-generic, machine : x86_64, processor : x86_64, pandas : 1.0.5, numpy : 1.18.5, pip : 20.1.1, scipy : 1.4.1, scikit-learn : 0.23.1., keras : 2.4.3, tensorflow : 2.3.1.

Please place the GlycoBiology_NGlyDE_Original.ipynb, Undersampling_Glycobiology_NGLYDE_Final6947757.h5, Independent_Test_Set_Prot_T5_feature_Aug_12.txt, Subash_August_8_2022_NGlyDE_Prot_T5_feature.txt, in the same directory where you will execute the python program, and execute the GlycoBiology_NGlyDE_Original.ipynb program to see the reported result.

Please place the GlycoBiology_NGlyDE_90__Training_10__Indepedent_Testing.ipynb, NGlyDE_Prot_T5_Final.h5, Glycobiology_NGlyDE_Independent_Positive_202_Negative_100.csv, Glycobiology_NGlyDE_Training_Positive_1821_Negative_901.csv, in the same directory where you will execute the python program, and execute the GlycoBiology_NGlyDE_90__Training_10__Indepedent_Testing.ipynb program to see the reported result.

Please place the GlycoBiology_NGlycositeAtlas.ipynb, df_indepenent_test_again_done_that_has_unique_protein_and_unique_sequence.csv, Final_GlycoBiology_ANN_Glycobiology_ER_RSA(GA_Extracell_cellmem)187.h5, df_train_data_without_indepenent_test_and_protein.csv, in the same directory where you will execute the python program, and execute the GlycoBiology_NGlycositeAtlas.ipynb program to see the reported result.

*** For your convenience we have uploaded the ProtT5 feature extraction program (analyze_Cell_Mem_ER_Extrac_Protein.py) for the protein sequence from ProtT5 as well as corresponding 1024 feature vector extraction program (Feature Extraction Program from the generated files.ipynb) from the ProtT5 file. ***

If you need any futher help please contact Dr. Dukka B. KC at dbkc@mtu.edu.
