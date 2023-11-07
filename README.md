## LMNglyPred: Prediction of Human N-Linked Glycosylation Site using embeddings from pre-trained protein language model

## Authors

Subash C Pakhrin<sup>1</sup>, Suresh Pokharel<sup>2</sup>, Kiyoko F Aoki-Kinoshita<sup>3</sup>, Moriah R Beck<sup>4</sup>,Tarun K Dam<sup>5</sup>, Doina Caragea<sup>6</sup>, Dukka B KC<sup>2*</sup>
<br><br>
<sup>1</sup>School of Computing, Wichita State University, 1845 Fairmount St., Wichita, KS 67260, USA<br>
Department of Computer Science and Engineering Technology, University of Houston-Downtown, Houston, TX 77002, USA<br>
<sup>2</sup>Department of Computer Science, Michigan Technological University, Houghton, MI, USA
<br>
<sup>3</sup>Glycan and Life Systems Integration Center (GaLSIC), Soka University, Tokyo 192-8577, Japan
<br>
<sup>4</sup>Department of Chemistry and Biochemistry, Wichita State University, 1845 Fairmount St., Wichita, KS 67260, USA
<br>
<sup>5</sup>Laboratory of Mechanistic Glycobiology, Department of Chemistry, Michigan Technological University, Houghton, MI 49931, USA
<br>
<sup>6</sup>Department of Computer Science, Kansas State University, Manhattan, KS 66506, USA
<br><br>
<sup>*</sup> Corresponding Author: dbkc@mtu.edu

## Installation
Clone the repository: `git clone git@github.com:KCLabMTU/LMNglyPred.git` or download `https://github.com/KCLabMTU/LMNglyPred`
### Install Libraries
Python version: `3.9.7`

Install from requirement.txt: 
<code>
pip install -r requirements.txt
</code>

Required libraries and versions: 
<code>
Bio==1.5.2
keras==2.9.0
matplotlib==3.5.1
numpy==1.23.5
pandas==1.5.0
requests==2.27.1
scikit_learn==1.2.0
seaborn==0.11.2
tensorflow==2.9.1
torch==1.11.0
tqdm==4.63.0
transformers==4.18.0
xgboost==1.5.0
</code>

### Install Transformers
<code> pip install -q SentencePiece transformers</code>

### To run `LMNglyPred` model on your own sequences 

In order to predict human N-linked glycosylation site using your own sequence, you need to have two inputs:
1. Copy sequences you want to predict to `input/sequence.fasta`
2. Run `python predict.py`
3. Find results inside `output` folder


### Training, evaluation, and other experiments
1. Find data, code at `training_and_evaluation` folder
2. Follow the `readme.md` file inside the folder


## Citation
Subash C Pakhrin, PhD and others, LMNglyPred: prediction of human N-linked glycosylation sites using embeddings from a pre-trained protein language model, Glycobiology, Volume 33, Issue 5, May 2023, Pages 411–422, https://doi.org/10.1093/glycob/cwad033

Link: https://academic.oup.com/glycob/article-abstract/33/5/411/7126679


## Contact
If you need any further help please contact Dr. Subash Chandra Pakhrin (pakhrins@uhd.edu) or Dr. Dukka B. KC at dbkc@mtu.edu.
