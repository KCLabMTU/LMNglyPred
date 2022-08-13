from transformers import TFT5EncoderModel, T5Tokenizer
import numpy as np
import re
import gc
import os
import glob
import re
from Bio import SeqIO

basedir = "/home/t326h379/Prot_T5/ubi_prot_file"

os.chdir(basedir)
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
model = TFT5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", from_pt=True)

name=os.environ['FILENAME']

for seq_record in SeqIO.parse(name,"fasta"):
        placeholder = seq_record.id
        seq = str(seq_record.seq)
        length_of_protein = len(seq)
        sequence_of_amino_acid_of_protein = seq
        placeholder = placeholder.split("|")[1]
       
        seq = seq.replace("U", "X")
        seq = seq.replace("Z", "X")
        seq = seq.replace("O", "X")
        seq = seq.replace("B", "X")
        seq = str(' '.join(seq))
                       
        sequences_Example = [seq]
        ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True, return_tensors="tf")
        input_ids = ids['input_ids']
        attention_mask = ids['attention_mask']
        embedding = model(input_ids)
        embedding = np.asarray(embedding.last_hidden_state)
        attention_mask = np.asarray(attention_mask)
        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            features.append(seq_emd)
        features = features[0]
        list_of_lists = features.tolist()
        filename = placeholder+"_Prot_Trans_"+".csv"
        fp = open(filename,"a+")
        for i in range(length_of_protein):
            features = list_of_lists[i]
            features = str(features)
            features = features.strip("[")
            features = features.strip("]")
            fp.write(sequence_of_amino_acid_of_protein[i])
            fp.write(",")
            fp.write(features)
            fp.write("\n")
        fp.close()
