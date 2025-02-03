import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import torch
from fastapi import UploadFile
from typing import List
from io import StringIO

async def readfiles(list_of_files: List[UploadFile]):
    prote_data = trans_data = geno_data = None

    for file in list_of_files:
        file_content = (await file.read()).decode('utf-8')
        string_io = StringIO(file_content)
        
        if 'prote' in file.filename.lower():
            prote_data = pd.read_csv(string_io, sep='\t')
        elif 'trans' in file.filename.lower():
            trans_data = pd.read_csv(string_io, sep='\t')
        elif 'geno' in file.filename.lower():
            geno_data = pd.read_csv(string_io, sep='\t')
        
        await file.seek(0)

    if any(data is None for data in [trans_data, prote_data, geno_data]):
        missing = []
        if trans_data is None: missing.append("transcriptome")
        if prote_data is None: missing.append("proteome")
        if geno_data is None: missing.append("genome")
        raise ValueError(f"Missing required data files: {', '.join(missing)}")

    return (trans_data, prote_data, geno_data)

def preprocess(trans_data,prote_data,geno_data):
  mgt = pd.merge(geno_data, trans_data, on='gene_id', how='inner')
  ma = pd.merge(mgt, prote_data,left_on='gene_name_x', right_on='peptide_target', how='inner')
  columns=['Unnamed: 9','Unnamed: 10','gene_name_y','set_id','catalog_number','lab_id','AGID','stranded_first','stranded_second','unstranded','gene_id','start','end','min_copy_number','max_copy_number','gene_type','unstranded']
  for i in columns:
    try:
      ma = ma.drop(columns=[i])
    except:
      pass
  ma.to_csv('all_data.csv', index=False)
  all_data_cleaned = pd.read_csv('all_data.csv')

  label_encoder = LabelEncoder()
  all_data_cleaned['gene_name'] = label_encoder.fit_transform(all_data_cleaned['gene_name_x'])
  all_data_cleaned['chromosome'] = label_encoder.fit_transform(all_data_cleaned['chromosome'])
  all_data_cleaned['peptide_target'] = label_encoder.fit_transform(all_data_cleaned['peptide_target'])
  all_data_cleaned = all_data_cleaned.drop(columns=['gene_name_x'])
  all_data_cleaned = all_data_cleaned.dropna()

  features = all_data_cleaned[['chromosome','copy_number','tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded', 'peptide_target', 'protein_expression', 'gene_name']]  # or other relevant features
  scaler = StandardScaler()
  features_scaled = scaler.fit_transform(features)

  X_test1 = torch.tensor(features_scaled, dtype=torch.float32)
  return X_test1,all_data_cleaned


def GAT_pre_process_for_testing(all_data_cleaned):
  features = all_data_cleaned[['chromosome','copy_number','tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded', 'peptide_target', 'protein_expression', 'gene_name']].fillna(0)
  correlation_matrix = features.corr()

  threshold = 0.8
  edges = []
  edge_attributes = []

  for i in range(len(correlation_matrix.columns)):
      for j in range(i):
          corr_value = correlation_matrix.iloc[i, j]
          if abs(corr_value) > threshold:
              edges.append([features.columns.get_loc(correlation_matrix.columns[i]),
                            features.columns.get_loc(correlation_matrix.columns[j])])
              edge_attributes.append(abs(corr_value))

  edge_index = np.array(edges).T
  edge_attributes = np.array(edge_attributes)

  return edge_index, edge_attributes