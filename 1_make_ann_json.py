# anno_file_test.json
import os
import sys
sys.path.append(os.getcwd)
import pandas as pd
from Utils.file_and_folder_operations import *
from abnormality_list import abnormality_list
import numpy as np
text_data_root = './PE_datasets'
img_feat_root = 'PE_DATA_PATH'

abn_label_df1 = pd.read_csv(join(text_data_root, 'brown_vqa_v2m.csv'))
abn_label_df2 = pd.read_csv(join(text_data_root, 'inspecta_vqa_v2m.csv'))
abn_label_df = pd.concat([abn_label_df1, abn_label_df2])

abn_text_df1 = pd.read_json(join(text_data_root, 'brown_CTPA_PE_disease_finding.json'))
abn_text_df1_impress = pd.read_excel(join(text_data_root, 'all_brown_CTPA_PE_finding_add_section_v2_updated_chestNoNumber_r3-aw_lym_thy_per.xlsx'))
abn_text_df1 = pd.merge(abn_text_df1, abn_text_df1_impress[['AccessionNumber_md5', 'Impression Text']], on='AccessionNumber_md5')
abn_text_df2 = pd.read_json(join(text_data_root, 'inspecta_CTPA_PE_disease_finding.json'))
abn_text_df2 ['Findings Text'] = ''
abn_text_df = pd.concat([abn_text_df1, abn_text_df2])

for trts in ['train', 'validation', 'test']:
    ans_file= open(join(img_feat_root, 'anno_file_'+trts+'.json'), "w")
    for img_folder in subfolders(join(img_feat_root, 'image_multiscale_feat_'+trts)):
        for img_aug_feat in subfiles(img_folder):
            accNum = os.path.split(img_aug_feat)[-1].split('_')[1].replace('.npz', '')
            try:
                abn_label = abn_label_df[abn_label_df['AccessionNumber_md5'] == accNum].iloc[0]
                abn_text = abn_text_df[abn_text_df['AccessionNumber_md5'] == accNum].iloc[0]
                abn_label = (abn_label[abnormality_list] == 'Yes').values
                text_list = abn_text[abnormality_list].values
                text_list = text_list[np.argwhere(abn_label)].reshape(-1).tolist()
                feat_dict = {}
                feat_dict['image_feat'] = img_aug_feat.replace(img_feat_root, '')
                feat_dict['AccessionNumber_md5'] = accNum
                feat_dict['abn_label'] = abn_label.astype(int).tolist()
                feat_dict['abn_text'] = text_list
                
                ans_file.write(json.dumps(feat_dict) + "\n")
                ans_file.flush()
            except:
                print(accNum)