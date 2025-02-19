# anno_file_test.json
import os
import sys
sys.path.append(os.getcwd)
import pandas as pd
from Utils.file_and_folder_operations import *
from abnormality_list import abnormality_dict, finding_section_list
import numpy as np
text_data_root = './PE_datasets'
img_feat_root = 'PE_DATA_PATH'


text_data_root = './PE_datasets'
abn_text_df1 = pd.read_json(os.path.join(text_data_root, 'brown_CTPA_PE_disease_finding.json'))
abn_text_df1_region = pd.read_excel(os.path.join(text_data_root, 'all_brown_CTPA_PE_finding_add_section_v2_updated_chestNoNumber_r3-aw_lym_thy_per.xlsx'))
abn_text_df1 = pd.merge(abn_text_df1, abn_text_df1_region[['AccessionNumber_md5', 'Impression Text'] + finding_section_list], on='AccessionNumber_md5')
abn_text_df2 = pd.read_json(os.path.join(text_data_root, 'inspecta_CTPA_PE_disease_finding.json'))
abn_text_df2_region = pd.read_excel(os.path.join(text_data_root, 'inspecta_CTPA_region_finding.xlsx'), index_col=0)
abn_text_df2 = pd.merge(abn_text_df2, abn_text_df2_region[['AccessionNumber_md5', ] + finding_section_list], on='AccessionNumber_md5')

abn_text_df2 ['Findings Text'] = abn_text_df2 ['Impression Text']
abn_text_df = pd.concat([abn_text_df1, abn_text_df2])

abn_text_df['study_text_gt'] = 'FINDINGS:\n'
for region_name in finding_section_list:
    abn_text_df['study_text_gt'] += region_name + ': '
    abn_text_df['study_text_gt'] += abn_text_df[region_name].replace('No findings', 'Normal').replace('No finding', 'Normal').replace('No acute abnormality', 'Normal').replace('No pulmonary emboli are identified.', 'Normal.')
    abn_text_df['study_text_gt'] += '\n'
    
abn_text_df.to_excel(os.path.join(text_data_root, 'brown_inspecta_CTPA_region_finding_gt.xlsx'))