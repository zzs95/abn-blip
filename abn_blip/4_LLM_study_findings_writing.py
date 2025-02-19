# conda env llava with new version transformers for llama 3.1
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # debug
import json
import copy
import numpy as np
import pandas as pd
from rewrite_process_llama3 import LLM_pipline
from abnormality_list import abnormality_dict, finding_section_list
from region_writing_prompts import finding_rewrite_prompt

lm3_model_path = ''
model_LLM = LLM_pipline(model_id=lm3_model_path, max_new_tokens=1024, )

results_file = 'test_epochbest_rank0.json'
exp_path = './lavis/output/ABNCLIP/Captioning/20241202153_cls_embed/ep_26/result/' # multi-scale, cls_embed 

results_json = os.path.join(exp_path, results_file)
result_df = pd.read_json(results_json)

text_data_root = './PE_datasets'
abn_text_df1 = pd.read_json(os.path.join(text_data_root, 'brown_CTPA_PE_disease_finding.json'))
abn_text_df1_region = pd.read_excel(os.path.join(text_data_root, 'all_brown_CTPA_PE_finding_add_section_v2_updated_chestNoNumber_r3-aw_lym_thy_per.xlsx'))
abn_text_df1 = pd.merge(abn_text_df1, abn_text_df1_region[['AccessionNumber_md5', 'Impression Text'] + finding_section_list], on='AccessionNumber_md5')
abn_text_df2 = pd.read_json(os.path.join(text_data_root, 'inspecta_CTPA_PE_disease_finding.json'))
abn_text_df2_region = pd.read_excel(os.path.join(text_data_root, 'inspecta_CTPA_region_finding.xlsx'), index_col=0)
abn_text_df2 = pd.merge(abn_text_df2, abn_text_df2_region[['AccessionNumber_md5', ] + finding_section_list], on='AccessionNumber_md5')

abn_text_df2 ['Findings Text'] = abn_text_df2 ['Impression Text']
abn_text_df = pd.concat([abn_text_df1, abn_text_df2])

test_abn_text_df = abn_text_df[abn_text_df['AccessionNumber_md5'].isin(result_df['image_id'])]

ans_file_image = open(os.path.join(exp_path, 'study_findings_noAbnProbs_debug.jsonl'), "w")

for study_df in test_abn_text_df.iterrows():
    study_df = study_df[1]
    img_id = study_df['AccessionNumber_md5']
    gen_text_df = result_df[result_df['image_id'] == img_id]
    img_num = len(gen_text_df)
    pred_captions = [gen_text_df['caption'].iloc[i][0] for i in range(img_num) ]
    pred_probs = [gen_text_df['caption'].iloc[i][1] for i in range(img_num) ]
    organ_text_dict = {}
    abn_num = 0
    for organ in abnormality_dict.keys():
        organ_abn_len = len(abnormality_dict[organ])
        organ_abn_text = []
        for image_caption in pred_captions:
            organ_abn_text += image_caption[abn_num:abn_num+organ_abn_len]
        organ_text_abnormal_list = copy.deepcopy(organ_abn_text)
        for organ_abn_text_i in organ_abn_text:
            if 'no abnormal' in organ_abn_text_i:
                # print(organ_abn_text_i)
                organ_text_abnormal_list.remove(organ_abn_text_i)
        if len(organ_text_abnormal_list) == 0:
            findings_text_rewrite = 'Normal.'
        else:
            qs = 'Descriptions of potential abnormalities in ' + organ + ' are ' + str(organ_text_abnormal_list) + '\n' + finding_rewrite_prompt.replace('<REGION>', organ)
            findings_text_rewrite = model_LLM.forward(qs)
        organ_text_dict[organ+'_pred'] = findings_text_rewrite
        organ_text_dict[organ+'_gt'] = study_df[organ].replace('No findings', 'Normal').replace('No finding', 'Normal').replace('No acute abnormality', 'Normal').replace('No pulmonary emboli are identified.', 'Normal.')
        abn_num += organ_abn_len

    findings_gt = study_df['Findings Text']
    # print('*'*30)
    # print('pred finding report')
    # print(findings_text_rewrite)
    # print('\ngt finding report')
    # print(findings_gt)
    
    out_dict = {
    "AccessionNumber_md5": img_id,
    "findings_image_gt": findings_gt,
    }
    out_dict.update(organ_text_dict)
    
    ans_file_image.write(json.dumps(out_dict) + "\n")    
    ans_file_image.flush()
ans_file_image.close()