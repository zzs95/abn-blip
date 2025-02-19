finding_rewrite_prompt = """
You are given a list of abnormality specific descriptions of <REGION> region in a medical image. 
Please generalize them to a findings conclusion with concise, professional and medical terminology to describe the findings of the <REGION> region. 
Appropriately modify the expression to succinctly summarize the descriptions of similar diseases, the same organs, and the same tissues. Delete the same meaning or the same disease and repeated descriptions. Reduce the contradictory descriptions between abnormalities. Eliminate ambiguity caused by grammatical errors and repetitions.
Do NOT state the abnormality one by one, particularly a normal description such as "No abnormal xxxx are observed". 
Output "Normal." as a conclusion if there is not any abnormality in this region. Otherwise, please output a sentence summary with a conclusion of the observed abnormalities. 
Do NOT output prompt text (such as 'Here is the rewritten FINDINGS section:'). Only output findings content. Output plain text.
"""

# Appropriately modify the expression to succinctly summarize the descriptions of similar diseases, the same organs, and the same tissues. Delete the same meaning or the same disease and repeated descriptions. Reduce the contradictory descriptions between abnormalities. Eliminate ambiguity caused by grammatical errors and repetitions.
# For each "finding part", only keep the abnormal description belonging to this area, and delete the finding text belonging to other areas.  