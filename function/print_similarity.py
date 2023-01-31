import re
import pandas as pd
from tabulate import tabulate

def find_word(text, word):
   result = re.findall('\\b'+word+'\\b', text, flags=re.IGNORECASE)
   if len(result)>0:
      return True
   else:
      return False


def print_similarity(keyword_0, keyword_1, dist_0, dist_1, df_0):
    result = {
        "Keyword": keyword_0,
        "Score": dist_0.cpu().numpy(),
        "Acc." : [],
        "Bias" : []
    }

    match_dict = {}
    for keyword, diff in zip(keyword_1, dist_1):
        match_dict[keyword] = diff.item()

    for keyword, diff in zip(keyword_0, dist_0):
        biased_index =  df_0['caption'].apply(find_word, word=keyword) 
        biased_dataset = df_0[biased_index]  
        biased = biased_dataset.shape[0]
        correct_of_biased = sum(biased_dataset['actual'] == biased_dataset['pred'])
        # correct_of_biased = sum(biased_dataset['correct'])
        biased_accuracy = correct_of_biased / biased
        result["Acc."].append(biased_accuracy)
        if diff < 0:
            result["Bias"].append("")
            continue
        if keyword in match_dict.keys() and match_dict[keyword] > 0:
            result["Bias"].append("M")
        else:
            result["Bias"].append("S")

    diff = pd.DataFrame(result)
    diff = diff.sort_values(by = ["Score"], ascending = False)
    print(tabulate(diff, headers='keys', showindex=False))
    return diff