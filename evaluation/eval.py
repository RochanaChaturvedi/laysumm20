import pandas as pd
import re
from rouge_score import rouge_scorer
import csv
import pathlib
import os
from os.path import join
import logging
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
import glob
import spacy
import scipy.stats as stats


# Format system summaries in expected format for LAYSUMM
def format_summ(path,models):
    for model in models:
        for f in os.listdir(path+"system\\"+model+"\\"):
            ref_file = path+"model\\"+f.replace('.TXT',"_LAYSUMM.TXT")
            dest=path+"system\\"+model+"\\"+f.replace('.TXT','')+"_LAYSUMM.TXT"
            with open(path+"system\\"+model+"\\"+f, 'r',encoding="utf8") as original: 
                try:
                    data = original.read()
                except Exception as e:
                    with open(path+"system\\"+model+"\\"+f,"r",encoding='cp1252'):
                        data = original.read()                    
                data=".\n".join(data.split("."))
            with open(ref_file, 'r',encoding="utf8") as original: 
                pre = original.read()
                pre=pre.split("PARAGRAPH")[0]+"PARAGRAPH"+"\n\n"
            with open(dest, 'w',encoding="utf8") as modified:
                modified.write(pre + data)

def impose_max_length(summary_text, max_tokens=150):
    text = summary_text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = re.split(r"\s+", text)
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", x)]
    tokens = tokens[0:min(max_tokens, len(tokens))]
    return " ".join(tokens)

def evaluate(path):
    metrics = ['rouge1', 'rouge2', 'rougeL']

    for model in models:
            
            df=pd.DataFrame()
            df_r=pd.DataFrame()
            input_dir = path
            output_dir = path+"output\\" #sys.argv[2]

            submit_dir = os.path.join(input_dir, 'system\\'+model)
            truth_dir = os.path.join(input_dir, 'model')

            if not os.path.isdir(submit_dir):
                print ("%s doesn't exist" % submit_dir)

            if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
                results = {"paperid":[],"rouge1_f":[], "rouge1_r":[], "rouge2_f":[], "rouge2_r":[], "rougeL_f":[], "rougeL_r":[]}

                summary_search_string = "PARAGRAPH"
                reference_summaries = {}
                submitted_summaries = {}

                print("open ground truth file")
                for each_file in glob.glob(truth_dir+"\\*LAYSUMM.TXT"):
                    try:
                        truth_file = each_file
                        truth = open(truth_file, "rb")
                        paper_id = each_file.split("\\")[-1].split("_")[0]
                        
                        results["paperid"].append(paper_id)
                        all_lines = truth.read().decode('utf-8', errors='ignore')
                        if all_lines!="" and summary_search_string in all_lines:
                            summary = all_lines.split(summary_search_string)[1]
                            reference_summaries[paper_id] = summary
                    except Exception as e:
                        print(e)
                        continue
    wd=path+"\\output\\"
    scores=["rouge1","rouge2","rougeL"]#,"rouge-SUX"]
    for suffix in ["_r","_f"]:
        for score in scores:
            print(score)
            df=pd.DataFrame()
            for model in models:
                try:
                    df_temp = pd.read_csv(wd+model+".csv",index_col=0)
                    df['paperid']=df_temp['paperid']
                    df[model]=df_temp[score+suffix].copy()
                except Exception as e:
                    print(e)
            df.to_csv(wd+score+suffix+".csv")

                print("open submission file")
                for each_file in glob.glob(submit_dir+"\\*LAYSUMM.TXT"):       
                    try:
                        submission_answer  = open(each_file, "rb")
                        paper_id = each_file.split("\\")[-1].split("_")[0]
                        all_lines = submission_answer.read().decode('utf-8', errors='ignore')
                        if all_lines!="" and summary_search_string in all_lines:
                            summary = all_lines.split(summary_search_string)[1]
                            submitted_summaries[paper_id] = impose_max_length(summary)
                    except Exception as e:
                        print(e)
                        continue

                default_score = 0.0
                index=0
                for paper_id in reference_summaries.keys():
                    try:
                        if paper_id in submitted_summaries:
                            reference_summary, submitted_summary = reference_summaries[paper_id], submitted_summaries[paper_id]
                            scores = scorer.score(reference_summary.strip(),submitted_summary.strip())
                            for metric in metrics:
                                results[metric+"_f"].append(scores[metric].fmeasure)
                                results[metric + "_r"].append(scores[metric].recall)
                        else:
                            print("Missing summary for Paper ID, %d", paper_id, 'for model ', model)
                            for metric in metrics:
                                results[metric+"_f"].append(default_score)
                                results[metric + "_r"].append(default_score)
                    except Exception as e:
                        print(e)
                        print("Error for Paper ID %d", paper_id)
                        for metric in metrics:
                            results[metric+"_f"].append(default_score)
                            results[metric + "_r"].append(default_score)
                        continue
                df=pd.DataFrame.from_dict(results, orient='index').transpose()
                df.to_csv(path+"output\\"+model+".csv")

                print("evaluation finished")

#Get Document Length (number of words)
def get_num_word(doc_dr):
    df=pd.DataFrame()
    data_word={}
    for everyfile in (os.listdir(doc_dr)):
        try:
            with open(doc_dr+"\\"+everyfile,"r",encoding="utf8") as everyfile:
                text=everyfile.read()
        except UnicodeDecodeError as ex:
            with open(summ_dr+"\\"+everyfile,"r",encoding='cp1252') as everyfile:
                text=everyfile.read()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        data_word[os.path.basename(everyfile.name)]=len(tokens)
    return data_word

#num sentences
def get_num_sentence(doc_dr):
    data_sent=dict()
    for everyfile in (os.listdir(doc_dr)):
        try:
            with open(doc_dr+"\\"+everyfile,"r",encoding='utf-8') as everyfile:
                text=everyfile.read()
        except UnicodeDecodeError as ex:
            with open(summ_dr+"\\"+everyfile,"r",encoding='cp1252') as everyfile:
                text=everyfile.read()
        sentences=sent_tokenize(text) 
        data_sent[os.path.basename(everyfile.name)]=len(sentences)
    return data_sent


#return normalized count of nouns or pronouns,
#specify pos="PRONOUN" for counting pronouns
def count_nouns(path,pos="NOUN"):
    dct={}
    nlp = spacy.load("en_core_web_sm")
    for f in os.listdir(path):
        with open(path+f, 'r',encoding="utf8") as original:
            doc = nlp(original.read())
            count=0
            total=0
            for token in doc:
                total=total+1
                if pos=="NOUN":
                    if(token.pos_=="NOUN" or token.pos_=="PROPN"):
                        count=count+1
                elif pos=="PRONOUN":
                    if(token.pos_=="PRON"):
                        count=count+1
            dct[f]=count/total
    return dct

def ComputeChiSquareGOF(gold, system):
    """
    Runs a chi-square goodness-of-fit test and returns the p-value.
    Inputs:
    - expected: numpy array of expected values.
    - observed: numpy array of observed values.
    Returns: p-value
    """
    result = stats.chisquare(f_obs=system, f_exp=gold)
    print("distributions gold vs system = ","different" if p_value < 0.05 else "same")
    return result[1]

