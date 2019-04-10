import pandas as pd
import numpy as np
import scipy.stats
import math

class MediqaEvaluatorT3:
  def __init__(self, answer_file_path, round=1):
    """
    `round` : Holds the round for which the evaluation is being done. 
    can be 1, 2...upto the number of rounds the challenge has.
    Different rounds will mostly have different ground truth files.
    """
    self.answer_file_path = answer_file_path
    self.round = round

  def _evaluate(self, client_payload, _context={}):
    """
    `client_payload` will be a dict with (atleast) the following keys :
      - submission_file_path : local file path of the submitted file
      - aicrowd_submission_id : A unique id representing the submission
      - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
    """
    submission_file_path = client_payload["submission_file_path"]
    aicrowd_submission_id = client_payload["aicrowd_submission_id"]
    aicrowd_participant_uid = client_payload["aicrowd_participant_id"]
    
    # Result file format: q_id,a_id,label{0/1} 

    col_names = ['question_id','answer_id', 'label']

    submission = pd.read_csv(submission_file_path, header=None, names=col_names)
    gold_truth = pd.read_csv(self.answer_file_path, header=None, names=col_names)
  
   
    submission.label = submission.label.astype(str)
    gold_truth.label = gold_truth.label.astype(str)

    submission['entry'] = submission.apply(lambda x: '_'.join(x), axis=1)
    gold_truth['entry'] = gold_truth.apply(lambda x: '_'.join(x), axis=1)
     
    s1 = submission[submission['entry'].isin(gold_truth['entry'])]

    accuracy = s1.size / gold_truth.size
    print(accuracy)
    
    question_ids = []
    correct_answers = {}
    for index, row in gold_truth.iterrows():
    	qid = row['question_id']
    	
    	if qid not in question_ids:
    		question_ids.append(qid)
    		
    	if row['label'] == '1':
    		if qid not in correct_answers:
    			correct_answers[qid] = []
    		
    		correct_answers[qid].append(row['answer_id'])   		
    print(correct_answers)
    P1 = 0.
    P5 = 0.
    P10 = 0.
    spearman = 0.
    pv = 0.
    ref_sizeAt5 = 0.
    ref_sizeAt10 = 0.
    mrr = 0.
    map = 0.
    count_nan=0
    for qid in question_ids:
        submitted_correct_answers = []
        index = 1

        first = True

        ap_sum = 0.

        for _, row in submission[submission['question_id']==qid].iterrows():
            aid = row['answer_id']
            if row['label'] == '1':
                if first:
                    mrr += 1. / index
                    first=False
            
                if aid in correct_answers[qid]:
                    submitted_correct_answers.append(aid)
                    ap_sum += len(submitted_correct_answers) / index

                if index == 1:
                    P1 += 1
                if index <= 5:
                    P5 += 1
                if index <= 10:
                    P10 += 1

            index += 1
        if(len(submitted_correct_answers)>0):
            map += ap_sum / len(submitted_correct_answers)

        matched_gold_subset = []

        for x in correct_answers[qid]:
            if x in submitted_correct_answers:
                matched_gold_subset.append(x)
        print(submitted_correct_answers,matched_gold_subset)
        print(qid)
        rho, p_value = scipy.stats.spearmanr(submitted_correct_answers, matched_gold_subset)
        
        if not math.isnan(rho):
            spearman += abs(rho)
        else:
            count_nan=count_nan+1
        pv += p_value
        ref_sizeAt5 += min(5, len(correct_answers[qid]))
        ref_sizeAt10 += min(10, len(correct_answers[qid]))


    map = map / len(question_ids)
    question_nb = len(question_ids)
    spearman = spearman / (question_nb-count_nan)
    P1 = P1 / question_nb
    P5 = P5 / ref_sizeAt5
    P10 = P10 / ref_sizeAt10
    mrr = mrr / question_nb
    
    _result_object = {
    	"MAP": map,
    	"MRR": mrr,
        "P@1": P1,
        "P@5": P5,
        "P@10": P10,
        "Accuracy": accuracy,
        "Spearman": spearman
    }

    return _result_object

if __name__ == "__main__":
    answer_file_path = "true_out.csv"
    _client_payload = {}
    _client_payload["submission_file_path"] = "out.csv"
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234
    
    # Instaiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = MediqaEvaluatorT3(answer_file_path)
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
print(result)