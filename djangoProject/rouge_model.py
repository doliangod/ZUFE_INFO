from rouge import Rouge

def evaluate_ROUGE(predict, reference):
    """
    Input: str格式 predict和reference分别为预测和真实摘要
    Output: Rouge-Recall
    """
    rouge = Rouge()
    predict = ' '.join(list(predict))
    reference = ' '.join(list(reference))
    scores = rouge.get_scores(predict, reference)

    rouge_1 = scores[0]['rouge-1']['r']
    rouge_2 = scores[0]['rouge-2']['r']
    rouge_L = scores[0]['rouge-l']['r']

    return rouge_1, rouge_2, rouge_L
