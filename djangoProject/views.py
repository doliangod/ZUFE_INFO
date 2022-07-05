from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import json
import os
import time
from djangoProject.summary_textRank import main
from djangoProject.bert_seq2seq_summary import RoBERTa_news
from t5.t5_summary import t5
from djangoProject.interact import GPT2
from PGN.predict import PGN_pre, PGN_pre_gov

# TODO: 引入model_name.py，其中model_name为你自己的任务+姓名，例如News_TextClassification_StudentName.py
from djangoProject.sklearn import model_dest


# from djangoProject.transformer_news.summarize import api_eval
# from djangoProject.rouge_model import evaluate_ROUGE

'''
项目目录
'''
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
num_progress = 0
"""
各个模型名称
"""
all_model_name = [
    'RBERTPa',
    'HPGNs',
    'LM-UML',
    'T-TransferL',
    'TranSum',
    # TODO: ‘model_name’,  更改model_name
    'TextRank',
    'sklearn',
]

"""
返回主页
@:param
"""
def index(request):
    model_infos = all_model_name
    return render(request, "index.html",  {'model_infos': model_infos})
"""
返回模型介绍
@:param model_name 模型名称
"""
def article(request, model_name):
    # path = os.path.join(BASE_DIR, "templates/static/modelContext/"+model_name+".txt")
    # with open(path, 'r', encoding="utf-8") as f:
    #     content = f.read()
    return render(request, "model_introduction/"+model_name+".html")
"""
返回模型介绍主页
@:param
"""
def layouts(request):
    model_infos = all_model_name
    return render(request, "layouts.html", {'model_infos': model_infos})
"""
返回模型应用界面
@:param model_name 模型名称
"""
def model(request, model_name):
    data = {
        'model_name': model_name,
    }
    return render(request, "model.html", data)

"""
返回多文本上传界面
@:param
"""
def mulPage(request):
    model_infos = all_model_name
    return render(request, "agile_board.html", {'model_infos': model_infos})

"""
回退模型介绍首页
@:param
"""
def back_index(request):
    return render(request, "layouts.html")

"""
返回各个模型上传界面，单文本上传
@:param model_name 模型名称
"""
def mulUp(request, model_name):
    data = {
        'model_name': model_name,
    }
    return render(request, "mulUpload.html", data)

"""
返回进度条进程
@:param
"""
def getProgress(request):
    global num_progress
    return JsonResponse(num_progress, safe=False)

"""
返回数据集介绍界面
@:param datasetName 数据集名称
"""
def introduceDataset(request, datasetName):
    if datasetName == "news":
        path = "introduction/新闻数据集.html"
    elif datasetName == "gov":
        path = "introduction/政府数据集.html"
    else:
        path = "introduction/专利数据集.html"
    return render(request, path)

'''
通过粘贴文本实现摘要生成 ajax请求（POST）
@:param model_name 模型名称
@:return 摘要内容
'''
def output(request, model_name):
    if request.method == "POST":
        text = request.POST.get("original")
        if model_name == "TextRank":
            summarization = main(text)
        elif model_name == "RBERTPa":
            # summarization = RoBERTa_news(text)
            summarization = RoBERTa_news(text)
        # elif model_name == "RBERTPa_gov":
        #     summarization = RoBERTa_gov(text)

        elif model_name == "LM-UML":
            summarization = GPT2(text)
        elif model_name == "T-TransferL":
            summarization = t5(text)
        elif model_name == "HPGNs":
            summarization = PGN_pre(text)
        elif model_name == "HPGNs_gov":
            summarization = PGN_pre_gov(text)
        # elif model_name == "TranSum":
        #     summarization = api_eval(text)
        elif model_name == 'sklearn':
            summarization = model_dest(text)

# TODO:

        else:
            summarization = "模型未添加"
        if summarization == None or summarization == "":
            summarization = "-"
        # rouge_1, rouge_2, rouge_L = evaluate_ROUGE(text, summarization)

        data = {
            'abstract_model': summarization,
            # 'rouge_1': format(rouge_1, '.2f'),
            # 'rouge_2': format(rouge_2, '.2f'),
            # 'rouge_L': format(rouge_L, '.2f')
        }
        return HttpResponse(json.dumps(data))
    return render(request, "model.html")
'''
通过文本上传实现摘要生成 ajax请求（POST）
@:param model_name 模型名称
@:return 摘要内容
'''
def upload(request, model_name):
    if request.method == "POST":
        file = request.FILES.get("file")
        text = file.read().decode("utf-8")
        if model_name == "TextRank":
            summarization = main(text)
        elif model_name == "RBERTPa":
            summarization = RoBERTa_news(text)
        # elif model_name == "RBERTPa_gov":
        #     summarization = RoBERTa_gov(text)
        elif model_name == "LM-UML":
            summarization = GPT2(text)
        elif model_name == "T-TransferL":
            summarization = t5(text)
        elif model_name == "HPGNs":
            summarization = PGN_pre(text)
        elif model_name == "HPGNs_gov":
            summarization = PGN_pre_gov(text)
        # elif model_name == "TranSum":
        #     summarization = api_eval(text)
        else:
            summarization = "模型未添加"

        if summarization == None or summarization == "":
            summarization = "-"
        # rouge_1, rouge_2, rouge_L = evaluate_ROUGE(text, summarization)

        data = {
            'original_text': text,
            'abstract_model': summarization,
            # 'rouge_1': format(rouge_1, '.2f'),
            # 'rouge_2': format(rouge_2, '.2f'),
            # 'rouge_L': format(rouge_L, '.2f')
        }
        return HttpResponse(json.dumps(data))
    return render(request, "model.html")
#
'''
通过文本粘贴实现多个模型的摘要生成 ajax请求（POST）
@:param model_name 模型名称
@:return 各个模型摘要内容（json）
'''
def mulOutput(request):
    global num_progress
    tmp_progress = 0
    all_len = len(all_model_name) - 1
    if request.method == "POST":
        text = request.POST.get("original")
        # textRank的算法结果
        summarization_textRank = main(text)
        if summarization_textRank == None or summarization_textRank == "":
            summarization_textRank = "-"
        # rouge1_textRank,rouge2_textRank, rougeL_textRank = evaluate_ROUGE(text, summarization_textRank)
        tmp_progress += 1
        num_progress = tmp_progress/all_len*100
        # RoBERTa的算法结果
        summarization_RoBERTa = RoBERTa_news(text)
        if summarization_RoBERTa == None or summarization_RoBERTa == "":
            summarization_RoBERTa = "-"
        # rouge1_RoBERTa,rouge2_RoBERTa, rougeL_RoBERTa = evaluate_ROUGE(text, summarization_RoBERTa)
        tmp_progress += 1
        num_progress = tmp_progress / all_len * 100
        # GPT2算法结果
        summarization_GPT2 = GPT2(text)
        if summarization_GPT2 == None or summarization_GPT2 == "":
            summarization_GPT2 = "-"
        # rouge1_GPT2, rouge2_GPT2, rougeL_GPT2 = evaluate_ROUGE(text, summarization_GPT2)
        tmp_progress += 1
        num_progress = tmp_progress / all_len * 100
        # T5算法结果
        summarization_T5 = t5(text)
        if summarization_T5 == None or summarization_T5 == "":
            summarization_T5 = "-"
        # rouge1_T5, rouge2_T5, rougeL_T5 = evaluate_ROUGE(text, summarization_T5)
        tmp_progress += 1
        num_progress = tmp_progress / all_len * 100
        # PGN算法结果
        summarization_PGN = PGN_pre(text)
        if summarization_PGN == None or summarization_PGN == "":
            summarization_PGN = "-"
        # rouge1_PGN, rouge2_PGN, rougeL_PGN = evaluate_ROUGE(text, summarization_PGN)
        tmp_progress += 1
        num_progress = tmp_progress / all_len * 100
        # TransformerSum
        # summarization_TransformerSum = api_eval(text)
        # if summarization_TransformerSum == None or summarization_TransformerSum == "":
        #     summarization_TransformerSum = "-"
        # rouge1_TransformerSum, rouge2_TransformerSum, rougeL_TransformerSum = evaluate_ROUGE(text, summarization_TransformerSum)
        # tmp_progress += 1
        # num_progress = tmp_progress / len(all_model_name) * 100
        time.sleep(1)
        data = {
            'TextRank': summarization_textRank,
            'RBERTPa': summarization_RoBERTa,
            'LM-UML': summarization_GPT2,
            'T-TransferL': summarization_T5,
            'HPGNs': summarization_PGN,
            # 'TranSum': summarization_TransformerSum
        }
        return HttpResponse(json.dumps(data))
    return render(request, "agile_board.html")

'''
多个文本上传实现摘要生成 表单 POST
@:param model_name 模型名称
@:return 多个文本相应的摘要内容（json）
'''
def mulUpload(request, model_name):
    if request.method == "POST":
        files = request.FILES.getlist("files")
        texts = []
        if model_name == "TextRank":
            model = main
        elif model_name == "RBERTPa":
            model = RoBERTa_news
        # elif model_name == "RBERTPa_gov":
        #     model = RoBERTa_gov
        elif model_name == "LM-UML":
            model = GPT2
        elif model_name == "T-TransferL":
            model = t5
        # elif model_name == "TranSum":
        #     model = api_eval
        elif model_name == "HPGNs":
            model = PGN_pre
        elif model_name == "HPGNs_gov":
            model = PGN_pre_gov
        else:
            model = lambda x: "模型未添加"
        for file in files:
            text = file.read().decode("utf-8")
            print(file.name)
            summarization = model(text)
            texts.append({
                'name': file.name,
                'text': text,
                'summarization': summarization
            })
        return render(request, "displayTexts.html", {"texts": texts})
    return render(request, "mulUpload.html")
