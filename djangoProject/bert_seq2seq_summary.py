
import torch
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

vocab_path = os.path.join(BASE_DIR, "templates/static/state_dict/roberta_wwm_vocab.txt")

word2idx, keep_tokens = load_chinese_base_vocab(vocab_path, simplfied=True)
model_name = "roberta"  # 选择模型名字
model_path = os.path.join(BASE_DIR, "templates/static/state_dict/roberta_wwm_pytorch_model.bin")
recent_model_path = os.path.join(BASE_DIR, "templates/static/state_dict/model_Policy.bin")
model_save_path = os.path.join(BASE_DIR, "templates/static/state_dict/model_Policy.bin")
batch_size = 16
lr = 1e-5
maxlen = 256

auto_title_model = os.path.join(BASE_DIR, "templates/static/state_dict/model_Policy.bin")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
os.path.join(BASE_DIR, "templates/static/state_dict/model_Policy.bin")
vocab_path = os.path.join(BASE_DIR, "templates/static/state_dict/roberta_wwm_vocab.txt")
# print(vocab_path)
# 加载字典
word2idx, keep_tokens = load_chinese_base_vocab(vocab_path, simplfied=True)
# 定义模型
bert_model = load_bert(word2idx, model_name=model_name)
bert_model.set_device(device)
bert_model.eval()
## 加载训练的模型参数～
bert_model.load_all_params(model_path=auto_title_model, device=device)


"""
def RoBERTa_gov(orignal_text):
    result = orignal_text.split('\n')
    li_temp = []
    for i in range(len(result)):
        li_temp.append(result[i].strip())
    # file_list = ''.join(li_temp).split("。")
    text_ = []
    for i in range(len(li_temp)):
        if len(li_temp[i]) >= 30:
            text_.append(li_temp[i])
    list_ = []
    for text in text_:
        with torch.no_grad():
            list_.append((bert_model.generate(text, beam_size=1)))
    list_ = list_[2:]
    aaa = ''
    for i in range(len(list_)):
        temp = ''.join(list_[i].split(' '))
        # temp = temp.replace('一',"")
        # temp = temp.replace('二',"")
        # temp = temp.replace('三',"")
        # temp = temp.replace('四',"")
        # temp = temp.replace('五',"")
        # temp = temp.replace('六',"")
        # temp = temp.replace('七',"")
        # temp = temp.replace('八',"")
        # temp = temp.replace('九',"")
        # temp = temp.replace('十',"")
        # temp = temp.replace('：',"")
        # temp = temp.replace('北大',"")
        # temp = temp.replace('法宝引证码',"")

        temp = temp + '，\n' if i < len(list_) - 1 else temp + '。'
        aaa += temp
    return aaa

"""

def RoBERTa_news(text):
    # data = ''
    # data_list = []
    # text = text.replace(' ', '')  # 删除所有空格
    # text = text.replace('\n', '')  # 删除所有回车
    # data = data + text
    # data_list.append(data)
    # for text_ in data_list:
    #     with torch.no_grad():
    #         generate = ((bert_model.generate(text_, beam_size=1)))
    # generate = generate.replace(' ', '')
    return text
