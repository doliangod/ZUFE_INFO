import torch
import os
from djangoProject.myTransformers import GPT2LMHeadModel
from djangoProject.myTransformers import BertTokenizer

import torch.nn.functional as F
"""
PAD = '[PAD]'
pad_id = 0
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mydevice = "0,1"
dialogue_model_path = os.path.join(BASE_DIR, 'templates/static/summary_model/')
voca_path = os.path.join(BASE_DIR, 'templates/static/vocabulary/vocab_small.txt')
max_len = 300 # 每个utterance的最大长度,超过指定长度则进行截断
repetition_penalty = 1.0 # 重复惩罚参数，若生成的对话重复性较高，可适当提高该参数
temperature = 1 # 生成的temperature
topk = 8 # 最高k选1
topp = 0 # 最高积累概率

# args = set_interact_args()
# logger = create_logger(args)
# 当用户使用GPU,并且GPU可用时
# args.cuda = torch.cuda.is_available() and not args.no_cuda
# args.cuda = False
device = 'cuda' if torch.cuda.is_available()  else 'cpu'
# logger.info('using device:{}'.format(device))
os.environ["CUDA_VISIBLE_DEVICES"] = mydevice
tokenizer = BertTokenizer(vocab_file=voca_path)
model = GPT2LMHeadModel.from_pretrained(dialogue_model_path)
model.to(device)
model.eval()
"""

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


# def GPT2(text):
#     try:
#         if len(text):
#             text = text[:1000]
#         input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
#         input_ids.extend(tokenizer.encode(text))
#         input_ids.append(tokenizer.sep_token_id)
#         curr_input_tensor = torch.tensor(input_ids).long().to(device)
#
#         generated = []
#         # 最多生成max_len个token
#         for _ in range(max_len):
#             outputs = model(input_ids=curr_input_tensor)
#             next_token_logits = outputs[0][-1, :]
#             # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
#             for id in set(generated):
#                 next_token_logits[id] /= repetition_penalty
#             next_token_logits = next_token_logits / temperature
#             # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
#             next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
#             filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
#             # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
#             next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
#             if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
#                 break
#             generated.append(next_token.item())
#             curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)
#
#         text = tokenizer.convert_ids_to_tokens(generated)
#
#     except KeyboardInterrupt:
#         print("error")
#     return "".join(text)

def GPT2(text):
    strr = 'GPT2的输出结果测试'
    return strr