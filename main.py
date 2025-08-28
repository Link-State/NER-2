# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install boto3
# pip install matplotlib
# pip install --upgrade transformers>=4.5

# PYTHON 3.12.6
# PYTORCH 2.5.1

import os
import sys
import re
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler
from transformers import AutoModel

import etri_tokenizer.file_utils
from etri_tokenizer.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from etri_tokenizer.eojeol_tokenization import eojeol_BertTokenizer


LABEL = ["[PAD]", "B-DT", "I-DT", "O", "B-LC",
         "I-LC", "B-OG", "I-OG", "B-PS", "I-PS",
         "B-TI", "I-TI", "X", "[CLS]", "[SEP]"]
LABEL_MAP = dict()
for (i, label) in enumerate(LABEL) :
    LABEL_MAP[label] = i
ENTITY_LABEL = [i for i in range(len(LABEL)) if len(LABEL[i])>=2 and LABEL[i][0]=="B" and LABEL[i][1]=="-"]
NUM_CLASS = len(LABEL)
MSL = 128
LIFE = 15
HIDDEN_SIZE = 768
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = eojeol_BertTokenizer.from_pretrained("./003_bert_eojeol_pytorch/vocab.korean.rawtext.list", do_lower_case=False)


class BERT_archi(nn.Module) :
    def __init__(self, srcbert="./003_bert_eojeol_pytorch") :
        super(BERT_archi, self).__init__()
        bert = AutoModel.from_pretrained(srcbert)
        self.bert = bert
        self.linear1 = nn.Linear(HIDDEN_SIZE, 512, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, 256, bias=True)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(256, NUM_CLASS)
        return
    
    def forward(self, X, mask) :
        outputs = self.bert(input_ids=X, attention_mask=mask)
        x2 = outputs.last_hidden_state
        x3 = self.linear1(x2)
        x4 = self.relu(x3)
        x5 = self.linear2(x4)
        x6 = self.relu(x5)
        x7 = self.linear3(x6)
        return x7


# 로그 출력
def log(msg="", path="./log.txt", output=True) :
    log_file = open(path, "a", encoding="utf-8")
    log_file.write(str(msg) + "\n")
    log_file.close()
    if output :
        print(msg)
    return


# 문장 분석
def eojeol() :
    if not os.path.exists("./NER_tagged_corpus_ETRI_exobrain_team.txt") :
        log("NER_tagged_corpus_ETRI_exobrain_team.txt파일이 같은 디렉토리 내 존재하지 않습니다.")
        return False

    fp = open("./NER_tagged_corpus_ETRI_exobrain_team.txt", "r", encoding="utf-8")

    regex = re.compile(r'<[^:<>]*:[^:<>]*>')

    for line in fp :
        _line = line.strip()
        index = dict()
        iterator = regex.finditer(_line)
        for match in iterator :
            tag = match.group()[1:-1].split(":")
            index[match.start()] = [match.end(), tag[0], tag[1]]
        
        start = 0
        end = 0
        eojeol_label = open("./ner_eojeol_label_per_line.txt", "a", encoding="utf-8")
        while end < len(_line) :
            if _line[end] == " " :
                word = _line[start:end].strip()
                if word != "" :
                    eojeol_label.write(word + "\t" + "O\n")
                start = end + 1
                end = start
            
            if end in index :
                if start < end :
                    word = _line[start:end].strip()
                    if word != "" :
                        eojeol_label.write(word + "\t" + "O\n")
                
                splited_tag = index[end][1].split()
                eojeol_label.write(splited_tag[0].strip() + "\t" + "B-" + index[end][2] + "\n")
                for i in range(1, len(splited_tag)) :
                    eojeol_label.write(splited_tag[i].strip() + "\t" + "I-" + index[end][2] + "\n")
                
                start = index[end][0] + 1
                end = start
            else :
                end += 1
                if end == len(_line) and end >= start :
                    word = _line[start:].strip()
                    if word != "" :
                        eojeol_label.write(word + "\t" + "O\n")
        eojeol_label.write("\n")
        eojeol_label.close()    
    fp.close()

    return True


# 토큰화
def tokenizer() :
    fp = open("./ner_eojeol_label_per_line.txt", "r", encoding="utf-8")

    while True :
        e_sent = fp.readline()
        if e_sent == '' :
            break
        if len(e_sent) < 2 :
            label_str_file = open("./ner_token_label_per_line.txt", "a", encoding="utf-8")
            label_str_file.write("\n")
            label_str_file.close()
            continue
        
        splited_e_sent = e_sent.split()
        if len(splited_e_sent) < 2 :
            continue

        token = TOKENIZER.tokenize(splited_e_sent[0])
        labeling(tokens=token, tag=splited_e_sent[1])
    fp.close()

    return


# 레이블링
def labeling(tokens=list(), tag="") :
    label_str_file = open("./ner_token_label_per_line.txt", "a", encoding="utf-8")

    label_str_file.write(tokens[0] + "\t" + tag + "\n")
    for i in range(1, len(tokens)) :
        label_str_file.write(tokens[i] + "\t" + "X\n")

    label_str_file.close()

    return


# 훈련 예제 생성 및 반환
def getTrainExample(srcpath="./ner_token_label_per_line.txt") :
    ids_list = []

    f = open(srcpath, "r", encoding="utf-8-sig")
    for line in f :
        ids_list.append(line[:-1])
    f.close()

    num_total_lines = len(ids_list)
    x = []
    y = []
    mask = []

    i = 0
    while i < num_total_lines :
        temp_x = [2]
        temp_y = [13]
        temp_mask = [1]

        while len(ids_list[i]) > 0 :
            cur_line = ids_list[i]
            line_splited = cur_line.split('\t')
            tok_str = line_splited[0]
            label_str = line_splited[1]
            tok_id = TOKENIZER.convert_tokens_to_ids([tok_str])
            tok_id = tok_id[0]
            label_id = LABEL_MAP[label_str]
            temp_x.append(tok_id)
            temp_y.append(label_id)
            temp_mask.append(1)
            i += 1
        i += 1

        leng = len(temp_x)
        if (leng > MSL - 2) :
            temp_x = temp_x[0:MSL-2]
            temp_y = temp_y[0:MSL-2]
            temp_mask = temp_mask[0:MSL-2]
        
        temp_x.append(3)
        temp_y.append(14)
        temp_mask.append(1)

        while len(temp_x) < MSL :
            temp_x.append(0)
            temp_y.append(0)
            temp_mask.append(0)
        
        x.append(temp_x)
        y.append(temp_y)
        mask.append(temp_mask)
    
    x = np.array(x, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    mask = np.array(mask, dtype=np.int32)

    return x, y, mask


# 학습 시작
def startLearning(x=np.array([0]), y=np.array([0]), mask=np.array([0]), model=BERT_archi(), opt="AdamW") :
    num_examples = x.shape[0]
    num_tra = int(0.6 * num_examples)
    num_tes = int(0.8 * num_examples)
    num_val = num_tes - num_tra

    # 학습용 데이터 셋
    train_x = np.zeros((num_tra, MSL), np.int32)
    train_y = np.zeros((num_tra, MSL), np.int32)
    train_mask = np.zeros((num_tra, MSL), np.int32)

    train_x[:, :] = x[:num_tra, :]
    train_y[:, :] = y[:num_tra, :]
    train_mask[:, :] = mask[:num_tra, :]

    train_x = torch.LongTensor(train_x)
    train_y = torch.LongTensor(train_y)
    train_mask = torch.LongTensor(train_mask)

    train_data = TensorDataset(train_x, train_y, train_mask)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # 검증용 데이터 셋
    valid_x = np.zeros((num_val, MSL), np.int32)
    valid_y = np.zeros((num_val, MSL), np.int32)
    valid_mask = np.zeros((num_val, MSL), np.int32)

    valid_x[:, :] = x[num_tra:num_tes, :]
    valid_y[:, :] = y[num_tra:num_tes, :]
    valid_mask[:, :] = mask[num_tra:num_tes, :]

    valid_x = torch.LongTensor(valid_x)
    valid_y = torch.LongTensor(valid_y)
    valid_mask = torch.LongTensor(valid_mask)

    valid_data = TensorDataset(valid_x, valid_y, valid_mask)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

    model = model.to(DEVICE)

    # optimizer 선택
    optimizer = None
    if opt == "SGD" :
        optimizer = optim.SGD(model.parameters(), lr=1e-5, weight_decay=0.001)
    elif opt == "Adam" :
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.1)
    else :
        optimizer = optim.AdamW(model.parameters(), lr=0.5e-4, betas=(0.9, .999), eps=1e-08, weight_decay=0.01)

    modelName = str(type(model)).replace("class", "", 1).replace(".", "-").replace("<", "").replace(">", "").replace(" ", "").replace("'", "")
    modelName = modelName.replace("__main__-", "")

    train_loss = list()
    valid_loss = list()

    minimum_loss = -1.0
    best_epoch = -1
    epoch = 0
    life = 0
    while True :
        model, avg_loss = trainModel(train_dataloader, optimizer, model)
        train_loss.append(avg_loss)

        log(f"\tEpoch : {epoch} is finished. Avg_loss = {avg_loss}")

        avg_loss = validateModel(valid_dataloader, model)
        valid_loss.append(avg_loss)

        if  minimum_loss < 0 or minimum_loss > avg_loss :
            minimum_loss = avg_loss
            best_epoch = epoch + 1
            life = 0
            torch.save(model.state_dict(), f"./{modelName}_{opt}.pth")
        elif life >= LIFE :
            break

        epoch += 1
        life += 1

    log(f"\tBest epoch : {best_epoch}")
    
    x_axis = range(epoch + 1)
    plt.plot(x_axis, train_loss, label="training")
    plt.plot(x_axis, valid_loss, label="validation")
    plt.axvline(x=best_epoch-1, ymin=0, ymax=1, linestyle="--", label="best epoch point")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("./" + modelName + "_" + opt + ".png")
    
    return model


# 모델 학습
def trainModel(dataloader, optimizer, model) :
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    total_loss = 0.0
    total_num_batches = len(dataloader)
    model.train()
    for j, batch in enumerate(dataloader) :
        batch = tuple(r.to(DEVICE) for r in batch)
        _x, _y, _mask = batch
        optimizer.zero_grad()
        model.zero_grad()
        output_seq = model(_x, _mask)
        preds_tr = torch.transpose(output_seq, 1, 2)
        loss = loss_fn(preds_tr, _y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / (total_num_batches * BATCH_SIZE)

    return (model, avg_loss)


# 모델 검증
def validateModel(dataloader, model) :
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    total_loss = 0.0
    total_num_batches = len(dataloader)
    model.eval()
    with torch.no_grad() :
        for k, batch in enumerate(dataloader) :
            batch = tuple(r.to(DEVICE) for r in batch)
            _x, _y, _mask = batch
            output_seq = model(_x, _mask)
            preds_tr = torch.transpose(output_seq, 1, 2)

            loss = loss_fn(preds_tr, _y).item()
            total_loss += loss
        avg_loss = total_loss / (total_num_batches * BATCH_SIZE)
    
    return avg_loss


# 모델 테스트
def testModel(x=np.array([0]), y=np.array([0]), mask=np.array([0]), model=BERT_archi()) :
    num_examples = x.shape[0]
    num_tes = int(0.8 * num_examples)

    test_x = np.zeros((num_examples-num_tes, MSL), np.int32)
    test_y = np.zeros((num_examples-num_tes, MSL), np.int32)
    test_mask = np.zeros((num_examples-num_tes, MSL), np.int32)

    test_x[:, :] = x[num_tes:, :]
    test_y[:, :] = y[num_tes:, :]
    test_mask[:, :] = mask[num_tes:, :]

    test_x = torch.LongTensor(test_x)
    test_y = torch.LongTensor(test_y)
    test_mask = torch.LongTensor(test_mask)

    test_data = TensorDataset(test_x, test_y, test_mask)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    model = model.to(DEVICE)
    
    softmax_fx = torch.nn.Softmax(dim=2)

    total_TN, total_FP, total_FN, total_TP = 0, 0, 0, 0
    model.eval()
    with torch.no_grad() :
        for k, batch in enumerate(test_dataloader) :
            batch = tuple(r.to(DEVICE) for r in batch)
            _x, _y, _mask = batch
            output_seq = model(_x, _mask)
            preds = softmax_fx(output_seq)
            pred_label = torch.argmax(preds, dim=2)

            for i in range(len(_y)) :
                target_label_seq = _y[i]
                pred_label_seq = pred_label[i]
                tn, fp, fn, tp = 0, 0, 0, 0
                for j in range(MSL) :
                    if target_label_seq[j] == 0 :
                        break
                    # NE인데, 정확한 태깅을 한 경우 (True Positive)
                    if target_label_seq[j] in ENTITY_LABEL and target_label_seq[j] == pred_label_seq[j] :
                        tp += 1
                    # NE가 아닌데, 실제로도 NE가 아니었던 경우 (True Negative)
                    elif target_label_seq[j] not in ENTITY_LABEL and pred_label_seq[j] not in ENTITY_LABEL :
                        tn += 1
                    # NE인데, 정확히 태깅을 못한 경우 (False Positive)
                    elif target_label_seq[j] in ENTITY_LABEL and target_label_seq[j] != pred_label_seq[j] :
                        fp += 1
                    # NE가 아닌데, 실제로는 NE였던 경우 (False Negative)
                    elif target_label_seq[j] not in ENTITY_LABEL and pred_label_seq[j] in ENTITY_LABEL :
                        fn += 1
                total_TP += tp
                total_TN += tn
                total_FP += fp
                total_FN += fn

        accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
        recall = total_TP / (total_TP + total_FN)
        precision = total_TP / (total_TP + total_FP)
        f1_score = (2 * recall * precision) / (recall + precision)
        log(f"\tAccuracy : {accuracy}")
        log(f"\tRecall : {recall}")
        log(f"\tPrecision : {precision}")
        log(f"\tF1-Score : {f1_score}")

    return


# 문장 -> 토큰
def getToken(line="") :
    words = line.split()

    result_x = []
    result_mask = []
    tokens = ['[CLS]_']

    temp_x = [2]
    temp_mask = [1]

    for word in words :
        if len(word) <= 0 :
            continue
        token = TOKENIZER.tokenize(word)
        token_id = TOKENIZER.convert_tokens_to_ids(token)
        token_id = token_id[0]
        temp_x.append(token_id)
        temp_mask.append(1)
        tokens.extend(token)

    if len(temp_x) > MSL-2 :
        temp_x = temp_x[0:MSL-2]
        temp_mask = temp_mask[0:MSL-2]
    
    temp_x.append(3)
    temp_mask.append(1)

    while len(temp_x) < MSL :
        temp_x.append(LABEL_MAP[LABEL[0]])
        temp_mask.append(0)

    result_x.append(temp_x)
    result_mask.append(temp_mask)
    
    result_x = np.array(result_x, dtype=np.int32)
    result_mask = np.array(result_mask, dtype=np.int32)
    
    return result_x, result_mask, tokens


# 토큰열 -> 훈련예제
def convertExample(line="") :
    x, mask, tokens = getToken(line)
    num_examples = x.shape[0]

    train_x = np.zeros((num_examples, MSL), np.int32)
    train_mask = np.zeros((num_examples, MSL), np.int32)
    train_x[:, :] = x[:num_examples, :]
    train_mask[:, :] = mask[:num_examples, :]

    train_x = torch.LongTensor(train_x)
    train_mask = torch.LongTensor(train_mask)

    train_data = TensorDataset(train_x, train_mask)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    
    _x = None
    _mask = None
    for j, batch in enumerate(train_dataloader) :
        batch = tuple(r.to(DEVICE) for r in batch)
        _x, _mask = batch
    
    return _x, _mask, tokens


# 예측결과 -> 태깅된 개체
def findEntity(tokens, label) :
    word_list = list()
    word = ""
    tag = ""
    for i in range(len(tokens)) :
        if label[i].item() in ENTITY_LABEL :
            if word == "" :
                word = tokens[i]
                tag = LABEL[label[i].item()][2:]
            else :
                word = word.replace("_", "").strip()
                word_list.append([word, tag])
                word = tokens[i]
                tag = LABEL[label[i].item()][2:]
            if tokens[i][-1] == "_" and label[i+1].item() not in [2, 5, 7, 9, 11] :
                word = word.replace("_", " ").strip()
                word_list.append([word, tag])
                word = ""
                tag = ""
        elif word != "" and len(tokens[i]) >= 1 and tokens[i][-1] == "_"  :
            word += tokens[i]
            if label[i+1].item() not in [2, 5, 7, 9, 11] :
                word = word.replace("_", " ").strip()
                word_list.append([word, tag])
                word = ""
                tag = ""
        elif word != "" :
            word += tokens[i]
    
    if word != "" :
        word = word.replace("_", " ").strip()
        word_list.append([word, tag])

    return word_list



def main() :
    log(msg="\n------------------------" + str(datetime.now()) + "------------------------\n", output=False)
    log(f"DEVICE : {DEVICE}\n")
    
    # 저장된 모델 불러오기
    model_files = list()
    file_list = os.listdir("./")
    for file_name in file_list :
        if os.path.splitext(file_name)[1] == ".pth" :
            model_files.append(file_name)
    
    selectedModelFile = ""
    if len(model_files) > 0 :
        log("------------------------")
        for i in range(len(model_files)) :
            log(f"[{i+1}]   {model_files[i]}")
        log("[N]   사용 안함")
        log("------------------------")
        cmd = input("불러올 모델파일 번호 입력   >> ").upper()
        log(f"불러올 모델파일 번호 입력   >> {cmd}", output=False)
        log("")

        if cmd.isnumeric() and len(model_files) >= int(cmd) and 0 < int(cmd) :
            selectedModelFile = model_files[int(cmd)-1]

    model = BERT_archi()

    if selectedModelFile != "" :
        # 선택한 모델파일 불러오기
        model.load_state_dict(torch.load("./" + selectedModelFile))
        model = model.to(DEVICE)
        log("")
    else :
        # 옵티마이저 선택
        optimizer = "AdamW"
        log("-----------------------------")
        log(f"1.SGD\t2.Adam\t3.AdamW(기본)")
        log("-----------------------------")
        cmd = input("옵티마이저 번호 입력 >> ")
        log(f"옵티마이저 번호 입력 >> {cmd}", output=False)
        log("\n")
        if cmd == "1" :
            optimizer = "SGD"
        elif cmd == "2" :
            optimizer = "Adam"
        
        # 문장 분석한 파일 생성
        if not os.path.exists("./ner_eojeol_label_per_line.txt") :
            log("문장 분석 중")
            start_time = time.perf_counter()
            isSuccess = eojeol()
            if not isSuccess :
                return
            end_time = time.perf_counter()
            elapse = end_time - start_time
            log(f"└분석 완료 ({elapse:.3f} sec)\n")

        # 각 문장 토큰화 및 레이블링
        if not os.path.exists("./ner_token_label_per_line.txt") :
            log("토큰화 중")
            start_time = time.perf_counter()
            tokenizer()
            end_time = time.perf_counter()
            elapse = end_time - start_time
            log(f"└토큰화 완료 ({elapse:.3f} sec)\n")
    
        # 학습 예제 생성
        log("학습 예제 생성 중")
        start_time = time.perf_counter()
        x, y, mask = getTrainExample()
        end_time = time.perf_counter()
        elapse = end_time - start_time
        log(f"└학습 예제 생성 완료 ({elapse:.3f} sec)\n")

        # 모델 훈련
        log("훈련 중")
        start_time = time.perf_counter()
        model = startLearning(x, y, mask, model, opt=optimizer)
        end_time = time.perf_counter()
        elapse = end_time - start_time
        log(f"└훈련 완료 ({elapse:.3f} sec)\n")

        # 모델 테스팅
        log("테스팅 중")
        start_time = time.perf_counter()
        testModel(x, y, mask, model)
        end_time = time.perf_counter()
        elapse = end_time - start_time
        log(f"└테스팅 완료 ({elapse:.3f} s)\n")
    
    # 실제 입력
    cmd = "dummy"
    while True :
        cmd = input("문장 입력 >> ")
        log(f"문장 입력 >> {cmd}", output=False)

        if cmd == "" :
            break
        user_x, user_mask, tokens = convertExample(cmd)

        pred = model(user_x, user_mask)
        softmax_fn = torch.nn.Softmax(dim=2)
        pred = softmax_fn(pred)
        pred_label = torch.argmax(pred, dim=2)

        tagged_list = findEntity(tokens, pred_label[0])
        for t in tagged_list :
            log(f"{t[0]} : {t[1]}")
        if len(tagged_list) <= 0 :
            log("결과가 없습니다.")
        log("\n")
    
    return

if __name__ == "__main__" :
    main()
