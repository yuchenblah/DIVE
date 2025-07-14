import random
import torch
from datasets import load_dataset


def get_wikitext2(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_c4(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_mnli(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('SetFit/mnli', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        sol_choice = random.choice(['text1', 'text2'])
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_qnli(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('SetFit/qnli', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        sol_choice = random.choice(['text1', 'text2'])
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        
    return trainloader


def get_rte(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('SetFit/rte', split='train', cache_dir="./datasets")
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        sol_choice = random.choice(['text1', 'text2'])
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_anli(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('facebook/anli', split='train_r3', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        sol_choice = random.choice(['premise', 'reason'])
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_squad2(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('rajpurkar/squad_v2', split='train', cache_dir="./datasets")
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['context'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_boolq(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('google/boolq', split='train', cache_dir="./datasets")
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['passage'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_piqa(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('piqa', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        # Randomly choose between sol1 and sol2
        sol_choice = random.choice(['sol1', 'sol2'])
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_drop(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('ucinlp/drop', split='train', cache_dir="./datasets")
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['passage'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_coqa(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('stanfordnlp/coqa', split='train', cache_dir="./datasets")
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['story'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_logiqa(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('lucasmccabe/logiqa', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['context'], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i]['context'], return_tensors='pt')

                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_record(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('Patt/ReCoRD_TH', split='train', cache_dir="./datasets")
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['passage'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_gsm8k(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('gsm8k', "main", split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['answer'], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i]['answer'], return_tensors='pt')

                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_mathqa(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('math_qa', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        sol_choice = random.choice(['options', 'Rationale'])
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_sciq(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('allenai/sciq', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        sol_choice = random.choice(['correct_answer', 'support'])
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_qasper(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('allenai/qasper', split='train', cache_dir="./datasets")
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['abstract'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_pubmedqa(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('fedml/PubMedQA_instruction', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['context'], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i]['context'], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_race(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('ehovy/race', 'all', split='train', cache_dir="./datasets")
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['article'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_winogrande(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('winogrande', 'winogrande_xl', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['sentence'], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i]['sentence'], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_hellaswag(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('Rowan/hellaswag', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['ctx'], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i]['ctx'], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_qqp(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('SetFit/qqp', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        sol_choice = random.choice(['text1', 'text2'])
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_mrpc(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('SetFit/mrpc', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        sol_choice = random.choice(['text1', 'text2'])
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i][sol_choice], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_sst2(nsamples_random, seed, seqlen, tokenizer):
    traindata = load_dataset('stanfordnlp/sst2', split='train', cache_dir="./datasets")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples_random):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['sentence'], return_tensors='pt')
            while trainenc.input_ids.shape[1] <= seqlen:
                i = random.randint(0, len(traindata) - 1)
                newenc = tokenizer(traindata[i]['sentence'], return_tensors='pt')

                # Concatenate the newenc to trainenc
                trainenc.input_ids = torch.cat((trainenc.input_ids, newenc.input_ids), dim=1)
                trainenc.attention_mask = torch.cat((trainenc.attention_mask, newenc.attention_mask), dim=1)
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_loaders(name='WikiText2', nsamples_random=1024, seed=0, seqlen=256, tokenizer=None):
    if "WikiText2" in name:
        return get_wikitext2(nsamples_random, seed, seqlen, tokenizer)
    elif "C4" in name:
        return get_c4(nsamples_random, seed, seqlen, tokenizer)
    elif "MNLI" in name:
        return get_mnli(nsamples_random, seed, seqlen, tokenizer)
    elif "QNLI" in name:
        return get_qnli(nsamples_random, seed, seqlen, tokenizer)
    elif "RTE" in name:
        return get_rte(nsamples_random, seed, seqlen, tokenizer)
    elif "ANLI" in name:
        return get_anli(nsamples_random, seed, seqlen, tokenizer)
    elif "SQuAD2" in name:
        return get_squad2(nsamples_random, seed, seqlen, tokenizer)
    elif "BoolQ" in name:
        return get_boolq(nsamples_random, seed, seqlen, tokenizer)
    elif "PIQA" in name:
        return get_piqa(nsamples_random, seed, seqlen, tokenizer)
    elif "DROP" in name:
        return get_drop(nsamples_random, seed, seqlen, tokenizer)
    elif "CoQA" in name:
        return get_coqa(nsamples_random, seed, seqlen, tokenizer)
    elif "LogiQA" in name:
        return get_logiqa(nsamples_random, seed, seqlen, tokenizer)
    elif "ReCoRD" in name:
        return get_record(nsamples_random, seed, seqlen, tokenizer)
    elif "GSM8K" in name:
        return get_gsm8k(nsamples_random, seed, seqlen, tokenizer)
    elif "MathQA" in name:
        return get_mathqa(nsamples_random, seed, seqlen, tokenizer)
    elif "SciQ" in name:
        return get_sciq(nsamples_random, seed, seqlen, tokenizer)
    elif "Qasper" in name:
        return get_qasper(nsamples_random, seed, seqlen, tokenizer)
    elif "PubMedQA" in name:
        return get_pubmedqa(nsamples_random, seed, seqlen, tokenizer)
    elif "RACE" in name:
        return get_race(nsamples_random, seed, seqlen, tokenizer)
    elif "WinoGrande" in name:
        return get_winogrande(nsamples_random, seed, seqlen, tokenizer)
    elif "HellaSwag" in name:
        return get_hellaswag(nsamples_random, seed, seqlen, tokenizer)
    elif "QQP" in name:
        return get_qqp(nsamples_random, seed, seqlen, tokenizer)
    elif "MRPC" in name:
        return get_mrpc(nsamples_random, seed, seqlen, tokenizer)
    elif "SST2" in name:
        return get_sst2(nsamples_random, seed, seqlen, tokenizer)

expert_groups = {
    0: ["C4", "WikiText2", "MNLI", "QNLI", "RTE", "ANLI", "SQuAD2", "BoolQ", "PIQA", "DROP", "CoQA", "LogiQA", "ReCoRD", "GSM8K", "MathQA", "SciQ", "Qasper", "PubMedQA", "RACE", "WinoGrande", "HellaSwag", "QQP", "MRPC", "SST2"],
    1: ["C4", "WikiText2", "MNLI", "QNLI", "RTE", "ANLI", "SQuAD2", "BoolQ", "PIQA", "DROP", "CoQA", "LogiQA", "ReCoRD", "GSM8K", "MathQA", "SciQ", "Qasper", "PubMedQA", "RACE", "WinoGrande", "HellaSwag", "QQP", "MRPC", "SST2"],
    2: ["C4", "WikiText2", "MNLI", "QNLI", "RTE", "ANLI", "SQuAD2", "BoolQ", "PIQA", "DROP", "CoQA", "LogiQA", "ReCoRD", "GSM8K", "MathQA", "SciQ", "Qasper", "PubMedQA", "RACE", "WinoGrande", "HellaSwag", "QQP", "MRPC", "SST2"],
    3: ["C4", "WikiText2", "MNLI", "QNLI", "RTE", "ANLI", "SQuAD2", "BoolQ", "PIQA", "DROP", "CoQA", "LogiQA", "ReCoRD", "GSM8K", "MathQA", "SciQ", "Qasper", "PubMedQA", "RACE", "WinoGrande", "HellaSwag", "QQP", "MRPC", "SST2"],
    4: ["C4", "WikiText2", "MNLI", "QNLI", "RTE", "ANLI", "SQuAD2", "BoolQ", "PIQA", "DROP", "CoQA", "LogiQA", "ReCoRD", "GSM8K", "MathQA", "SciQ", "Qasper", "PubMedQA", "RACE", "WinoGrande", "HellaSwag", "QQP", "MRPC", "SST2"],
    5: ["C4", "WikiText2", "MNLI", "QNLI", "RTE", "ANLI", "SQuAD2", "BoolQ", "PIQA", "DROP", "CoQA", "LogiQA", "ReCoRD", "GSM8K", "MathQA", "SciQ", "Qasper", "PubMedQA", "RACE", "WinoGrande", "HellaSwag", "QQP", "MRPC", "SST2"],
    6: ["C4", "WikiText2", "MNLI", "QNLI", "RTE", "ANLI", "SQuAD2", "BoolQ", "PIQA", "DROP", "CoQA", "LogiQA", "ReCoRD", "GSM8K", "MathQA", "SciQ", "Qasper", "PubMedQA", "RACE", "WinoGrande", "HellaSwag", "QQP", "MRPC", "SST2"],
    7: ["C4", "WikiText2", "MNLI", "QNLI", "RTE", "ANLI", "SQuAD2", "BoolQ", "PIQA", "DROP", "CoQA", "LogiQA", "ReCoRD", "GSM8K", "MathQA", "SciQ", "Qasper", "PubMedQA", "RACE", "WinoGrande", "HellaSwag", "QQP", "MRPC", "SST2"],
}


def get_expert_samples(expert_datasets, nsamples, seed, seqlen, tokenizer):
    random.seed(seed)
    all_samples = []

    for dataset_name in expert_datasets:
        dataset_samples = get_loaders(name=dataset_name, nsamples_random=1024, seed=seed, seqlen=seqlen, tokenizer=tokenizer)
        all_samples.extend(dataset_samples)

    if len(all_samples) < nsamples:
        raise ValueError("Not enough samples to meet the required number of samples")

    random.shuffle(all_samples)
    print(len(all_samples[:nsamples]))
    return all_samples[:nsamples]


# Calidata_loaders for each expert
def calidata_loaders(tokenizer, name='Expert0', nsamples=1024, seed=0, seqlen=256):
    if "Expert0" in name:
        return get_expert_samples(expert_groups[0], nsamples, 0, seqlen, tokenizer)
    if "Expert1" in name:
        return get_expert_samples(expert_groups[1], nsamples, 1, seqlen, tokenizer)
    if "Expert2" in name:
        return get_expert_samples(expert_groups[2], nsamples, 2, seqlen, tokenizer)
    if "Expert3" in name:
        return get_expert_samples(expert_groups[3], nsamples, 3, seqlen, tokenizer)
    if "Expert4" in name:
        return get_expert_samples(expert_groups[4], nsamples, 4, seqlen, tokenizer)
    if "Expert5" in name:
        return get_expert_samples(expert_groups[5], nsamples, 5, seqlen, tokenizer)
    if "Expert6" in name:
        return get_expert_samples(expert_groups[6], nsamples, 6, seqlen, tokenizer)
    if "Expert7" in name:
        return get_expert_samples(expert_groups[7], nsamples, 7, seqlen, tokenizer)