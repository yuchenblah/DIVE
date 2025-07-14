from datasets import load_dataset


def get_ptb(tokenizer):
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', cache_dir="./datasets")

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    trainloader = []
    return trainloader, testenc


def get_wikitext2(tokenizer):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir="./datasets")

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    trainloader = []
    return trainloader, testenc


def get_loaders(name='WikiText2', tokenizer=None):
    if "WikiText2" in name:
        return get_wikitext2(tokenizer)
    elif "PTB" in name:
        return get_ptb(tokenizer)