import torch
import torch.nn as nn
from .evaldata import get_loaders


def eval_ppl(model, tokenizer, seqlen, device=torch.device("cuda:0")):
    """
    Evaluate perplexity (ppl) on a specified model and tokenizer.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        list: List of perplexities for each test dataset.
    """
    test_datasets = ["WikiText2"]
    perplexities = []

    for dataset in test_datasets:
        print("# evaldata: " + dataset)
        _, testloader = get_loaders(name=dataset, tokenizer=tokenizer)

        with torch.no_grad():
            ppl = eval_ppl_testdata(model, testloader, seqlen, 1, device)
            perplexities.append(ppl)

    return perplexities


def eval_ppl_testdata(model, testenc, seqlen, bs=1, device=None):
    """
    Evaluate perplexity (ppl).

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        testenc (TokenizerWrapper): Encoded input IDs from test set.
        bs (int): Batch size for evaluation.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the test dataset.
    """
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    nlls = []

    for i in range(0, nsamples, bs):
        j = min(i + bs, nsamples)

        inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
        inputs = inputs.reshape(j-i, seqlen)

        lm_logits = model(inputs).logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * seqlen * (j-i)

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    torch.cuda.empty_cache()
    return ppl.item()