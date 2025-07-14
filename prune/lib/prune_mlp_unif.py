import torch
import torch.nn as nn
from .layerwrapper import BiasGPT
from .calidata_select_8 import calidata_loaders
from tqdm import tqdm


"""
    'IFV': Input Feature Variance
    'WIFV': Weighted Input Feature Variance
    'WIFN': Weighted Input Feature Norm
"""
metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
}


def find_layers(module, layers = [nn.Linear], name = ''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers = layers, name = name + '.' + name1 if name != '' else name1
        ))
    return res


def check_mlp_sparsity(model):
    """
    Check the sparsity of the weights in different layers of mlp in the model.

    Args:
        model (nn.Module): The model to check.

    Returns:
        float: Ratio of the count of non-zero weights to total parameters of mlp in the model.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size

    mlp_count = 0
    mlp_params = 0
    model_count = 0
    model_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_mlp_count = 0
        sub_mlp_params = 0
        
        for name in subset:
            W = subset[name].weight.data
            model_count += W.numel()
            
            if 'self_attn' in name:
                model_params += hidden_size * hidden_size
            else:
                model_params += hidden_size * intermediate_size
                mlp_params += hidden_size * intermediate_size
                sub_mlp_params += hidden_size * intermediate_size
                mlp_count += W.numel()
                sub_mlp_count += W.numel()
            
            if subset[name].bias is not None:
                model_count += subset[name].bias.data.numel()

    print(f"Total MLP Density: {float(mlp_count) / mlp_params:.4f}")
    print(f"Total Model Density: {float(model_count) / model_params:.4f}")
    
    model.config.use_cache = use_cache
    return float(mlp_count) / mlp_params


def prepare_calibration_input(model, dataloader, device):
    """
    Prepare inputs for model calibration. 

    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded.

    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((2048, model.seqlen, model.config.hidden_size), dtype = dtype, device = device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False):
    """
    Compress a model layer by masking or pruning based on the given masks.
    
    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        unstr (bool, optional): If True, only mask without real pruning. Defaults to False.
        
    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """
    if unstr:
        if mlp_mask is not None:
            layer.mlp.up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            layer.mlp.gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)

            output_weight = layer.mlp.down_proj.weight.data

            if bias:
                output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)
                
            if bias:
                layer.mlp.down_proj.bias.data = output_bias
            layer.mlp.down_proj.weight.data = output_weight
    
    else:
        if mlp_mask is not None:
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
            
            layer.mlp.up_proj.out_features = mlp_mask.sum().item()
            layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
            
            output_weight = layer.mlp.down_proj.weight.data
            layer.mlp.intermediate_size = mlp_mask.sum().item()
            if bias:
                output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)
              
            output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]  

            if bias:
                layer.mlp.down_proj.in_features = mlp_mask.sum().item()
                layer.mlp.down_proj.bias.data = output_bias

            layer.mlp.down_proj.weight.data = output_weight

    torch.cuda.empty_cache()

# FLAP for MLPonly
def prune_flap_mlp(args, model, tokenizer, device = torch.device("cuda:0")):

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("calibration data loading starts")
    print("# expert: " + args.expert)
    dataloader= calidata_loaders(name = args.expert, nsamples = args.nsamples, seed = args.seed, seqlen = model.seqlen, tokenizer = tokenizer)
    print("calibration data loading completes")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    layers = model.model.layers

    mlp_metric_list = []
    mlp_baseline_inp_list = []
    mlp_mask, attn_mask = [], []

    # Split into sub-problems, separate statistics for each module
    for i in tqdm(range(len(layers)), desc = "Processing layers"):
        layer = layers[i]
        subset = {}
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
        
        wrapped_layers = {}

        for name in subset:
            wrapped_layers[name] = BiasGPT(subset[name], args.metrics)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # ***************************
        results = torch.empty(args.nsamples, model.seqlen, model.config.hidden_size)

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                results[j] = outs[j]
        
        for h in handles:
            h.remove()

        for name in subset:
            W_metric = metrics[args.metrics](wrapped_layers, subset, name)
            thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel() * args.mlp_pruning_ratio)].cpu()
            W_mask = (W_metric >= thresh)
            mlp_mask.append(W_mask)

            mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            wrapped_layers[name].free()

        inps, outs = outs, inps # Use the original output as input to the next layer
        torch.cuda.empty_cache()

    mlp_mask = torch.stack(mlp_mask)

    for idx in range(len(layers)):
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress(model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], model.hf_device_map[f"model.layers.{idx}"], unstr = args.unstr)
        else:
            compress(model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], device, unstr = args.unstr)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()