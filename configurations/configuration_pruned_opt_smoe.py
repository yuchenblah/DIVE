""" Pruned OPT SMoE model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class PrunedOPTSMoEConfig(PretrainedConfig):
    model_type = "pruned_opt_smoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # -------- original opt configs --------
        vocab_size=50272,
        hidden_size=768,
        num_hidden_layers=12,
        ffn_dim=3072,
        max_position_embeddings=2048,
        do_layer_norm_before=True,
        _remove_final_layer_norm=False,
        word_embed_proj_dim=None,
        dropout=0.1,
        attention_dropout=0.0,
        num_attention_heads=12,
        # -------- moe expert configs --------
        num_experts=8,
        num_selects=2,
        # -------- original opt configs --------
        activation_function="relu",
        layerdrop=0.0,
        init_std=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
        enable_bias=True,
        layer_norm_elementwise_affine=True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.word_embed_proj_dim = word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size
        self.ffn_dim = ffn_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.num_selects = num_selects
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.do_layer_norm_before = do_layer_norm_before
        # We keep these variables at `True` for backward compatibility.
        self.enable_bias = enable_bias
        self.layer_norm_elementwise_affine = layer_norm_elementwise_affine

        # Note that the only purpose of `_remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        self._remove_final_layer_norm = _remove_final_layer_norm
