""" Pruned TinyLlama SMoE model configuration"""

from transformers.configuration_utils import PretrainedConfig


LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class PrunedLlamaSMoEConfig(PretrainedConfig):
    model_type = "pruned_llama_smoe"

    def __init__(
        self,
        # -------- original llama configs --------
        vocab_size=32000,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=22,
        num_attention_heads=32,
        # -------- moe expert configs --------
        num_experts=8,
        num_selects=2,
        # -------- original llama configs --------
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.num_selects = num_selects
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
