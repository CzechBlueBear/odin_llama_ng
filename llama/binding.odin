package llama

import "core:c"
import "ggml"

main :: proc() {
}

MAX_DEVICES :: 16
DEFAULT_SEED :: 0xFFFFFFFF

gpt_params_ptr :: distinct rawptr

// Some objects are better considered opaque and only handled with methods
llama_model_ptr :: distinct rawptr
llama_context_ptr :: distinct rawptr
llama_vocab_ptr :: distinct rawptr
llama_memory_ptr :: distinct rawptr  // llama_memory_t, which is (llama_memory_i*)

Token :: distinct c.int32_t
llama_pos :: distinct c.int32_t
llama_seq_id :: distinct c.int32_t

// Input data for llama_encode/llama_decode
// A llama_batch object can contain input about one or many sequences
// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
//
// - token  : the token ids of the input (used when embd is NULL)
// - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
// - pos    : the positions of the respective token in the sequence
//            (if set to NULL, the token position will be tracked automatically by llama_encode/llama_decode)
// - seq_id : the sequence to which the respective token belongs
//            (if set to NULL, the sequence ID will be assumed to be 0)
// - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
//            (if set to NULL:
//               - if embeddings: all tokens are output
//               - if not:        only the last token is output
//            )
//
Batch :: struct {
	n_tokens:   c.int32_t,
	token:      [^]Token,
	embd:       [^]c.float,
	pos:        [^]llama_pos,
	n_seq_id:   [^]c.int32_t,
	seq_id:     ^^llama_seq_id, // TODO: should it be [^][^]llama_seq_id?
	logits:     [^]c.int8_t,
	all_pos_0:  llama_pos, // used if pos == NULL
	all_pos_1:  llama_pos, // used if pos == NULL
	all_seq_id: llama_seq_id, // used if seq_id == NULL
}

Tokens :: struct {
	len:    c.int32_t,
	tokens: [^]Token,
}

Perf_Type :: enum c.int {
	Context = 0,
    Sampler_Chain = 1,
}

Split_Mode :: enum c.int {
    None = 0,   // single GPU
    Layer = 1,  // split layers and KV across GPUs
    Row = 2,    // split layers and KV across GPUs, use tensor parallelism if supported
    Tensor = 3
}

// ggml_backend_device :: struct {
//     struct ggml_backend_device_i iface;
//     ggml_backend_reg_t reg;
//     context: rawptr,
// }

// ggml_backend_dev_t :: ^ggml_device

Model_Params :: struct {

    // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
    devices: [^]rawptr,		// [^]ggml_backend_dev_t

    // NULL-terminated list of buffer types to use for tensors that match a pattern
    tensor_buft_overrides: [^]rawptr, // [^]llama_model_tensor_buft_override,

    // number of layers to store in VRAM, a negative value means all layers
    n_gpu_layers: c.int32_t,

    // how to split the model across multiple GPUs
    split_mode: Split_Mode,

    // the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
    main_gpu: c.int32_t,

    // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
    tensor_split: [^]f32,

    // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
    // If the provided progress_callback returns true, model loading continues.
    // If it returns false, model loading is immediately aborted.
    progress_callback: proc(progress: c.float) -> c.bool,

    // context pointer passed to the progress callback
    progress_callback_user_data: rawptr,

    // override key-value pairs of the model meta data
    kv_overrides: [^]rawptr, // [^]llama_model_kv_override,

    // Keep the booleans together to avoid misalignment during copy-by-value.
    vocab_only: c.bool,      // only load the vocabulary, no weights
    use_mmap: c.bool,        // use mmap if possible
    use_direct_io: c.bool,   // use direct io, takes precedence over use_mmap when supported
    use_mlock: c.bool,       // force system to keep model in RAM
    check_tensors: c.bool,   // validate model tensor data
    use_extra_bufts: c.bool, // use extra buffer types (used for weight repacking)
    no_host: c.bool,         // bypass host buffer allowing extra buffers to be used
    no_alloc: c.bool,        // only load metadata and simulate memory allocations
}

Rope_Scaling_Type :: enum c.int32_t {
	UNSPECIFIED = -1,
	NONE        = 0,
	LINEAR      = 1,
	YARN        = 2,
	LONGROPE    = 3,
}

Pooling_Type :: enum c.int32_t {
    UNSPECIFIED = -1,
    NONE = 0,
    MEAN = 1,
    CLS  = 2,
    LAST = 3,
    RANK = 4, // used by reranking models to attach the classification head to the graph
}

Attention_Type :: enum c.int32_t {
    Unspecified = -1,
    Causal = 0,
    Non_Causal = 1,
}

Flash_Attention_Type :: enum c.int32_t {
    Auto = -1,
    Disabled = 0,
    Enabled = 1,
}

// TODO: simplify (https://github.com/ggml-org/llama.cpp/pull/9294#pullrequestreview-2286561979)
Token_Data :: struct {			// llama_token_data
	id:     Token,	// token id
	logit:  c.float,  // log-odds of the token
	p:      c.float,  // probability of the token
}

Token_Data_Array :: struct {	// llama_token_data_array

    // TODO: consider SoA
    // NOTE: this pointer can be modified by the samplers
    data:  ^Token_Data,
    size:  c.size_t,
    selected:  c.int64_t,    // this is the index in the data array (i.e. not the token id)
    sorted:    bool          // note: do not assume the data is sorted - always check this flag
}

Sampler_Data :: struct {
	logits:    ^ggml.Tensor,
	probs:     ^ggml.Tensor,
	sampled:   ^ggml.Tensor,
	candidates: ^ggml.Tensor
}

Sampler_Interface :: struct {
	name:	proc "c" (smpl: ^Sampler) -> cstring,                 // can be NULL
	accept: proc "c" (smpl: ^Sampler, token: Token),              // can be NULL
	apply:  proc "c" (smpl: ^Sampler, cur_p: ^Token_Data_Array),  // required
	reset:  proc "c" (smpl: ^Sampler),                            // can be NULL
	clone:  proc "c" (smpl: ^Sampler) -> ^Sampler,          // can be NULL if ctx is NULL
	free:   proc "c" (smpl: ^Sampler),                            // can be NULL if ctx is NULL

    // [EXPERIMENTAL]
    // backend sampling interface:

    // return true if the backend supports all ops needed by the sampler
    // note: call once per sampler
    backend_init:    proc "c" (smpl: ^Sampler, buft: rawptr /* ggml_backend_buffer_type_t */),

    // call after .backend_apply()
    backend_accept:  proc "c" (smpl: ^Sampler, ctx: ^ggml.Context, gf: ^ggml.Cgraph, selected_token: ^ggml.Tensor),

    // call after .backend_init()
    backend_apply:   proc "c" (smpl: ^Sampler, ctx: ^ggml.Context, gf: ^ggml.Cgraph, data: ^Sampler_Data),

    // called before graph execution to set inputs for the current ubatch
    backend_set_input:    proc "c" (smpl: ^Sampler)
}

Sampler_Chain_Params :: struct {
    no_perf: bool,  // whether to measure performance timings
}

Sampler :: struct {
    iface: ^Sampler_Interface,
    ctx: rawptr
}

Sampler_Seq_Config :: struct {
	seq_id: llama_seq_id,
	sampler: ^Sampler,
}

Context_Params :: struct {
    n_ctx: c.uint32_t,            // text context, 0 = from model
    n_batch: c.uint32_t,          // logical maximum batch size that can be submitted to llama_decode
    n_ubatch: c.uint32_t,         // physical maximum batch size
    n_seq_max: c.uint32_t,        // max number of sequences (i.e. distinct states for recurrent models)
    n_threads: c.int32_t,         // number of threads to use for generation
    n_threads_batch: c.int32_t,   // number of threads to use for batch processing

    rope_scaling_type: Rope_Scaling_Type,        // RoPE scaling type, from `enum llama_rope_scaling_type`
    llama_pooling_type: Pooling_Type,            // whether to pool (sum) embedding results by sequence id
    llama_attention_type: Attention_Type,        // attention type to use for embeddings
    llama_flash_attn_type: Flash_Attention_Type,   // when to enable Flash Attention

    // ref: https://github.com/ggml-org/llama.cpp/pull/2054
    rope_freq_base: c.float,   // RoPE base frequency, 0 = from model
    rope_freq_scale: c.float,  // RoPE frequency scaling factor, 0 = from model
    yarn_ext_factor: c.float,  // YaRN extrapolation mix factor, negative = from model
    yarn_attn_factor: c.float, // YaRN magnitude scaling factor
    yarn_beta_fast: c.float,   // YaRN low correction dim
    yarn_beta_slow: c.float,   // YaRN high correction dim
    yarn_orig_ctx: c.uint32_t, // YaRN original context size
    defrag_thold: c.float,     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

    cb_eval:      proc "c" (tensor: ^ggml.Tensor, ask: bool, user_data: rawptr) -> bool,
    cb_eval_user_data: rawptr,

    type_k: ggml.Type,         // data type for K cache [EXPERIMENTAL]
    type_v: ggml.Type,         // data type for V cache [EXPERIMENTAL]

    // Abort callback
    // if it returns true, execution of llama_decode() will be aborted
    // currently works only with CPU execution
    abort_callback:      proc "c" (data: rawptr) -> bool,
    abort_callback_data: rawptr,

    // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
    embeddings: bool,          // if true, extract embeddings (together with logits)
    offload_kqv: bool,         // offload the KQV ops (including the KV cache) to GPU
    no_perf: bool,             // measure performance timings
    op_offload: bool,          // offload host tensor operations to device
    swa_full: bool,    // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
                       // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
                       //       ref: https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573
    kv_unified: bool,  // use a unified buffer across the input sequences when computing the attention
                       // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
                       // ref: https://github.com/ggml-org/llama.cpp/pull/14363

    // [EXPERIMENTAL]
    // backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
    // note: the samplers must be sampler chains (i.e. use llama_sampler_chain_init)
    samplers: [^]rawptr, // [^]llama_sampler_seq_config,
    n_samplers: c.size_t,
};

Chat_Message :: struct {
    role: cstring,
    content: cstring
}
