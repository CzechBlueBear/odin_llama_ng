package llama_odin_ng

import "core:c"

LLAMA_MAX_DEVICES :: 16
LLAMA_DEFAULT_SEED :: 0xFFFFFFFF

GGML_MAX_SRC :: 10
GGML_MAX_NAME :: 64
GGML_MAX_DIMS :: 4
GGML_MAX_OP_PARAMS :: 64

gpt_params_ptr :: distinct rawptr

// Some objects are better considered opaque and only handled with methods
llama_model_params_ptr :: distinct rawptr
llama_model_ptr :: distinct rawptr
llama_context_params_ptr :: distinct rawptr
llama_context_ptr :: distinct rawptr
llama_sampler_ptr :: distinct rawptr
llama_vocab_ptr :: distinct rawptr
llama_memory_ptr :: distinct rawptr  // llama_memory_t, which is (llama_memory_i*)

Token :: distinct c.int32_t
llama_pos :: distinct c.int32_t
llama_seq_id :: distinct c.int32_t

llama_sampler_chain_params :: struct {
	no_perf: c.bool,
}

// Input data for llama_decode
// A llama_batch object can contain input about one or many sequences
// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
//
// - token  : the token ids of the input (used when embd is NULL)
// - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
// - pos    : the positions of the respective token in the sequence
// - seq_id : the sequence to which the respective token belongs
// - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
//
llama_batch :: struct {
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


llama_example :: enum c.int {
	LLAMA_EXAMPLE_COMMON,
	LLAMA_EXAMPLE_SPECULATIVE,
	LLAMA_EXAMPLE_MAIN,
	LLAMA_EXAMPLE_INFILL,
	LLAMA_EXAMPLE_EMBEDDING,
	LLAMA_EXAMPLE_PERPLEXITY,
	LLAMA_EXAMPLE_RETRIEVAL,
	LLAMA_EXAMPLE_PASSKEY,
	LLAMA_EXAMPLE_IMATRIX,
	LLAMA_EXAMPLE_BENCH,
	LLAMA_EXAMPLE_SERVER,
	LLAMA_EXAMPLE_CVECTOR_GENERATOR,
	LLAMA_EXAMPLE_EXPORT_LORA,
	LLAMA_EXAMPLE_LLAVA,
	LLAMA_EXAMPLE_LOOKUP,
	LLAMA_EXAMPLE_PARALLEL,
	LLAMA_EXAMPLE_COUNT,
}

// numa strategies
ggml_numa_strategy :: enum c.int {
	GGML_NUMA_STRATEGY_DISABLED = 0,
	GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
	GGML_NUMA_STRATEGY_ISOLATE = 2,
	GGML_NUMA_STRATEGY_NUMACTL = 3,
	GGML_NUMA_STRATEGY_MIRROR = 4,
	GGML_NUMA_STRATEGY_COUNT,
}

llama_perf_type :: enum c.int {
	LLAMA_PERF_TYPE_CONTEXT       = 0,
	LLAMA_PERF_TYPE_SAMPLER_CHAIN = 1,
}

llama_split_mode :: enum c.int {
    NONE   = 0, // single GPU
    LAYER  = 1, // split layers and KV across GPUs
    ROW    = 2, // split layers and KV across GPUs, use tensor parallelism if supported
    TENSOR = 3,
}

// ggml_backend_device :: struct {
//     struct ggml_backend_device_i iface;
//     ggml_backend_reg_t reg;
//     context: rawptr,
// }

// ggml_backend_dev_t :: ^ggml_device

llama_model_params :: struct {

    // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
    devices: [^]rawptr,		// [^]ggml_backend_dev_t

    // NULL-terminated list of buffer types to use for tensors that match a pattern
    tensor_buft_overrides: [^]rawptr, // [^]llama_model_tensor_buft_override,

    n_gpu_layers: c.int32_t,       // number of layers to store in VRAM, a negative value means all layers
    split_mode: llama_split_mode,  // how to split the model across multiple GPUs

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
    UNSPECIFIED = -1,
    CAUSAL      = 0,
    NON_CAUSAL  = 1,
}

Flash_Attention_Type :: enum c.int32_t {
	AUTO     = -1,
	DISABLED = 0,
	ENABLED  = 1,
}

GGML_Type :: enum c.int32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
    GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
    GGML_TYPE_Q1_0    = 41,
}

GGML_Context :: struct {
	mem_size:   c.size_t,
	mem_buffer: rawptr,
	mem_buffer_owned:   bool,
	no_alloc:   bool,
	n_objects:  c.int,
	objects_begin:    rawptr, // struct ggml_object*
	objects_end:      rawptr, // struct ggml_object*
}

GGML_Op :: enum c.int {
    NONE = 0,
	DUP,
    ADD,
    ADD_ID,
    ADD1,
    ACC,
    SUB,
    MUL,
    DIV,
    SQR,
    SQRT,
    LOG,
    SIN,
    COS,
    SUM,
    SUM_ROWS,
    CUMSUM,
    MEAN,
    ARGMAX,
    COUNT_EQUAL,
    REPEAT,
    REPEAT_BACK,
    CONCAT,
    SILU_BACK,
    NORM, // normalize
    RMS_NORM,
    RMS_NORM_BACK,
    GROUP_NORM,
    L2_NORM,
	MUL_MAT,
    MUL_MAT_ID,
    OUT_PROD,
	SCALE,
    SET,
    CPY,
    CONT,
    RESHAPE,
    VIEW,
    PERMUTE,
    TRANSPOSE,
    GET_ROWS,
    GET_ROWS_BACK,
    SET_ROWS,
    DIAG,
    DIAG_MASK_INF,
    DIAG_MASK_ZERO,
    SOFT_MAX,
    SOFT_MAX_BACK,
    ROPE,
    ROPE_BACK,
    CLAMP,
    CONV_TRANSPOSE_1D,
    IM2COL,
    IM2COL_BACK,
    IM2COL_3D,
    CONV_2D,
    CONV_3D,
    CONV_2D_DW,
    CONV_TRANSPOSE_2D,
    POOL_1D,
    POOL_2D,
    POOL_2D_BACK,
    UPSCALE,
    PAD,
    PAD_REFLECT_1D,
    ROLL,
    ARANGE,
    TIMESTEP_EMBEDDING,
    ARGSORT,
    TOP_K,
    LEAKY_RELU,
    TRI,
    FILL,
	FLASH_ATTN_EXT,
    FLASH_ATTN_BACK,
    SSM_CONV,
    SSM_SCAN,
    WIN_PART,
    WIN_UNPART,
    GET_REL_POS,
    ADD_REL_POS,
    RWKV_WKV6,
    GATED_LINEAR_ATTN,
    RWKV_WKV7,
    SOLVE_TRI,
    GATED_DELTA_NET,

    UNARY,
	MAP_CUSTOM1,
    MAP_CUSTOM2,
    MAP_CUSTOM3,
	CUSTOM,
	CROSS_ENTROPY_LOSS,
    CROSS_ENTROPY_LOSS_BACK,
    OPT_STEP_ADAMW,
    OPT_STEP_SGD,
	GLU,
	COUNT,
}

// n-dimensional tensor
GGML_Tensor :: struct {
	type: GGML_Type,
	buffer: rawptr,            // ^ggml_backend_buffer

    ne: [GGML_MAX_DIMS]c.int64_t,	// number of elements
    nb: [GGML_MAX_DIMS]c.size_t,	// stride in bytes:
                               // nb[0] = ggml_type_size(type)
                               // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                               // nb[i] = nb[i-1] * ne[i-1]

    // compute data
    op: GGML_Op,

    // op params - allocated as int32_t for alignment
    op_params: [GGML_MAX_OP_PARAMS / size_of(c.int32_t)]c.int32_t,

    flags: c.int32_t,

    src: [GGML_MAX_SRC]^GGML_Tensor,

    // source tensor and offset for views
    view_src: ^GGML_Tensor,
    view_offs: c.size_t,

    data: rawptr,

    name: [GGML_MAX_NAME]c.char,

    extra: rawptr,

    padding: [8]c.char,
};

GGML_Cgraph_Eval_Order :: enum c.int {
    LEFT_TO_RIGHT = 0,
    RIGHT_TO_LEFT,
    COUNT
}

GGML_Cgraph :: struct {
	size:    c.int,      // maximum number of nodes/leafs/grads/grad_accs
	n_nodes: c.int,      // number of nodes currently in use
	n_leafs: c.int,      // number of leafs currently in use

	nodes:   ^^GGML_Tensor,   // tensors with data that can change if the graph is evaluated
	grads:   ^^GGML_Tensor,   // the outputs of these tensors are the gradients of the nodes
	grad_accs: ^^GGML_Tensor, // accumulators for node gradients
	leafs:   ^^GGML_Tensor,   // tensors with constant data
	use_counts: c.int32_t, // number of uses of each tensor, indexed by hash table slot

	visited_hash_set:  rawptr, //^GGML_Hash_Set,
	order:    GGML_Cgraph_Eval_Order,

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
	uid: c.uint64_t,
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
	logits:    ^GGML_Tensor,
	probs:     ^GGML_Tensor,
	sampled:   ^GGML_Tensor,
	candidates: ^GGML_Tensor
}

LLAMA_Sampler_Interface :: struct {
	name:	proc "c" (smpl: ^LLAMA_Sampler) -> cstring,                 // can be NULL
	accept: proc "c" (smpl: ^LLAMA_Sampler, token: Token),              // can be NULL
	apply:  proc "c" (smpl: ^LLAMA_Sampler, cur_p: ^Token_Data_Array),  // required
	reset:  proc "c" (smpl: ^LLAMA_Sampler),                            // can be NULL
	clone:  proc "c" (smpl: ^LLAMA_Sampler) -> ^LLAMA_Sampler,          // can be NULL if ctx is NULL
	free:   proc "c" (smpl: ^LLAMA_Sampler),                            // can be NULL if ctx is NULL

    // [EXPERIMENTAL]
    // backend sampling interface:

    // return true if the backend supports all ops needed by the sampler
    // note: call once per sampler
    backend_init:    proc "c" (smpl: ^LLAMA_Sampler, buft: rawptr /* ggml_backend_buffer_type_t */),

    // call after .backend_apply()
    backend_accept:  proc "c" (smpl: ^LLAMA_Sampler, ctx: ^GGML_Context, gf: ^GGML_Cgraph, selected_token: ^GGML_Tensor),

    // call after .backend_init()
    backend_apply:   proc "c" (smpl: ^LLAMA_Sampler, ctx: ^GGML_Context, gf: ^GGML_Cgraph, data: ^Sampler_Data),

    // called before graph execution to set inputs for the current ubatch
    backend_set_input:    proc "c" (smpl: ^LLAMA_Sampler)
}

LLAMA_Sampler :: struct {
    iface: ^LLAMA_Sampler_Interface,
    ctx: rawptr
}

LLAMA_Sampler_Seq_Config :: struct {
	seq_id: llama_seq_id,
	sampler: ^LLAMA_Sampler,
}

LLAMA_Context_Params :: struct {
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

    cb_eval:      proc "c" (tensor: rawptr /* ^ggml_tensor */, ask: bool, user_data: rawptr) -> bool,
    cb_eval_user_data: rawptr,

    type_k: GGML_Type,         // data type for K cache [EXPERIMENTAL]
    type_v: GGML_Type,         // data type for V cache [EXPERIMENTAL]

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

@(default_calling_convention = "c")
foreign {
	// lpp_print_usage :: proc(argc: c.int, argv: ^^c.char) ---

	// lpp_make_gpt_params_ptr :: proc() -> gpt_params_ptr ---

	// lpp_gpt_params_parse :: proc(argc: c.int, argv: ^^c.char, params: gpt_params_ptr, ex: llama_example) -> c.bool ---

	// lpp_get_n_predict :: proc(params: gpt_params_ptr) -> c.int32_t ---
	// lpp_get_n_gpu_layers :: proc(params: gpt_params_ptr) -> c.int32_t ---
	// lpp_get_model :: proc(params: gpt_params_ptr) -> cstring ---
	// lpp_get_prompt :: proc(params: gpt_params_ptr) -> cstring ---

	llama_backend_init :: proc() ---
	llama_backend_free :: proc() ---
	llama_load_model_from_file :: proc(model_path: cstring, model_params: ^llama_model_params) -> llama_model_ptr ---
	llama_free_model :: proc(model: llama_model_ptr) ---

	lpp_get_numa :: proc(params: gpt_params_ptr) -> ggml_numa_strategy ---

	llama_numa_init :: proc(numa: ggml_numa_strategy) ---

	lpp_llama_model_params_from_gpt_params :: proc(params: gpt_params_ptr) -> llama_model_params_ptr ---

	lpp_llama_context_params_from_gpt_params :: proc(params: gpt_params_ptr) -> llama_context_params_ptr ---

	llp_llama_new_context_with_model :: proc(model: llama_model_ptr, ctx_params: llama_context_params_ptr) -> llama_context_ptr ---

	llama_sampler_chain_default_params :: proc() -> llama_sampler_chain_params ---

	llama_sampler_chain_init :: proc(params: llama_sampler_chain_params) -> llama_sampler_ptr ---

	llama_free :: proc(ctx: llama_context_ptr) ---

	llama_sampler_free :: proc(smpl: llama_sampler_ptr) ---

	llama_sampler_init_greedy :: proc() -> llama_sampler_ptr ---
	llama_sampler_chain_add :: proc(chain: llama_sampler_ptr, smpl: llama_sampler_ptr) ---

	lpp_llama_tokenize :: proc(ctx: llama_context_ptr, text: cstring, add_special: c.bool,  /*false*/parse_special: c.bool) -> Tokens ---

	lpp_llama_token_to_piece :: proc(ctx: llama_context_ptr, token: Token,  /*true*/special: c.bool) -> cstring ---

	llama_batch_init :: proc(n_tokens_alloc: c.int32_t, embd: c.int32_t, n_seq_max: c.int32_t) -> llama_batch ---
	llama_batch_free :: proc(batch: llama_batch) ---

	lpp_llama_batch_add :: proc(batch: ^llama_batch, id: Token, pos: llama_pos, seq_ids: [^]llama_seq_id, seq_ids_len: c.size_t, logits: c.bool) ---

	llama_decode :: proc(ctx: llama_context_ptr, batch: llama_batch) -> c.int32_t ---

	ggml_time_us :: proc() -> c.int64_t ---

	llama_n_ctx :: proc(ctx: llama_context_ptr) -> c.int ---

	llama_sampler_sample :: proc(smpl: llama_sampler_ptr, ctx: llama_context_ptr, idx: c.int32_t) -> Token ---

	Token_is_eog :: proc(model: llama_model_ptr, token: Token) -> c.bool ---

	lpp_llama_batch_clear :: proc(batch: ^llama_batch) ---

	llama_perf_print :: proc(ptr: rawptr, perf_type: llama_perf_type) ---
}
