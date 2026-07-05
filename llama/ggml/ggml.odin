package ggml

import "core:c"

MAX_SRC :: 10
MAX_NAME :: 64
MAX_DIMS :: 4
MAX_OP_PARAMS :: 64

Type :: enum c.int32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    // Q4_2 = 4, support has been removed
    // Q4_3 = 5, support has been removed
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    // Q4_0_4_4 = 31, support has been removed from gguf files
    // Q4_0_4_8 = 32,
    // Q4_0_8_8 = 33,
    TQ1_0   = 34,
    TQ2_0   = 35,
    // IQ4_NL_4_4 = 36,
    // IQ4_NL_4_8 = 37,
    // IQ4_NL_8_8 = 38,
    MXFP4   = 39, // MXFP4 (1 block)
    NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
    Q1_0    = 41,
}

Context :: struct {
	mem_size:   c.size_t,
	mem_buffer: rawptr,
	mem_buffer_owned:   bool,
	no_alloc:   bool,
	n_objects:  c.int,
	objects_begin:    rawptr, // struct ggml_object*
	objects_end:      rawptr, // struct ggml_object*
}

Op :: enum c.int {
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

// numa strategies
NUMA_Strategy :: enum c.int {
	Disabled = 0,
	Distribute = 1,
	Isolate = 2,
	Numactl = 3,
	Mirror = 4,
	COUNT,
}

// n-dimensional tensor
Tensor :: struct {
	type: Type,
	buffer: rawptr,            // ^ggml_backend_buffer

    ne: [MAX_DIMS]c.int64_t,	// number of elements
    nb: [MAX_DIMS]c.size_t,	// stride in bytes:
                               // nb[0] = ggml_type_size(type)
                               // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                               // nb[i] = nb[i-1] * ne[i-1]

    // compute data
    op: Op,

    // op params - allocated as int32_t for alignment
    op_params: [MAX_OP_PARAMS / size_of(c.int32_t)]c.int32_t,

    flags: c.int32_t,

    src: [MAX_SRC]^Tensor,

    // source tensor and offset for views
    view_src: ^Tensor,
    view_offs: c.size_t,

    data: rawptr,

    name: [MAX_NAME]c.char,

    extra: rawptr,

    padding: [8]c.char,
};

Cgraph_Eval_Order :: enum c.int {
	Left_To_Right = 0,
	Right_To_Left,
	Count
}

Cgraph :: struct {
	size:    c.int,      // maximum number of nodes/leafs/grads/grad_accs
	n_nodes: c.int,      // number of nodes currently in use
	n_leafs: c.int,      // number of leafs currently in use

	nodes:   ^^Tensor,   // tensors with data that can change if the graph is evaluated
	grads:   ^^Tensor,   // the outputs of these tensors are the gradients of the nodes
	grad_accs: ^^Tensor, // accumulators for node gradients
	leafs:   ^^Tensor,   // tensors with constant data
	use_counts: c.int32_t, // number of uses of each tensor, indexed by hash table slot

	visited_hash_set:  rawptr, //^GGML_Hash_Set,
	order:    Cgraph_Eval_Order,

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
	uid: c.uint64_t,
}
