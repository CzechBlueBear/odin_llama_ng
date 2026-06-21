package llama_odin_ng

import "core:strings"
import "core:dynlib"
import "core:os"
import "core:fmt"
import "core:c"

main :: proc() {
	if !load_llama_library() {
		panic("Could not load the libllama dynamic library")
	}

	// p_llama_backend_init    := cast(proc "c" ()) load_necessary_symbol(libllama, "llama_backend_init")
	// p_llama_backend_free    := cast(proc "c" ()) load_necessary_symbol(libllama, "llama_backend_free")
	// p_llama_model_load_from_file := cast(proc "c" (model_path: cstring, model_params: llama_model_params) -> llama_model_ptr) load_necessary_symbol(libllama, "llama_model_load_from_file")
	// p_llama_model_get_vocab := cast(proc "c" (model: llama_model_ptr) -> llama_vocab_ptr) load_necessary_symbol(libllama, "llama_model_get_vocab")
	// p_llama_init_from_model := cast(proc "c" (model: llama_model_ptr, params: LLAMA_Context_Params) -> llama_context_ptr) load_necessary_symbol(libllama, "llama_init_from_model")
	// p_llama_context_default_params := cast(proc "c" () -> LLAMA_Context_Params)           load_necessary_symbol(libllama, "llama_context_default_params")

    // // tokenizing and decoding
	// p_llama_tokenize        := cast(proc "c" (
	// 	vocab: llama_vocab_ptr, text: cstring, text_len: c.int32_t,
	// 	tokens: [^]Token, n_tokens_max: c.int32_t,
	// 	add_special: bool, parse_special: bool) -> c.int32_t)                                        load_necessary_symbol(libllama, "llama_tokenize")
	// p_llama_batch_get_one   := cast(proc "c" (tokens: [^]Token, n_tokens: c.int32_t) -> llama_batch) load_necessary_symbol(libllama, "llama_batch_get_one")
	// p_llama_n_ctx           := cast(proc "c" (ctx: llama_context_ptr) -> c.uint32_t)                 load_necessary_symbol(libllama, "llama_n_ctx")
	// p_llama_get_memory      := cast(proc "c" (ctx: llama_context_ptr) -> llama_memory_ptr)           load_necessary_symbol(libllama, "llama_get_memory")
	// p_llama_memory_seq_pos_max := cast(proc "c" (mem: llama_memory_ptr, seq_id: llama_seq_id) -> llama_pos) load_necessary_symbol(libllama, "llama_memory_seq_pos_max")
	// p_llama_decode          := cast(proc "c" (ctx: llama_context_ptr, batch: llama_batch) -> c.int32_t) load_necessary_symbol(libllama, "llama_decode")
	// p_llama_sampler_sample  := cast(proc "c" (smpl: llama_sampler_ptr, ctx: llama_context_ptr, idx: c.int32_t) -> Token) load_necessary_symbol(libllama, "llama_sampler_sample")
	// p_llama_vocab_is_eog    := cast(proc "c" (vocab: llama_vocab_ptr, token: Token) -> bool)         load_necessary_symbol(libllama, "llama_vocab_is_eog")
	// p_llama_token_to_piece  := cast(proc "c" (vocab: llama_vocab_ptr, token: Token, buf: ^c.char, length: c.int32_t, lstrip: c.int32_t, special: bool) -> c.int32_t) load_necessary_symbol(libllama, "llama_token_to_piece")

	// // operations with chained samplers
	// p_llama_sampler_chain_default_params := cast(proc "c" () -> llama_sampler_chain_params)          load_necessary_symbol(libllama, "llama_sampler_chain_default_params")
	// p_llama_sampler_chain_init := cast(proc "c" (params: llama_sampler_chain_params) -> llama_sampler_ptr) load_necessary_symbol(libllama, "llama_sampler_chain_init")
	// p_llama_sampler_init_min_p := cast(proc "c" (min_p: c.float, min_keep: c.size_t) -> llama_sampler_ptr) load_necessary_symbol(libllama, "llama_sampler_init_min_p")
	// p_llama_sampler_init_temp := cast(proc "c" (value: c.float) -> llama_sampler_ptr)                load_necessary_symbol(libllama, "llama_sampler_init_temp")
	// p_llama_sampler_init_dist := cast(proc "c" (seed: c.uint32_t) -> llama_sampler_ptr)              load_necessary_symbol(libllama, "llama_sampler_init_dist")
	// p_llama_sampler_chain_add := cast(proc "c" (chain: llama_sampler_ptr, sampler: llama_sampler_ptr)) load_necessary_symbol(libllama, "llama_sampler_chain_add")

	args := os.args
	//defer delete(args)

	model_path: cstring = nil
	prompt: cstring = nil
	for i in 1 ..< len(args) {
		if model_path == nil {
			model_path = strings.clone_to_cstring(args[i])
		}
		else if prompt == nil {
			prompt = strings.clone_to_cstring(args[i])
		}
	}

	if model_path == nil {
		fmt.eprintln("Path to a LLM model must be specified")
		return
	}
	defer delete(model_path)

	if prompt == nil {
		fmt.eprintln("Prompt for the LLM must be specified")
		return
	}
	defer delete(prompt)

	p_llama_backend_init()
	defer p_llama_backend_free()

	tensor_split := make([]f32, 2)
	tensor_split[0] = 1.0
	params := llama_model_params {
		devices = nil,
		tensor_buft_overrides = nil,
		n_gpu_layers = -1,
		split_mode = .NONE,
		main_gpu = 0,
		tensor_split = cast([^]f32)(&tensor_split),
		progress_callback = nil,
		progress_callback_user_data = nil,
		kv_overrides = nil,
		use_mmap = true
	}

	model := p_llama_model_load_from_file(model_path, params)
	if model == nil {
		fmt.eprintln("Could not load llama model")
		return
	}

	vocab := p_llama_model_get_vocab(model)
	if vocab == nil {
		fmt.eprintln("Could not obtain model vocabulary")
		return
	}

    // initialize the context
    ctx_params := p_llama_context_default_params()
	n_ctx: c.uint32_t = 2048	// context size; TODO: determined from model
    ctx_params.n_ctx = n_ctx
    ctx_params.n_batch = n_ctx

    ctx := p_llama_init_from_model(model, ctx_params)
    if ctx == nil {
		fmt.eprintln("Failed to create the llama_context")
		return
    }

    // initialize the sampler
    sampler := p_llama_sampler_chain_init(p_llama_sampler_chain_default_params())
    p_llama_sampler_chain_add(sampler, p_llama_sampler_init_min_p(0.05, 1))
    p_llama_sampler_chain_add(sampler, p_llama_sampler_init_temp(0.8))
    p_llama_sampler_chain_add(sampler, p_llama_sampler_init_dist(LLAMA_DEFAULT_SEED))

    // allocate space and tokenize the prompt
    prompt_length := i32(len(prompt))
    is_first := true
	n_prompt_tokens := -p_llama_tokenize(vocab, prompt, prompt_length, nil, 0, is_first, true)
	prompt_tokens := make([]Token, n_prompt_tokens)
	defer delete(prompt_tokens)
	if p_llama_tokenize(vocab, prompt, prompt_length, &prompt_tokens[0], n_prompt_tokens, is_first, true) < 0 {
		fmt.eprintln("Failed to tokenize prompt")
		return
	}

    // prepare a batch for the prompt, then decode tokens until the end of generation
    batch := p_llama_batch_get_one(&prompt_tokens[0], i32(len(prompt_tokens)));
    new_token_id: Token
    response := strings.builder_make()
    for {

        // check if we have enough space in the context to evaluate this batch
        n_ctx := p_llama_n_ctx(ctx);
        n_ctx_used := p_llama_memory_seq_pos_max(p_llama_get_memory(ctx), 0) + 1;
        if n_ctx_used + llama_pos(batch.n_tokens) > llama_pos(n_ctx) {
            fmt.eprintln("Context size exceeded")
            return
        }

        ret := p_llama_decode(ctx, batch);
        if ret != 0 {
        	fmt.eprintln("Failed decoding model reply")
        	return
        }

        // sample the next token
        new_token_id := p_llama_sampler_sample(sampler, ctx, -1);

        // is it an end of generation?
        if p_llama_vocab_is_eog(vocab, new_token_id) {
            break;
        }

        // convert the token to a string, print it and add it to the response
		token_text, ok := token_to_string(vocab, new_token_id)
		if !ok {
			fmt.eprintfln("Error decoding token")
			return;
		}
        fmt.printf("%s", token_text)
        strings.write_string(&response, token_text);
        delete(token_text)

        // prepare the next batch with the sampled token
        batch = p_llama_batch_get_one(&new_token_id, 1);
    }

}
