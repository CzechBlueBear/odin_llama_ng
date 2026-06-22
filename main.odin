package llama_odin_ng

import "core:strings"
import "core:dynlib"
import "core:os"
import "core:fmt"
import "core:c"
import "llama"

main :: proc() {
	if !llama.load_library() {
		panic("Could not load the libllama dynamic library")
	}

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

	llama.backend_init()
	defer llama.backend_free()

	tensor_split := make([]f32, 2)
	tensor_split[0] = 1.0
	params := llama.Model_Params {
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

	model := llama.model_load_from_file(model_path, params)
	if model == nil {
		fmt.eprintln("Could not load llama model")
		return
	}

	vocab := llama.model_get_vocab(model)
	if vocab == nil {
		fmt.eprintln("Could not obtain model vocabulary")
		return
	}

    // initialize the context
    ctx_params := llama.context_default_params()
	n_ctx: c.uint32_t = 2048	// context size; TODO: determined from model
    ctx_params.n_ctx = n_ctx
    ctx_params.n_batch = n_ctx

    ctx := llama.init_from_model(model, ctx_params)
    if ctx == nil {
		fmt.eprintln("Failed to create context for model (model unsupported?)")
		return
    }

    // initialize the sampler chain
    sampler := llama.sampler_chain_init(llama.sampler_chain_default_params())
    llama.sampler_chain_add(sampler, llama.sampler_init_min_p(0.05, 1))
    llama.sampler_chain_add(sampler, llama.sampler_init_temp(0.8))
    llama.sampler_chain_add(sampler, llama.sampler_init_dist(llama.DEFAULT_SEED))

    // allocate space and tokenize the prompt (first call to tokenize() determines the number of tokens,
    // the second call does the full tokenization)
    prompt_length := i32(len(prompt))
    is_first := true
	n_prompt_tokens := -llama.tokenize(vocab, prompt, prompt_length, nil, 0, is_first, true)
	prompt_tokens := make([]llama.Token, n_prompt_tokens)
	defer delete(prompt_tokens)
	if llama.tokenize(vocab, prompt, prompt_length, &prompt_tokens[0], n_prompt_tokens, is_first, true) < 0 {
		fmt.eprintln("Failed to tokenize prompt")
		return
	}

    // prepare a batch for the prompt, then decode tokens until the end of generation
    batch := llama.batch_get_one(&prompt_tokens[0], i32(len(prompt_tokens)));
    new_token_id: llama.Token
    response := strings.builder_make()
    for {

        // check if we have enough space in the context to evaluate this batch
        n_ctx := llama.n_ctx(ctx);
        n_ctx_used := llama.memory_seq_pos_max(llama.get_memory(ctx), 0) + 1;
        if n_ctx_used + llama.llama_pos(batch.n_tokens) > llama.llama_pos(n_ctx) {
            fmt.eprintln("Context size exceeded")
            return
        }

        ret := llama.decode(ctx, batch);
        if ret != 0 {
        	fmt.eprintln("Failed decoding model reply")
        	return
        }

        // sample the next token
        new_token_id := llama.sampler_sample(sampler, ctx, -1);

        // is it an end of generation?
        if llama.vocab_is_eog(vocab, new_token_id) {
            break;
        }

        // convert the token to a string, print it and add it to the response
		token_text, ok := llama.token_to_string(vocab, new_token_id)
		if !ok {
			fmt.eprintfln("Error decoding token")
			return;
		}
        fmt.printf("%s", token_text)
        strings.write_string(&response, token_text);
        delete(token_text)

        // prepare the next batch with the sampled token
        batch = llama.batch_get_one(&new_token_id, 1);
    }

}
