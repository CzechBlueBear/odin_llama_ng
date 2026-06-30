package llama_odin_ng

import "core:strings"
import "core:dynlib"
import "core:os"
import "core:fmt"
import "core:c"
import "core:log"
import "core:io"
import "base:runtime"
import "llama"

Client_State :: struct {
	model_path: cstring,
	prompt: cstring,
	history: [dynamic]llama.Chat_Message,
	model: llama.llama_model_ptr,
	vocab: llama.llama_vocab_ptr,
	ctx: llama.llama_context_ptr,
	sampler: llama.llama_sampler_ptr,
	batch: llama.llama_batch,
}

deinit_client_state :: proc (state: Client_State) {
}

main :: proc() {
	if !llama.load_library() {
		fmt.eprintln("Could not load the llama.cpp dynamic library (not installed?)")
		return
	}
	readline_init()

	log.debug("llama.load_library() ok")

	state: Client_State
	defer deinit_client_state(state)

	// get model name and prompt from the command line
	args := os.args
	for i in 1 ..< len(args) {
		if len(state.model_path) == 0 {
			state.model_path = strings.unsafe_string_to_cstring(args[i])
		}
		else if len(state.prompt) == 0 {
			state.prompt = strings.unsafe_string_to_cstring(args[i])
		}
	}

	if len(state.model_path) == 0 {
		fmt.eprintln("Path to a LLM model must be specified (as argument #1)")
		return
	}

	if len(state.prompt) == 0 {
		fmt.eprintln("Prompt for the LLM must be specified (as argument #2)")
		return
	}

	llama.backend_init()
	defer llama.backend_free()

	log.debug("llama.backend_init() ok")

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

	state.model = llama.model_load_from_file(state.model_path, params)
	if state.model == nil {
		fmt.eprintfln("Could not load llama model (bad path?)", state.model_path)
		return
	}

	// get vocabulary from the model
	state.vocab = llama.model_get_vocab(state.model)
	if state.vocab == nil {
		fmt.eprintln("Could not obtain model vocabulary (unsupported model?)")
		return
	}

    // initialize the context
    ctx_params := llama.context_default_params()
	n_ctx: c.uint32_t = 2048	// context size; TODO: determined from model
    ctx_params.n_ctx = n_ctx
    ctx_params.n_batch = n_ctx

    state.ctx = llama.init_from_model(state.model, ctx_params)
    if state.ctx == nil {
		fmt.eprintln("Failed to create context for model (unsupported model?)")
		return
    }

    // initialize the sampler chain
    state.sampler = llama.sampler_chain_init(llama.sampler_chain_default_params())
    llama.sampler_chain_add(state.sampler, llama.sampler_init_min_p(0.05, 1))
    llama.sampler_chain_add(state.sampler, llama.sampler_init_temp(0.8))
    llama.sampler_chain_add(state.sampler, llama.sampler_init_dist(llama.DEFAULT_SEED))

    // prepare the chat template
	chat_template := llama.model_chat_template(state.model, nil)
	if chat_template == nil {
		fmt.eprintln("Failed to obtain chat template for model (unsupported model?)")
		return
	}

	state.history = make([dynamic]llama.Chat_Message, 0, 8)
	append_cstring_to_chat_history(&state, "prompt", state.prompt)

    is_first := true
	for {
		prompt_tokens := llama.tokenize(state.vocab, state.prompt, is_first, true)
		defer delete(prompt_tokens)

	    // prepare a batch for decoding
	    state.batch = llama.batch_get_one(&prompt_tokens[0], i32(len(prompt_tokens)));
	    response := strings.builder_make()
	    for {

	        // check if we have enough space in the context to evaluate this batch
	        n_ctx := llama.n_ctx(state.ctx);
	        n_ctx_used := llama.memory_seq_pos_max(llama.get_memory(state.ctx), 0) + 1;
	        if n_ctx_used + llama.llama_pos(state.batch.n_tokens) > llama.llama_pos(n_ctx) {
	            fmt.eprintfln("Context size exceeded (would need %d, currently %d in use)", n_ctx_used, n_ctx)
	            return
	        }

	        ret := llama.decode(state.ctx, state.batch);
	        if ret != 0 {
	        	fmt.eprintln("Failed decoding model reply")
	        	return
	        }

	        // sample the next token
	        new_token_id := llama.sampler_sample(state.sampler, state.ctx, -1);

	        // is it an end of generation?
	        if llama.vocab_is_eog(state.vocab, new_token_id) {
	            break;
	        }

	        // convert the token to a string, print it and add it to the response
			token_text, ok := llama.token_to_string(state.vocab, new_token_id)
			if !ok {
				fmt.eprintfln("Error decoding token")
				return;
			}
	        fmt.printf("%s", token_text)
	        strings.write_string(&response, token_text);
	        delete(token_text)

	        // prepare the next batch with the sampled token
	        state.batch = llama.batch_get_one(&new_token_id, 1);
	    }

	    append_message_to_chat_history(&state, "ai", strings.to_string(response))
	    strings.builder_destroy(&response)

	    user_input := read_line("user: ")
	    if len(user_input) == 0 {
	    	return
	    }

	    append_message_to_chat_history(&state, "user", string(user_input))

	    state.prompt = llama.format_messages(chat_template, state.history[:])
	}
}

append_message_to_chat_history :: proc(state: ^Client_State, role: string, content: string) {
	append_elem(&state.history, llama.Chat_Message {
		role = strings.clone_to_cstring(role),
		content = strings.clone_to_cstring(content)
	})
}

append_cstring_to_chat_history :: proc(state: ^Client_State, role: cstring, content: cstring) {
	append_elem(&state.history, llama.Chat_Message {
		role = clone_cstring(role),
		content = clone_cstring(content)
	})
}

/// Makes a dynamically allocated copy of the given C-style string.
clone_cstring :: proc(s: cstring) -> cstring {
	length := len(s)
	mem, err := runtime.mem_alloc(length + 1)
	if err != nil {
		panic("Out of memory")
	}
	runtime.mem_copy(rawptr(&mem[0]), rawptr(s), length)
	return cstring(&mem[0])
}
