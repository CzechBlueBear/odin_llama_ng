package llama_odin_ng

import "core:strings"
import "core:dynlib"
import "core:os"
import "core:fmt"
import "core:c"
import "core:log"
import "core:io"
import "core:terminal/ansi"
import "base:runtime"
import "llama"

Client_State :: struct {
	model_path: string,
	prompt: string,
	history: [dynamic]llama.Chat_Message,
	model: llama.llama_model_ptr,
	vocab: llama.llama_vocab_ptr,
	ctx: llama.llama_context_ptr,
	sampler: ^llama.Sampler,
	batch: llama.Batch,
	chat_template: cstring,
}

deinit_client_state :: proc (state: Client_State) {
}

main :: proc() {
	if !llama.load_library() {
		fmt.eprintln("Could not load the llama.cpp dynamic library (not installed?)")
		return
	}
	//readline_init()	// FIXME: crashes later in readline() call

	log.debug("llama.load_library() ok")

	state: Client_State
	defer deinit_client_state(state)

	// get model name and prompt from the command line
	args := os.args
	for i in 1 ..< len(args) {
		if len(state.model_path) == 0 {
			state.model_path = args[i]
		}
		else if len(state.prompt) == 0 {
			state.prompt = strings.clone(args[i])
		}
	}

	if len(state.model_path) == 0 {
		fmt.eprintln("Path to a LLM model must be specified (as argument #1)")
		return
	}

	if len(state.prompt) == 0 {
		// permissible; we will ask for prompt later
	}

	llama.backend_init()
	defer llama.backend_free()

	log.debug("llama.backend_init() ok")

	if !load_model(&state, state.model_path) {
		fmt.eprintfln("Could not load model: %s", state.model_path)
		return
	}

	if len(state.prompt) == 0 {
		state.prompt = read_line("prompt: ")
		if len(state.prompt) == 0 {
			fmt.eprintln("Prompt must be specified")
			return
		}
	}

	// store the initial prompt (just "prompt" in LLM user parlance)
	append_message_to_chat_history(&state, "prompt", state.prompt)

	response := strings.builder_make()
    is_first := true
	for {

		// convert prompt into tokens
		prompt_tokens := llama.tokenize(state.vocab, state.prompt, is_first, true)
		defer delete(prompt_tokens)

		is_first = false

		// prepare a batch for response generation
		state.batch = llama.batch_get_one(&prompt_tokens[0], i32(len(prompt_tokens)));

		// check if we have enough space in the context to evaluate this batch
		// note that the context contains the prompt and the response is written after it
		// so in order to fit, it must accomodate both

		// maximum capacity of the context
		n_ctx := llama.n_ctx(state.ctx)

        // how many tokens we consume for the prompt
        n_ctx_used := llama.memory_seq_pos_max(llama.get_memory(state.ctx), 0) + llama.llama_pos(len(prompt_tokens));

        if i32(n_ctx_used) + i32(state.batch.n_tokens) > i32(n_ctx) {
            fmt.eprintfln("Context size exceeded (would need %d, currently %d in use)", n_ctx_used, n_ctx)
            return
        }

	    // evaluate the batch by calling decode(), each call generates a token
	    for {

			// sample from the logits of the last token in the batch
	        ret := llama.decode(state.ctx, state.batch)
	        if ret != 0 {
	        	fmt.eprintln("Failed decoding model reply")
	        	return
	        }

	        // sample the next token
	        new_token_id := llama.sampler_sample(state.sampler, state.ctx, -1)

	        // is it an end of generation?
	        if llama.vocab_is_eog(state.vocab, new_token_id) {
	            break
	        }

	        // convert the token to a string, print it and add it to the response
			token_text, ok := llama.token_to_string(state.vocab, new_token_id)
			if !ok {
				fmt.eprintfln("Error decoding token")
				break
			}

			// pending message
			print_thinking_spinner(&state, len(prompt_tokens))

			// append the token to the response
	        strings.write_string(&response, token_text);
	        delete(token_text)

	        // prepare the next batch with the sampled token
	        state.batch = llama.batch_get_one(&new_token_id, 1);
	    }

	    if len(response.buf) > 0 {
		    fmt.println(strings.to_string(response))
			append_message_to_chat_history(&state, "ai", strings.to_string(response))
			strings.builder_reset(&response)
		}

		user_input: string
		command_loop:
		for {

			// get user input (either another prompt, or a command)
			user_input = read_line(">> ")

			// commands start with '/'
			if strings.starts_with(user_input, "/") {
				if strings.starts_with(user_input, "/quit") {
					return
				}
				else if strings.starts_with(user_input, "/record") {
					write_chat_history_to_file("./chat_history.txt", state.history)
					continue command_loop
				}
				else {
					fmt.eprintln("Unrecognized command")
				}
			}
			else {

				// everything else is an input for the AI

				if state.model == nil {
					fmt.eprintln("No model loaded")
					continue command_loop
				}
				break command_loop
			}
		}

		append_message_to_chat_history(&state, "user", string(user_input))

		// take everything in the current conversation, reformat it according to
		// model's recommended format, and prepare for the next round
		state.prompt = llama.format_messages(state.chat_template, state.history[:])
	}
}

append_message_to_chat_history :: proc(state: ^Client_State, role: string, content: string) {
	if strings.contains_rune(content, 0) {
		fmt.println("Warning: recorded output contains NUL character, may get truncated")
	}
	new_message := llama.Chat_Message {
		role = strings.clone_to_cstring(role),
		content = strings.clone_to_cstring(content)
	}
	append_elem(&state.history, new_message)
	fmt.printfln("Appended to history: \"%s\":\"%s\"", new_message.role, new_message.content)
}

write_chat_history_to_file :: proc(path: string, history: [dynamic]llama.Chat_Message) {
	f, ferr := os.open(path, os.File_Flags{.Write, .Append, .Create})
	if ferr != nil {
		fmt.eprintfln("Could not open file for writing: %s", os.error_string(ferr))
		return
	}
	defer os.close(f)

	stream: io.Stream = os.to_stream(f)
	for msg in history {
		write_cstring(stream, msg.role)
		io.write_string(stream, ":")
		write_cstring(stream, msg.content)
		io.write_string(stream, "\n")
	}
}

write_cstring :: proc(stream: io.Stream, text: cstring) -> io.Error {

	input := strings.clone_from_cstring(text)
	defer delete(input)

	bytes_written, err := io.write_string(stream, input)
	if err != .None {
		return err
	}
	return .None
}

print_thinking_spinner :: proc(state: ^Client_State, token_count: int) {
	n_ctx := llama.n_ctx(state.ctx)
	n_ctx_used := int(llama.memory_seq_pos_max(llama.get_memory(state.ctx), 0)) + token_count
	fmt.printf(
		ansi.CSI + ansi.FG_BLUE + ansi.SGR +
		"Thinking... %d(+%d) tokens used/%d\r" +
		ansi.CSI + ansi.RESET + ansi.SGR,
		n_ctx_used, state.batch.n_tokens, n_ctx)
}

unload_model :: proc(state: ^Client_State) {
	if state.model != nil {
		llama.model_free(state.model)
		state.model = nil
		state.vocab = nil
		llama.batch_free(state.batch)
	}
}

load_model :: proc(state: ^Client_State, model_path: string) -> bool {

	// unload previous model, if any
	unload_model(state)

	params := llama.model_default_params()

	// load the model
	model_path_cstring := strings.clone_to_cstring(state.model_path)
	defer free(rawptr(model_path_cstring))
	state.model_path = model_path
	state.model = llama.model_load_from_file(model_path_cstring, params)
	if state.model == nil {
		fmt.eprintfln("Could not load llama model (bad path?): %s", state.model_path)
		return false
	}

	// get vocabulary from the model
	state.vocab = llama.model_get_vocab(state.model)
	if state.vocab == nil {
		fmt.eprintln("Could not obtain model vocabulary (unsupported model?)")
		return false
	}

    // initialize the context
    ctx_params := llama.context_default_params()
	n_ctx: c.uint32_t = 2048	// context size; TODO: determined from model
    ctx_params.n_ctx = n_ctx
    ctx_params.n_batch = n_ctx

    state.ctx = llama.init_from_model(state.model, ctx_params)
    if state.ctx == nil {
		fmt.eprintln("Failed to create context for model (unsupported model?)")
		return false
    }

    // initialize the sampler chain
    state.sampler = llama.sampler_chain_init(llama.sampler_chain_default_params())
    llama.sampler_chain_add(state.sampler, llama.sampler_init_min_p(0.05, 1))
    llama.sampler_chain_add(state.sampler, llama.sampler_init_temp(0.8))
    llama.sampler_chain_add(state.sampler, llama.sampler_init_dist(llama.DEFAULT_SEED))

    // prepare the chat template
	state.chat_template = llama.model_chat_template(state.model, nil)
	if state.chat_template == nil {
		fmt.eprintln("Failed to obtain chat template for model (unsupported model?)")
		return false
	}

	return true
}
