package llama

import "core:c"
import "core:dynlib"
import "core:fmt"
import "core:strings"
import "base:runtime"

LLAMA_DYNLIB_PATH :: "/usr/local/lib/libllama.so"

libllama: dynlib.Library

load_necessary_symbol :: proc(lib: dynlib.Library, symbol: string) -> rawptr {
	address, ok := dynlib.symbol_address(lib, symbol)
	if !ok {
		fmt.eprintfln("Missing necessary symbol: %s", symbol)
		panic("Error resolving dynamic symbols (mismatched libllama version?)")
	}
	return address
}

set_proc_address :: proc(p: rawptr, name: string) {
	(cast(^rawptr)p)^ = load_necessary_symbol(libllama, name)
}

// pointers to llama API functions - symbols to be dynamically loaded
backend_init: proc "c" ();
backend_free: proc "c" ();
model_load_from_file: proc "c" (model_path: cstring, model_params: Model_Params) -> llama_model_ptr
model_get_vocab: proc "c" (model: llama_model_ptr) -> llama_vocab_ptr
init_from_model: proc "c" (model: llama_model_ptr, params: Context_Params) -> llama_context_ptr
context_default_params: proc "c" () -> Context_Params
token_to_piece: proc "c" (vocab: llama_vocab_ptr, token: Token, buf: ^c.char, length: c.int32_t, lstrip: c.int32_t, special: bool) -> c.int32_t
tokenize_raw: proc "c" (vocab: llama_vocab_ptr, text: cstring, text_len: c.int32_t, tokens: [^]Token, n_tokens_max: c.int32_t, add_special: bool, parse_special: bool) -> c.int32_t

batch_get_one   : proc "c" (tokens: [^]Token, n_tokens: c.int32_t) -> Batch
n_ctx           : proc "c" (ctx: llama_context_ptr) -> c.uint32_t
get_memory      : proc "c" (ctx: llama_context_ptr) -> llama_memory_ptr
memory_seq_pos_max : proc "c" (mem: llama_memory_ptr, seq_id: llama_seq_id) -> llama_pos
decode          : proc "c" (ctx: llama_context_ptr, batch: Batch) -> c.int32_t
sampler_sample  : proc "c" (smpl: ^Sampler, ctx: llama_context_ptr, idx: c.int32_t) -> Token
vocab_is_eog    : proc "c" (vocab: llama_vocab_ptr, token: Token) -> bool

sampler_chain_default_params : proc "c" () -> llama_sampler_chain_params
sampler_chain_init : proc "c" (params: llama_sampler_chain_params) -> ^Sampler
sampler_init_min_p : proc "c" (min_p: c.float, min_keep: c.size_t) -> ^Sampler
sampler_init_temp : proc "c" (value: c.float) -> ^Sampler
sampler_init_dist : proc "c" (seed: c.uint32_t) -> ^Sampler
sampler_chain_add : proc "c" (chain: ^Sampler, sampler: ^Sampler)

chat_apply_template : proc "c" (tmpl: cstring, chat: ^Chat_Message, n_msg: c.size_t, add_assistant: bool, buf: ^u8, length: c.int32_t) -> c.int32_t
model_chat_template : proc "c" (model: llama_model_ptr, name: cstring) -> cstring

load_library :: proc () -> bool {
	ok: bool
	libllama, ok = dynlib.load_library(LLAMA_DYNLIB_PATH)
	if !ok {
		return false
	}

	set_proc_address(&backend_init, "llama_backend_init")
	set_proc_address(&backend_free, "llama_backend_free")
	set_proc_address(&model_load_from_file, "llama_model_load_from_file")
	set_proc_address(&model_get_vocab, "llama_model_get_vocab")
	set_proc_address(&init_from_model, "llama_init_from_model")
	set_proc_address(&context_default_params, "llama_context_default_params")
	set_proc_address(&token_to_piece, "llama_token_to_piece")
	set_proc_address(&tokenize_raw, "llama_tokenize")
	set_proc_address(&batch_get_one, "llama_batch_get_one")
	set_proc_address(&n_ctx, "llama_n_ctx")
	set_proc_address(&get_memory, "llama_get_memory")
	set_proc_address(&memory_seq_pos_max, "llama_memory_seq_pos_max")
	set_proc_address(&decode, "llama_decode")
	set_proc_address(&sampler_sample, "llama_sampler_sample")
	set_proc_address(&vocab_is_eog, "llama_vocab_is_eog")
	set_proc_address(&sampler_chain_default_params, "llama_sampler_chain_default_params")
	set_proc_address(&sampler_chain_init, "llama_sampler_chain_init")
	set_proc_address(&sampler_init_min_p, "llama_sampler_init_min_p")
	set_proc_address(&sampler_init_temp, "llama_sampler_init_temp")
	set_proc_address(&sampler_init_dist, "llama_sampler_init_dist")
	set_proc_address(&sampler_chain_add, "llama_sampler_chain_add")
	set_proc_address(&chat_apply_template, "llama_chat_apply_template")
	set_proc_address(&model_chat_template, "llama_model_chat_template")

	return true
}

/// Translates a token into a string.
/// The returned string is newly allocated.
token_to_string :: proc (vocab: llama_vocab_ptr, token: Token) -> (string, bool) {

	buf: [256]u8
	n := token_to_piece(vocab, token, &buf[0], len(buf), 0, true)
	if n < 0 {
		fmt.eprintfln("Could not decode token (#%d)", token)
		return "", false
	}

	token_text := strings.clone_from_bytes(buf[:])
	return token_text, true
}

format_messages :: proc (tmpl: cstring, messages: []Chat_Message) -> string
{
	// dry run to see how much space we will need (in bytes), plus 1 for the terminating zero
	needed_length := chat_apply_template(tmpl, &messages[0], len(messages), true, nil, 0);
	buf, err := runtime.mem_alloc(int(needed_length) + 1)
	if err != nil {
		panic("Out of memory when allocating formatting buffer")
	}

	// do the template application in real
	result := chat_apply_template(tmpl, &messages[0], len(messages), true, &buf[0], needed_length);
	if result < 0 {
		panic("Could not apply chat template (out of memory?)")
	}

	// transfer ownership of buf into a new string, and return it
	return string(buf)
}

/// Tokenizes the text using the given vocabulary and returns it as a dynamic array of tokens.
/// Params:
/// vocab - Vocabulary to use (must match the model, naturally).
/// token - The token to decode.
/// add_special - Allow to add BOS and EOS tokens if model is configured to do so.
/// parse_special - Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
///                 as plaintext. Does not insert a leading space.
tokenize :: proc (vocab: llama_vocab_ptr, text: string, add_special: bool, parse_special: bool) -> [dynamic]Token
{
	// convert to cstring; we can't be sure it ends with 0 so we must allocate
	text_cstring := strings.clone_to_cstring(text)
	defer delete(text_cstring)

	text_length := i32(len(text))

	// first call tokenize_raw() in a dry-run mode to determine how many tokens we will need
	tokens_needed := -tokenize_raw(vocab, text_cstring, text_length, nil, 0, add_special, parse_special)
	if tokens_needed <= 0 {
		panic("Failed to tokenize prompt (pass 1; mismatched vocabulary/model?)")
	}

	// allocate sufficient space and do the tokenization for real
	tokens: [dynamic]Token
	resize_dynamic_array(&tokens, tokens_needed)
	result := tokenize_raw(vocab, text_cstring, text_length, raw_data(tokens), i32(len(tokens)), add_special, parse_special)
	if result < 0 {
		panic("Failed to tokenize prompt (pass 2; mismatched vocabulary/model?)")
	}

	return tokens
}
