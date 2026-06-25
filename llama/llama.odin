package llama

import "core:c"
import "core:dynlib"
import "core:fmt"
import "core:strings"

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
init_from_model: proc "c" (model: llama_model_ptr, params: LLAMA_Context_Params) -> llama_context_ptr
context_default_params: proc "c" () -> LLAMA_Context_Params
token_to_piece: proc "c" (vocab: llama_vocab_ptr, token: Token, buf: ^c.char, length: c.int32_t, lstrip: c.int32_t, special: bool) -> c.int32_t
tokenize: proc "c" (vocab: llama_vocab_ptr, text: cstring, text_len: c.int32_t, tokens: [^]Token, n_tokens_max: c.int32_t, add_special: bool, parse_special: bool) -> c.int32_t

batch_get_one   : proc "c" (tokens: [^]Token, n_tokens: c.int32_t) -> llama_batch
n_ctx           : proc "c" (ctx: llama_context_ptr) -> c.uint32_t
get_memory      : proc "c" (ctx: llama_context_ptr) -> llama_memory_ptr
memory_seq_pos_max : proc "c" (mem: llama_memory_ptr, seq_id: llama_seq_id) -> llama_pos
decode          : proc "c" (ctx: llama_context_ptr, batch: llama_batch) -> c.int32_t
sampler_sample  : proc "c" (smpl: llama_sampler_ptr, ctx: llama_context_ptr, idx: c.int32_t) -> Token
vocab_is_eog    : proc "c" (vocab: llama_vocab_ptr, token: Token) -> bool
sampler_chain_default_params : proc "c" () -> llama_sampler_chain_params
sampler_chain_init : proc "c" (params: llama_sampler_chain_params) -> llama_sampler_ptr
sampler_init_min_p : proc "c" (min_p: c.float, min_keep: c.size_t) -> llama_sampler_ptr
sampler_init_temp : proc "c" (value: c.float) -> llama_sampler_ptr
sampler_init_dist : proc "c" (seed: c.uint32_t) -> llama_sampler_ptr
sampler_chain_add : proc "c" (chain: llama_sampler_ptr, sampler: llama_sampler_ptr)

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
	set_proc_address(&tokenize, "llama_tokenize")
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
	buf: [dynamic]u8 = nil
	new_len := chat_apply_template(tmpl, &messages[0], len(messages), true, nil, 0);
	if new_len > c.int32_t(len(buf)) {
		buf = make_dynamic_array([dynamic]u8)
		defer delete(buf)
		new_len = chat_apply_template(tmpl, &messages[0], len(messages), true, &buf[0], i32(len(buf)));
		if new_len < 0 {
			panic("Could not apply chat template (out of memory?)")
		}
	}

	return strings.clone_from_bytes(buf[:])
}