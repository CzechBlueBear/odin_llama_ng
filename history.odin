package llama_odin_ng

import "core:strings"
import "core:os"
import "core:io"
import "core:fmt"
import "core:encoding/json"
import "core:time"
import "llama"

HISTORY_FILE_PREFIX :: "./chat_history"

load_history :: proc(state: ^Client_State, path: string) {
	text, ferr := os.read_entire_file_from_path(path, context.allocator)
	if ferr != nil {
		panic("Could not open history file for reading")
	}

	tree := json.Array {}
	json.unmarshal(text, &tree)

	for item in tree {
		switch v in item {
			case json.Object:

				// extract the message items and load them to the in-memory array
				role := v["role"].(string)
				content := v["content"].(string)
				new_message := llama.Chat_Message {
					role = strings.clone_to_cstring(role),
					content = strings.clone_to_cstring(content)
				}
				append_elem(&state.history, new_message)
			case i64, f64, bool, string, json.Array, json.Null:
				// skip
		}
	}
}

open_history :: proc(state: ^Client_State) {
	t := time.now()
	year, month, day := time.date(t)
	hour, min, sec := time.clock_from_time(t)
	filename := fmt.aprintf("%s-%4d-%s-%2d-%2d%2d.json", HISTORY_FILE_PREFIX, year, month, day, hour, min)
	defer delete(filename)

	f, ferr := os.open(
		filename,
		os.File_Flags{.Write, .Append, .Create},
		os.Permissions{.Read_User, .Write_User, .Read_Group, .Write_Group, .Read_Other}
	)
	if ferr != nil {
		panic("Could not open history file for writing")
	}
	state.history_stream = os.to_stream(f)
}

close_history :: proc(state: ^Client_State) {
	io.close(state.history_stream)
}

append_message_to_chat_history :: proc(state: ^Client_State, role: string, content: string) {

	// add it to the memory-backed history record
	new_message := llama.Chat_Message {
		role = strings.clone_to_cstring(role),
		content = strings.clone_to_cstring(content)
	}
	append_elem(&state.history, new_message)

	// also immediately write it into the file so it's safe
	role_sanitized := sanitize_string_for_json(string(role))
	defer delete(role_sanitized)
	content_sanitized := sanitize_string_for_json(string(content))
	defer delete(content_sanitized)
	fmt.wprintf(state.history_stream, "{{\n   \"role\": \"%s\",\n   \"content\": \"%s\"\n}},\n",
		role_sanitized, content_sanitized)
}

/// Sanitizes the string for use in a JSON string.
/// The result is newly allocated.
sanitize_string_for_json :: proc(str: string) -> string {
	buf := strings.Builder {}

	for r in str {
		if r == '\n' {
			strings.write_string(&buf, "\\n")
		}
		else if r == '"' {
			strings.write_string(&buf, "\"")
		}
		else if r == '\\' {
			strings.write_string(&buf, "\\")
		}
		else if r == '/' {
			strings.write_string(&buf, "\\/")
		}
		else {
			strings.write_rune(&buf, r)
		}
	}
	return strings.to_string(buf)
}
