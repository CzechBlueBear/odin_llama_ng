package llama_odin_ng

import "core:c"
import "core:c/libc"
import "core:fmt"
import "core:os"
import "core:dynlib"
import "core:strings"
import "core:terminal/ansi"
import "base:runtime"

MAX_LINE_BUFFER :: 2048

READLINE_DYNLIB_PATH :: "/usr/lib/libreadline.so"

libreadline: dynlib.Library

readline_raw: proc "c" (prompt: cstring) -> cstring

readline_init :: proc() {
	ok: bool
	libreadline, ok = dynlib.load_library(READLINE_DYNLIB_PATH)
	if !ok {
		panic("Could not load the readline library (libreadline.so)")
	}

	proc_addr: rawptr
	proc_addr, ok = dynlib.symbol_address(libreadline, "readline")
	if !ok {
		panic("Missing symbol 'readline' in the readline library")
	}

	readline_raw = cast(proc "c" (cstring) -> cstring) proc_addr
}

// read_line :: proc(prompt: string) -> string {

// 	// we need a prompt in C-style, with a zero on the end
// 	prompt_len := len(prompt)
// 	prompt2 := make([]u8, prompt_len + 1)
// 	//defer delete(prompt2)
// 	runtime.mem_copy(&prompt2[0], raw_data(prompt), prompt_len)
// 	prompt2[prompt_len] = 0

// 	// readline() generates a string allocated through libc malloc() so it must be
// 	// freed with libc free()
// 	result := readline_raw(cstring(&prompt2[0]))
// 	defer libc.free(&result)

// 	result2 := strings.clone_from_cstring(result)
// 	return result2
// }

/// Emergency barebones implementation of readline().
/// Reads a line from the terminal, blocking as needed, and returns it
/// as a (newly allocated) string.
read_line :: proc (prompt: string) -> string {

	// set alternative color to visually separate the input from other text
	fmt.print(ansi.CSI + ansi.FG_GREEN + ansi.SGR + "\n", prompt)
	defer fmt.print(ansi.CSI + ansi.RESET + ansi.SGR)

	buf: [MAX_LINE_BUFFER]u8
	bytes_read, err := os.read(os.stdin, buf[:])
	if err != nil {
		panic("Error reading from terminal")
	}
	if bytes_read >= MAX_LINE_BUFFER {
		panic("Excess input from terminal (line too long?)")
	}

	return string(buf[:bytes_read])
}
