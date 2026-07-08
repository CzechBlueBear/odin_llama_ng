# odin_llama_ng

An attempt on a Odin language wrapper together with a basic inference utility for llama.cpp.

Idea-wise, it builds on Yevhen K's work, which is available at https://github.com/yevhen-k/llama.odin.

So far, only a single mode is available: run ``./llama-odin-ng`` with a model path and a prompt string. After the first inference, you are in dialogue mode; you can continue in a dialogue or just enter "/quit" to finish. (More such commands are incoming; for now you can try "/record" for writing the whole dialogue into the "./chat_history.txt" file.)

So far, it only works on Linux (due to reliance on the libllama library and dynamic loading; should be relatively easy to make it work on other platforms, though).

## Dynamic libllama library loading

The code does not need llama.cpp project to be present in compilation phase (this should make life of the packager or power user a bit easier); instead, it accesses the library directly at runtime using the dynamic linker (similarly to Vulkan). Time will show if this is a viable solution.

## Thanks

I hereby pronounce my deepest thanks to ggerganov and others from the llama.cpp project for keeping the crucial API in plain C format, without mangling. If the symbols were in C++ style, it would be a living hell to interface with.

Also thanks to Yevhen K for the previous work I am building on.

Also thanks to gingerbill et al. for devising such a good programming language!
