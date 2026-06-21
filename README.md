# odin_llama_ng
An attempt on a Odin language wrapper for llama.cpp.

Idea-wise, it builds on Yevhen K's work, which is available at https://github.com/yevhen-k/llama.odin.

So far, it only runs a single query using a local LLM, and only works on Linux (due to reliance on the libllama library and dynamic loading; should be relatively easy to make it work on other platforms, though).

## Dynamic libllama library loading

The code does not need llama.cpp project to be present in compilation phase (this should make life of a user a bit easier); instead, it accesses the library directly at runtime using the dynamic linker (similarly to Vulkan). Time will show if this is a viable solution.

I hereby pronounce my deepest thanks to ggerganov and others from the llama.cpp project for keeping the crucial API in plain C format, without mangling. If the symbols were in C++ style, it would be a living hell to interface with.
