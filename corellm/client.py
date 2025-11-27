import ctypes
from typing import List, Dict, Any, Generator
from llama_cpp import Llama, llama_log_set


def cur_log_callback(level, message, user_data):
    pass


log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)(
    cur_log_callback
)
llama_log_set(log_callback, ctypes.c_void_p())


class LLM:
    """
    LLM class providing efficient AI interaction.
    """

    def __init__(
        self,
        path: str,
        system_prompt: str = "You are a helpful CoreLLM assistant.",
        max_tokens: int = 4096,
    ) -> None:
        """
        Initialize the LLM with an optional system prompt and context window.

        Args:
            path (str): File system path to the GGUF model.
            system_prompt (str): Data to set as system prompt for LLM.
            max_tokens (int): Maximum number of context tokens the model can use. Defaults to 4096 tokens.
        """
        self.llm = Llama(model_path=path, n_ctx=max_tokens, verbose=False)

        self.system_prompt = system_prompt

        self.memory = [{"role": "system", "content": system_prompt}]

    def chat(self, prompt: str) -> str:
        """
        Prompts the LLM, and adds the query/response to current conversation history.
        Future chat calls will remember prior prompts unless memory is modified.

        Args:
            prompt (str): A string to prompt the LLM.

        Returns:
            str: The assistant's generated response.
        """

        self.memory.append({"role": "user", "content": prompt})

        response = self.llm.create_chat_completion(messages=self.memory)

        message = response["choices"][0]["message"]["content"]

        self.memory.append({"role": "assistant", "content": message})

        return message

    def prompt(self, prompt: str, use_memory: bool = True) -> str:
        """
        Prompts the LLM, but does not add query/response to conversation memory.

        Args:
            prompt (str): A string to prompt the LLM.
            use_memory (bool): Determines whether the LLM should use its current chat history (True) or only system prompt (False).

        Returns:
            str: The assistant's generated response.
        """

        if use_memory:
            temp_memory = self.memory.copy()
            temp_memory.append({"role": "user", "content": prompt})
        else:
            temp_memory = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

        response = self.llm.create_chat_completion(messages=temp_memory)

        message = response["choices"][0]["message"]["content"]

        return message

    def chat_stream(self, prompt: str, print_stream: bool = False) -> Generator[str]:
        """
        Prompts the LLM, and yields (streams) tokens as they are generated.
        Adds the query/response to current conversation history.

        Example Usage:
            for token in ai.chat_stream("What does LLM stand for?"):
                print(token, end="", flush=True)

        Args:
            prompt (str): A string to prompt the LLM.
            print_stream (bool): Determines whether to print the streamed tokens to standard out, defaults False.

        Yields:
            str (chunk): A chunk of the assistant's generated response.
        """

        def _generator():
            self.memory.append({"role": "user", "content": prompt})

            stream = self.llm.create_chat_completion(messages=self.memory, stream=True)

            full_message = ""

            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                token = delta.get("content", "")

                if token:
                    full_message += token
                    if print_stream:
                        print(token, end="", flush=True)
                    yield token

            if print_stream:
                print()
            self.memory.append({"role": "assistant", "content": full_message})

        gen = _generator()

        # Auto-run if print_stream=True
        if print_stream:
            for _ in gen:
                pass  # printing already happens inside the generator
            return None  # nothing to consume

        # Otherwise return the generator normally
        return gen

    def prompt_stream(
        self, prompt: str, print_stream: bool = False, use_memory: bool = True
    ) -> Generator[str]:
        """
        Prompts the LLM, and yields (streams) tokens as they are generated.
        Does not add the query/response to current conversation history.

        Example Usage:
            for token in ai.chat_stream("What does LLM stand for?"):
                print(token, end="", flush=True)

        Args:
            prompt (str): A string to prompt the LLM.
            print_stream (bool): Determines whether to print the streamed tokens to standard out, defaults False.
            use_memory (bool): Determines whether the LLM should use its current chat history (True) or only system prompt (False).

        Yields:
            str (chunk): A chunk of the assistant's generated response.
        """

        def _generator():
            # Build temporary memory context
            if use_memory:
                temp_memory = self.memory.copy()
                temp_memory.append({"role": "user", "content": prompt})
            else:
                temp_memory = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]

            # Start streaming
            stream = self.llm.create_chat_completion(messages=temp_memory, stream=True)

            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    if print_stream:
                        print(token, end="", flush=True)
                    yield token

            if print_stream:
                print()

        gen = _generator()

        # Auto-run if print_stream=True
        if print_stream:
            for _ in gen:
                pass  # printing happens inside the generator
            return None

        # Otherwise return the generator for manual iteration
        return gen

    def clear_memory(self) -> None:
        """
        Clears the LLM's conversation history.
        """

        self.memory = [{"role": "system", "content": self.system_prompt}]

    def set_memory(self, memory: list[dict[str, str]]) -> None:
        """
        Manually sets the LLM's conversation history.

        Example input:
            memory = [
                {"role": "system", "content": "You are a helpful CoreLLM assistant"},
                {"role": "user", "content": "What does LLM stand for?"},
                {"role": "assistant", "content": "It stands for Large Language Model"}
            ]

        Args:
            memory (list[dict[str, str]]): the conversation history of the LLM to inject.
        """

        self.memory = memory

    def get_memory(self) -> list[dict[str, str]]:
        """
        Returns the LLM's current conversation history.

        Example output:
            memory = [
                {"role": "system", "content": "You are a helpful CoreLLM assistant"},
                {"role": "user", "content": "What does LLM stand for?"},
                {"role": "assistant", "content": "It stands for Large Language Model"}
            ]

        Returns:
            list[dict[str, str]]: List of dictionaries of the current conversation history.
        """

        return self.memory

    def modify_system_prompt(self, system_prompt: str):
        """
        Modifies the system prompt to the given string.

        system_prompt (str): the string to change the system prompt to
        """

        self.memory[0] = {"role": "system", "content": system_prompt}
