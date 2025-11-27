from corellm import LLM
import gradio as gr


def _memory_to_chatbot(history):
    pairs = []
    current_user = None

    for msg in history:
        if msg["role"] == "user":
            current_user = msg["content"]
        elif msg["role"] == "assistant":
            if current_user is not None:
                pairs.append((current_user, msg["content"]))
                current_user = None

    return pairs


class Interface:
    """
    GUI interface to render LLM chats. Can accept prompts to LLMs and stream responses.
    """

    def __init__(self, llm: LLM, port: int = 3001) -> None:
        """
        Initializes the interface, and links an LLM.

        Args:
            llm (LLM): corellm llm object to link the interface to.
            port (int): port to host the interface on. Defaults to 3001.
        """
        self.llm = llm
        self.port = port

    def render(self) -> None:
        """
        Renders the interface via Gradio
        """

        initial_history = _memory_to_chatbot(self.llm.get_memory())

        def chat_fn(message, history):
            partial = ""
            for chunk in self.llm.chat_stream(message):
                partial += chunk
                yield partial

        app = gr.ChatInterface(
            fn=chat_fn,
            chatbot=gr.Chatbot(value=initial_history),
            title="Chatbot",
            fill_height=True,
        )
        app.launch(
            footer_links=[],
            quiet=True,
            css="footer{display: none !important}",
            server_port=self.port,
            inbrowser=True,
        )
