from types import SimpleNamespace

import pytest

from llama_index.llms.llama_cpp import base as llama_cpp_base
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in LlamaCPP.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_completion_to_prompt_v3_instruct():
    output = (
        "<|start_header_id|>system<|end_header_id|>\n\nSYSTEM PROMPT<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\nUSER MESSAGE<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert completion_to_prompt_v3_instruct("USER MESSAGE", "SYSTEM PROMPT") == output


def test_messages_to_prompt_v3_instruct():
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="The next sentence said by the assistant is true.",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="The previous sentence said by the user is false.",
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="Do you think the last sentence spoke by me is true or false?",
        ),
    ]
    output = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "SYSTEM PROMPT<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "The next sentence said by the assistant is true.<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "The previous sentence said by the user is false.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Do you think the last sentence spoke by me is true or false?<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert messages_to_prompt_v3_instruct(messages, "SYSTEM PROMPT") == output


def test_context_window_read_back_from_model(tmp_path, monkeypatch):
    """Requesting n_ctx=0 (model default) should report the model's real context."""

    class FakeLlama:
        def __init__(self, model_path, **kwargs):
            self.requested_n_ctx = kwargs.get("n_ctx")
            self.context_params = SimpleNamespace(n_ctx=4096)

        def n_ctx(self):
            return 4096

    monkeypatch.setattr(llama_cpp_base, "Llama", FakeLlama)

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"")

    llm = LlamaCPP(model_path=str(model_path), context_window=0)

    assert llm._model.requested_n_ctx == 0
    assert llm.context_window == 4096
    assert llm.metadata.context_window == 4096


class _FakeDownloadResponse:
    def __init__(self, headers, chunks):
        self.headers = headers
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, chunk_size):
        yield from self._chunks


def test_download_url_accepts_chunked_body_without_content_length(
    tmp_path, monkeypatch
):
    """A chunked response with no Content-Length is not a zero-byte file."""
    body = b"x" * (1000 * 1000)

    def fake_get(url, stream=True):
        assert stream is True
        return _FakeDownloadResponse(headers={}, chunks=[body])

    monkeypatch.setattr(llama_cpp_base.requests, "get", fake_get)
    dest = tmp_path / "model.gguf"

    LlamaCPP._download_url(None, "https://example.invalid/model.gguf", str(dest))

    assert dest.read_bytes() == body


def test_download_url_rejects_undersized_content_length(tmp_path, monkeypatch):
    def fake_get(url, stream=True):
        return _FakeDownloadResponse(
            headers={"Content-Length": "10"}, chunks=[b"0123456789"]
        )

    monkeypatch.setattr(llama_cpp_base.requests, "get", fake_get)
    dest = tmp_path / "model.gguf"

    with pytest.raises(ValueError, match="at least 1 MB, but is only 10 bytes") as exc:
        LlamaCPP._download_url(None, "https://example.invalid/model.gguf", str(dest))

    assert exc.value.args == ("Content should be at least 1 MB, but is only 10 bytes",)
    assert not dest.exists()
