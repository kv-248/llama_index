"""Regression tests for SummaryExtractor prev/next metadata."""

import pytest

from llama_index.core.extractors import SummaryExtractor
from llama_index.core.llms.mock import MockLLM
from llama_index.core.schema import Document, NodeRelationship, TextNode


def _nodes_from_document(doc_id: str, texts: list[str]) -> list[TextNode]:
    """Build nodes that point at one source document, in document order."""
    document = Document(text="\n".join(texts), id_=doc_id)
    source = document.as_related_node_info()
    nodes = []
    for index, text in enumerate(texts):
        node = TextNode(text=text, id_=f"{doc_id}-{index}")
        node.relationships[NodeRelationship.SOURCE] = source
        nodes.append(node)
    return nodes


@pytest.mark.asyncio
async def test_summary_extractor_keeps_prev_next_inside_one_document() -> None:
    """
    Prev/next summaries must not cross a document boundary.

    Ingestion runs the extractor on every node in the batch. Nodes from
    different documents are adjacent in that list, but they are not adjacent
    sections of the same document.
    """
    nodes = _nodes_from_document("doc-a", ["alpha-one", "alpha-two"])
    nodes.extend(_nodes_from_document("doc-b", ["beta-one", "beta-two"]))

    metadata = await SummaryExtractor(
        llm=MockLLM(),
        summaries=["self", "prev", "next"],
        show_progress=False,
        num_workers=1,
    ).aextract(nodes)

    assert "alpha-one" in metadata[0]["section_summary"]
    assert "prev_section_summary" not in metadata[0]
    assert "alpha-two" in metadata[0]["next_section_summary"]

    assert "alpha-one" in metadata[1]["prev_section_summary"]
    assert "alpha-two" in metadata[1]["section_summary"]
    assert "next_section_summary" not in metadata[1]
    assert "beta-one" not in metadata[1].get("next_section_summary", "")

    assert "prev_section_summary" not in metadata[2]
    assert "alpha-two" not in metadata[2].get("prev_section_summary", "")
    assert "beta-one" in metadata[2]["section_summary"]
    assert "beta-two" in metadata[2]["next_section_summary"]

    assert "beta-one" in metadata[3]["prev_section_summary"]
    assert "next_section_summary" not in metadata[3]


@pytest.mark.asyncio
async def test_summary_extractor_chains_nodes_without_a_source_document() -> None:
    """Nodes that are not tied to a parent document stay one sequence."""
    nodes = [TextNode(text="bare-one"), TextNode(text="bare-two")]
    assert all(node.ref_doc_id is None for node in nodes)

    metadata = await SummaryExtractor(
        llm=MockLLM(),
        summaries=["prev", "next"],
        show_progress=False,
        num_workers=1,
    ).aextract(nodes)

    assert "prev_section_summary" not in metadata[0]
    assert "bare-two" in metadata[0]["next_section_summary"]
    assert "bare-one" in metadata[1]["prev_section_summary"]
    assert "next_section_summary" not in metadata[1]
