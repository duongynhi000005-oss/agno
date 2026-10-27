"""Per-upload chunker overrides must not mutate readers held by the knowledge base."""

from unittest.mock import AsyncMock, patch

import pytest

from agno.knowledge.content import Content, FileData
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.text_reader import TextReader
from agno.os.routers.knowledge.knowledge import process_content


@pytest.mark.asyncio
async def test_process_content_does_not_mutate_custom_reader_chunker():
    custom_reader = TextReader(name="notes")
    knowledge = Knowledge(name="docs", readers={"notes": custom_reader})
    assert knowledge.readers["notes"].chunking_strategy.chunk_size == 5000

    first_content = Content(
        name="a.txt",
        file_data=FileData(content=b"hello", type="text/plain", filename="a.txt"),
    )
    second_content = Content(
        name="b.txt",
        file_data=FileData(content=b"world", type="text/plain", filename="b.txt"),
    )

    with patch.object(knowledge, "_aload_content", new_callable=AsyncMock):
        await process_content(
            knowledge,
            first_content,
            reader_id="notes",
            chunker="FixedSizeChunker",
            chunk_size=1000,
            chunk_overlap=50,
        )
        assert first_content.reader is not None
        assert first_content.reader is not custom_reader
        assert first_content.reader.chunking_strategy.chunk_size == 1000
        assert first_content.reader.chunking_strategy.overlap == 50

        # The held custom reader must keep its original chunking strategy
        assert knowledge.readers["notes"] is custom_reader
        assert knowledge.readers["notes"].chunking_strategy.chunk_size == 5000

        await process_content(
            knowledge,
            second_content,
            reader_id="notes",
        )
        assert second_content.reader is not None
        assert second_content.reader.chunking_strategy.chunk_size == 5000


@pytest.mark.asyncio
async def test_process_content_does_not_mutate_cached_default_reader_chunker():
    knowledge = Knowledge(name="docs")
    default_text_reader = knowledge.text_reader
    assert default_text_reader is not None
    assert default_text_reader.chunking_strategy.chunk_size == 5000
    assert knowledge.readers["text"] is default_text_reader

    content = Content(
        name="c.txt",
        file_data=FileData(content=b"hello", type="text/plain", filename="c.txt"),
    )

    with patch.object(knowledge, "_aload_content", new_callable=AsyncMock):
        await process_content(
            knowledge,
            content,
            reader_id="text",
            chunker="FixedSizeChunker",
            chunk_size=750,
            chunk_overlap=0,
        )

    assert content.reader is not None
    assert content.reader is not default_text_reader
    assert content.reader.chunking_strategy.chunk_size == 750
    assert knowledge.readers["text"] is default_text_reader
    assert knowledge.readers["text"].chunking_strategy.chunk_size == 5000
