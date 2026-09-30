from src.indexing.chunker import CodeChunker, detect_language


def test_detect_language():
    assert detect_language(file_path="main.py") == "python"
    assert detect_language(file_path="App.java") == "java"
    assert detect_language(file_path="service.ts") == "typescript"
    assert detect_language(file_path="server.js") == "javascript"
    assert detect_language(file_path="main.cpp") == "cpp"
    assert detect_language(text="def hello(): pass") == "python"


def test_code_chunker_single_chunk():
    chunker = CodeChunker(max_chunk_lines=50)
    code = "def parse_data(raw):\n    return raw.strip()\n"
    chunks = chunker.chunk_code(code, doc_id="doc1", file_path="parse.py")

    assert len(chunks) == 1
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].language == "python"
    assert chunks[0].symbol_name == "parse_data"
    assert chunks[0].symbol_type == "function"
    assert chunks[0].start_line == 1


def test_code_chunker_multi_chunk():
    chunker = CodeChunker(max_chunk_lines=5, overlap_lines=2)
    lines = [f"line_{i} = {i}" for i in range(12)]
    code = "\n".join(lines)
    chunks = chunker.chunk_code(code, doc_id="large_doc", language="python")

    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[1].doc_id == "large_doc_chunk_1"
