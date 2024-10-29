import requests
import json
import os

url = os.getenv("TEST_URL", "http://localhost:8000/evaluator-stream-json")

def test_fast_grade_1500_150():
    files = {
        'files': ('karpathy-pod.txt', open('docs/karpathy-lex-pod/karpathy-pod.txt', 'rb'), 'text/plain'),
        'num_eval_questions': (None, '1'),
        'chunk_chars': (None, '1000'),
        'overlap': (None, '50'),
        'split_method': (None, 'RecursiveTextSplitter'),
        'retriever_type': (None, 'similarity-search'),
        'embeddings': (None, 'HuggingFace'),
        'model_version': (None, 'llama-3.1-8B-instruct'),
        'grade_prompt': (None, 'Fast'), # OpenAI grading prompt
        'num_neighbors': (None, '3'),
    }

    response = requests.post(url, files=files, timeout=120)
    response.raise_for_status()
    json_str = json.dumps(response.json(), indent=4)
    print(json_str)
    assert "Incorrect" not in json_str


def test_fast_grade_1000_100():
    files = {
        'files': ('karpathy-pod.txt', open('docs/karpathy-lex-pod/karpathy-pod.txt', 'rb'), 'text/plain'),
        'num_eval_questions': (None, '1'),
        'chunk_chars': (None, '1000'),
        'overlap': (None, '50'),
        'split_method': (None, 'RecursiveTextSplitter'),
        'retriever_type': (None, 'similarity-search'),
        'embeddings': (None, 'HuggingFace'),
        'model_version': (None, 'llama-3.1-8B-instruct'),
        'grade_prompt': (None, 'Fast'), # OpenAI grading prompt
        'num_neighbors': (None, '3'),
    }

    response = requests.post(url, files=files, timeout=120)
    response.raise_for_status()
    json_str = json.dumps(response.json(), indent=4)
    print(json_str)
    assert "Incorrect" not in json_str

def test_fast_grade_1024_40():
    files = {
        'files': ('karpathy-pod.txt', open('docs/karpathy-lex-pod/karpathy-pod.txt', 'rb'), 'text/plain'),
        'num_eval_questions': (None, '1'),
        'chunk_chars': (None, '1024'),
        'overlap': (None, '40'),
        'split_method': (None, 'RecursiveTextSplitter'),
        'retriever_type': (None, 'similarity-search'),
        'embeddings': (None, 'HuggingFace'),
        'model_version': (None, 'llama-3.1-8B-instruct'),
        'grade_prompt': (None, 'Fast'), # OpenAI grading prompt
        'num_neighbors': (None, '3'),
    }

    response = requests.post(url, files=files, timeout=120)
    response.raise_for_status()
    json_str = json.dumps(response.json(), indent=4)
    print(json_str)
    assert "Incorrect" not in json_str

def test_fast_grade_500_20():
    files = {
        'files': ('karpathy-pod.txt', open('docs/karpathy-lex-pod/karpathy-pod.txt', 'rb'), 'text/plain'),
        'num_eval_questions': (None, '1'),
        'chunk_chars': (None, '500'),
        'overlap': (None, '20'),
        'split_method': (None, 'RecursiveTextSplitter'),
        'retriever_type': (None, 'similarity-search'),
        'embeddings': (None, 'HuggingFace'),
        'model_version': (None, 'llama-3.1-8B-instruct'),
        'grade_prompt': (None, 'Fast'), # OpenAI grading prompt
        'num_neighbors': (None, '3'),
    }

    response = requests.post(url, files=files, timeout=120)
    response.raise_for_status()
    json_str = json.dumps(response.json(), indent=4)
    print(json_str)
    assert "Incorrect" not in json_str
