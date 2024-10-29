import pytest
import pytest_asyncio
import aiohttp
import allure
import json
import csv
import os
from aioresponses import aioresponses
from aiohttp import FormData

url = os.getenv("TEST_URL", "http://localhost:8000/evaluator-stream-json")

def csv_load(csv_file_path):
    data = []
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)
    return data

@allure.step("Grade")
def grade(rsp):
    grade = {}
    grade['right'] = 0
    grade['wrong'] = 0
    grade['total'] = 0
    #pytest.set_trace()
    for r in rsp:
        if "Incorrect" in str(r):
            grade['wrong'] += 1
        else:
            grade['right'] += 1
        grade['total'] += 1
    g = (grade['right'] / grade['total']) * 100
    with allure.step("Grade: {}".format(g)):
        pass
    return g

@pytest.mark.asyncio
@allure.epic("RAG Testing")
@allure.feature("AutoGrading")
@allure.story("sno-for-100-fast-1024-40")
async def test_fast_grade_1024_40():
    data = FormData()
    data.add_field('files', open('docs/sno-for-100/sno-for-100.txt', 'rb'), filename='sno-for-100.txt', content_type='text/plain')
    data.add_field('test_dataset', json.dumps(csv_load("docs/sno-for-100/sno-for-100-qa.csv")))
    data.add_field('num_eval_questions', '5')
    data.add_field('chunk_chars', '1024')
    data.add_field('overlap', '40')
    data.add_field('split_method', 'RecursiveTextSplitter')
    data.add_field('retriever_type', 'similarity-search')
    data.add_field('embeddings', 'HuggingFace')
    data.add_field('model_version', 'llama-3.1-8B-instruct')
    data.add_field('grade_prompt', 'Fast')  # OpenAI grading prompt
    data.add_field('num_neighbors', '3')

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        async with session.post(url, data=data) as response:
            rsp = await response.json()
            response.raise_for_status()
            print(rsp)
            g = grade(rsp)
            print(f"Grade: {g}")
            assert g > 59.99


@pytest.mark.asyncio
@allure.epic("RAG Testing")
@allure.feature("AutoGrading")
@allure.story("sno-for-100-fast-500-20")
async def test_fast_grade_500_20():
    data = FormData()
    data.add_field('files', open('docs/sno-for-100/sno-for-100.txt', 'rb'), filename='sno-for-100.txt', content_type='text/plain')
    data.add_field('test_dataset', json.dumps(csv_load("docs/sno-for-100/sno-for-100-qa.csv")))
    data.add_field('num_eval_questions', '5')
    data.add_field('chunk_chars', '500')
    data.add_field('overlap', '20')
    data.add_field('split_method', 'RecursiveTextSplitter')
    data.add_field('retriever_type', 'similarity-search')
    data.add_field('embeddings', 'HuggingFace')
    data.add_field('model_version', 'llama-3.1-8B-instruct')
    data.add_field('grade_prompt', 'Fast')  # OpenAI grading prompt
    data.add_field('num_neighbors', '3')

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        async with session.post(url, data=data) as response:
            rsp = await response.json()
            response.raise_for_status()
            print(rsp)
            g = grade(rsp)
            print(f"Grade: {g}")
            assert g > 59.99


@pytest.mark.asyncio
@allure.epic("RAG Testing")
@allure.feature("AutoGrading")
@allure.story("sno-for-100-fast-1500-100")
async def test_fast_grade_1500_100():
    data = FormData()
    data.add_field('files', open('docs/sno-for-100/sno-for-100.txt', 'rb'), filename='sno-for-100.txt', content_type='text/plain')
    data.add_field('test_dataset', json.dumps(csv_load("docs/sno-for-100/sno-for-100-qa.csv")))
    data.add_field('num_eval_questions', '5')
    data.add_field('chunk_chars', '1500')
    data.add_field('overlap', '100')
    data.add_field('split_method', 'RecursiveTextSplitter')
    data.add_field('retriever_type', 'similarity-search')
    data.add_field('embeddings', 'HuggingFace')
    data.add_field('model_version', 'llama-3.1-8B-instruct')
    data.add_field('grade_prompt', 'Fast')  # OpenAI grading prompt
    data.add_field('num_neighbors', '3')

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        async with session.post(url, data=data) as response:
            rsp = await response.json()
            response.raise_for_status()
            print(rsp)
            g = grade(rsp)
            print(f"Grade: {g}")
            assert g > 59.99