import torch
import os
import shutil
import time
import re
import gc
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, AutoModelForCausalLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from peft import PeftModel

from config import DEVICE, MAX_LENGTH, TEMPERATURE, REPETITION_PENALTY, FINETUNED_GPT2_LARGE_PATH, INPUT_LIMIT

import warnings
warnings.filterwarnings('ignore')

# Base class
class GPT2TextGenerator:
    retriever = None  #
    chroma_loaded = False  # checker if Chroma is loaded

    def __init__(self, model_name, device=DEVICE):
        self.device = device
        self.tokenizer = self.set_tokenizer(model_name)
        self.model = self.set_model(model_name).to(self.device)
        self.model.eval()

    def set_model(self, model_name):
        raise NotImplementedError("Subclasses must implement set_model()")

    def set_tokenizer(self, model_name):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_generator(self, max_length=MAX_LENGTH, temperature=TEMPERATURE):
        """Creation of generator pipeline with parameters"""
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=max_length,
            temperature=temperature,
            repetition_penalty=REPETITION_PENALTY,
        )

    def generate_text(self, prompt, max_length=MAX_LENGTH, temperature=TEMPERATURE, repetition_penalty=REPETITION_PENALTY):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # Setting max answer length (input length + user max length | or max limit = 1024)
        max_length_with_input = min(input_length + max_length, INPUT_LIMIT)
        print("Input length: ", input_length)
        print("Max length: ", max_length_with_input)

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                max_length=max_length_with_input,
                num_beams=5,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask,
                no_repeat_ngram_size=2,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()

        # Answer cleaning to implement
        # ...

        return generated_text


    def generate_text_from_pdf(self, query, max_length=MAX_LENGTH, temperature=TEMPERATURE):
        if not BaseGPT2TextGenerator.retriever:
            raise ValueError("At first load PDF using load_pdf().")

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="\nUse the following context to answer the question concisely:\n{context}\nQuestion: {question}\nAnswer:"
        )

        # Get context from retriever
        context_docs = BaseGPT2TextGenerator.retriever.get_relevant_documents(query)
        context_texts = "\n".join([doc.page_content for doc in context_docs])

        # Create prompt (context + question)
        full_prompt = qa_prompt.format(context=context_texts, question=query)

        # Prompt tokenization
        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
        input_length = input_ids.shape[1]

        # Setting max answer length (input length + user max length | or max limit = 1024)
        max_length_with_input = min(input_length + max_length, INPUT_LIMIT)
        print("Input length: ", input_length)
        print("Max length: ", max_length_with_input)

        generator = self.get_generator(max_length_with_input, temperature)
        llm = HuggingFacePipeline(pipeline=generator)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=BaseGPT2TextGenerator.retriever,
            verbose=True,
            return_source_documents=False,
            chain_type_kwargs={"prompt": qa_prompt}
        )

        result = qa.run(query)
        print("Full answer:\n", result)

        # Cleaning answer from repetition of questions and answers
        match = re.search(r"Answer:\s*(.*?)(?=\n\n|\nQuestion:|\Z)", result, re.DOTALL | re.IGNORECASE)
        clean_answer = match.group(1).strip() if match else "No answer found."
        clean_answer = re.sub(r"(?<!\n)\n(?!\n)", "", clean_answer)

        return clean_answer


    @classmethod
    def load_pdf(cls, pdf_path, chunk_size=500, chunk_overlap=100):
        if cls.chroma_loaded:
            print("Chroma database is already loaded. Skipping reloading.")
            return

        cls.clear_pdf()

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(pages)

        embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": DEVICE})

        vectordb = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")
        cls.retriever = vectordb.as_retriever()
        cls.chroma_loaded = True

        print(f"Loaded {len(vectordb.get()['documents'])} documents into the database.")

    @classmethod
    def clear_pdf(cls):
        """Delete retriever and clear vector base."""
        if cls.retriever:
            cls.retriever = None
            cls.chroma_loaded = False

        if os.path.exists("chroma_db"):
            try:
                shutil.rmtree("chroma_db")
            except PermissionError:
                gc.collect()
                time.sleep(1)
                shutil.rmtree("chroma_db", ignore_errors=True)

        print("Chroma database cleared.")


# GPT-2 Large
class BaseGPT2TextGenerator(GPT2TextGenerator):
    def __init__(self, model_name, device=DEVICE):
        super().__init__(model_name, device)

    def set_model(self, model_name):
        return GPT2LMHeadModel.from_pretrained(model_name)

# GPT-2 Large fine-tuned on squad using LoRA
class FineTunedGPT2TextGenerator(GPT2TextGenerator):
    def __init__(self, model_name, device=DEVICE):
        super().__init__(model_name, device)

    def set_model(self, model_name):
        base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        return PeftModel.from_pretrained(base_model, FINETUNED_GPT2_LARGE_PATH)
