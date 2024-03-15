import os
import logging
import string
import sys
from enum import Enum, auto
from llama_index.core import (
    SimpleDirectoryReader, KnowledgeGraphIndex, VectorStoreIndex,
    StorageContext, load_index_from_storage, Settings
)
from llama_index.core.query_engine import BaseQueryEngine
# from llama_index.core.indices.composability.graph import ComposableGraph
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from src.KgBuilder import KnowledgeGraphGenerator
# from app.treeparse6 import get_knowledge_graph  # Ensure this path is correct
from pyvis.network import Network
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import PromptTemplate

class IndexType(Enum):
    KNOWLEDGE_GRAPH = auto()
    VECTOR_STORE = auto()
    COMBINED = auto()

class IndexManager:
    def __init__(self, repo_name, document_dir, llm_settings, embedding_settings):
        self.repo_name = repo_name
        self.document_dir = document_dir
        self.llm_settings = llm_settings
        self.embedding_settings = embedding_settings
        self.indices = {}  # Store indices with their types as keys
        self.query_engines = {}  # Store query_engines with their types as keys
        self.base_persist_dir = "./.idxstorage"
        self.documents = None  # Placeholder for loaded documents
        self.kg_generator = KnowledgeGraphGenerator(self.document_dir)
        self.kg_generated = False
        self.token_counter = None
        # self.setup_logging()
        self.initialize_settings()
        self.load_documents()  # Load documents at initialization

    def setup_logging(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    def initialize_settings(self):
        llm = OpenAI(temperature=self.llm_settings['temperature'], model=self.llm_settings['model'], api_key=self.llm_settings['api_key'])
        Settings.llm = llm
        Settings.embed_model = OpenAIEmbedding(
            model=self.embedding_settings['model'],
            embed_batch_size=self.embedding_settings['embed_batch_size'],
            api_key=self.embedding_settings['api_key']
        )
        # you can set a tokenizer directly, or optionally let it default
        # to the same tokenizer that was used previously for token counting
        # NOTE: The tokenizer should be a function that takes in text and returns a list of tokens
        self.token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model(self.llm_settings['model']).encode,
            verbose=True,  # set to true to see usage printed to the console
        )
        Settings.callback_manager = CallbackManager([self.token_counter])

    def load_documents(self):
        # This method can be adjusted in the future for different document loading strategies
        self.documents = SimpleDirectoryReader(input_dir=self.document_dir, recursive=True).load_data()

    def create_or_load_index(self, index_type: IndexType):
        if index_type == IndexType.KNOWLEDGE_GRAPH:
            self.create_or_load_graph_index()
        elif index_type == IndexType.VECTOR_STORE:
            self.create_or_load_vector_index()

    def get_knowledge_graph(self, input_text):
        if not self.kg_generated:
            self.kg_generated = True
            kgg = self.kg_generator.generate_knowledge_graph()
            return kgg
        return []
    
    def create_or_load_graph_index(self):
        persist_dir = os.path.join(self.base_persist_dir, self.repo_name, IndexType.KNOWLEDGE_GRAPH.name.lower())
        if not os.path.exists(persist_dir):
            graph_store = SimpleGraphStore()
            storage_context = StorageContext.from_defaults(graph_store=graph_store)
            index = KnowledgeGraphIndex.from_documents(
                self.documents,
                kg_triplet_extract_fn=self.get_knowledge_graph,
                storage_context=storage_context,
                include_embeddings=True,
            )
            index.storage_context.persist(persist_dir=persist_dir)
            self.indices[IndexType.KNOWLEDGE_GRAPH] = index
        else:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            self.indices[IndexType.KNOWLEDGE_GRAPH] = index

    def create_or_load_vector_index(self):
        persist_dir = os.path.join(self.base_persist_dir, self.repo_name, IndexType.VECTOR_STORE.name.lower())
        if not os.path.exists(persist_dir):
            index = VectorStoreIndex.from_documents(self.documents)
            index.storage_context.persist(persist_dir=persist_dir)
            self.indices[IndexType.VECTOR_STORE] = index
        else:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            self.indices[IndexType.VECTOR_STORE] = index

    def create_combined_index(self):
        if IndexType.KNOWLEDGE_GRAPH in self.indices and IndexType.VECTOR_STORE in self.indices:
            retriever = QueryFusionRetriever(
                    [self.indices[IndexType.KNOWLEDGE_GRAPH].as_retriever(), self.indices[IndexType.VECTOR_STORE].as_retriever()],
                    similarity_top_k=10,
                    num_queries=1,  # set this to 1 to disable query generation
                    use_async=True,
                    verbose=True,
                    # query_gen_prompt="...",  # we could override the query generation prompt here
                )
            query_engine = RetrieverQueryEngine.from_args(retriever)
            self.indices[IndexType.COMBINED] = query_engine
            # self.indices[IndexType.COMBINED] = ComposableGraph.from_indices(indices=[self.indices[IndexType.KNOWLEDGE_GRAPH], self.indices[IndexType.VECTOR_STORE]])
            # index = GPTListIndex([index1, index2])
        else:
            print("Both knowledge_graph and vector_store indices must be loaded or created before combining.")

    def display_prompt_dict(self, prompts_dict):
        for k, p in prompts_dict.items():
            text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
            print(text_md)
            print(p.get_template())
            print("<br><br>")

    def query_index(self, index_type: IndexType, query, response_mode="tree_summarize", embedding_mode="hybrid", similarity_top_k=10):
        qa_prompt_str = (
            "You are an expert Software Engineer. Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the question: {query_str}\n"
            "When asked for sequence diagram, provide the sequence diagram in the format:\n"
            "```sequence\n"
            "Actor 1 -> Actor 2: Message 1\n"
            "Actor 2 -> Actor 1: Message 2\n"
            "```"
        )
        qa_prompt_str_tmpl = PromptTemplate(qa_prompt_str)
        if index_type in self.indices:
            index = self.indices[index_type]
            if index_type == IndexType.COMBINED:
                query_engine = index
            else:
                query_engine = index.as_query_engine(
                    # verbose=True,
                    include_text=True,
                    text_qa_template = qa_prompt_str,
                    # response_mode=response_mode,
                    embedding_mode=embedding_mode,
                    similarity_top_k=similarity_top_k
                )
                query_engine.update_prompts(
                    {"response_synthesizer:text_qa_template": qa_prompt_str_tmpl}
                )
                prompts_dict = query_engine.get_prompts()
                self.display_prompt_dict(prompts_dict)
                # print(
                #     "Embedding Tokens: ",
                #     self.token_counter.total_embedding_token_count,
                #     "\n",
                #     "LLM Prompt Tokens: ",
                #     self.token_counter.prompt_llm_token_count,
                #     "\n",
                #     "LLM Completion Tokens: ",
                #     self.token_counter.completion_llm_token_count,
                #     "\n",
                #     "Total LLM Token Count: ",
                #     self.token_counter.total_llm_token_count,
                # )
            response = query_engine.query(query)
            prompts_dict = query_engine.get_prompts()
            # self.display_prompt_dict(prompts_dict)
            print(response)
            return response
        else:
            print(f"{index_type.name} index is not initialized.")
            return None

    def query_index_special(self, query: string):
        qa_prompt_str = (
            "You are an expert Software Engineer. Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the question: {query_str}\n"
            "When asked for sequence diagram, provide the sequence diagram in the format:\n"
            "```sequence\n"
            "Actor 1 -> Actor 2: Message 1\n"
            "Actor 2 -> Actor 1: Message 2\n"
            "```"
        )
        qa_prompt_str_tmpl = PromptTemplate(qa_prompt_str)
        is_seq = "sequence diagram" in query.lower()
        if self.query_engines.get(IndexType.KNOWLEDGE_GRAPH) is None:
            graphIdx: KnowledgeGraphIndex = self.indices[IndexType.KNOWLEDGE_GRAPH]
            query_engine = graphIdx.as_query_engine(
                verbose=True,
                include_text=True,
                # response_mode=response_mode,
                embedding_mode="hybrid",
                similarity_top_k=10
            )
            self.query_engines[IndexType.KNOWLEDGE_GRAPH] = query_engine
        
        if self.query_engines.get(IndexType.VECTOR_STORE) is None:
            vectorIdx: VectorStoreIndex = self.indices[IndexType.VECTOR_STORE]
            query_engine = vectorIdx.as_query_engine(
                verbose=True,
                include_text=True,
                # response_mode=response_mode,
                embedding_mode="hybrid",
                similarity_top_k=5
            )
            # query_engine.update_prompts(
            #     {"response_synthesizer:text_qa_template": qa_prompt_str_tmpl}
            # )
            self.query_engines[IndexType.VECTOR_STORE] = query_engine

        kgEngine: BaseQueryEngine = self.query_engines[IndexType.KNOWLEDGE_GRAPH]
        vecEngine: BaseQueryEngine = self.query_engines[IndexType.VECTOR_STORE]

        if is_seq:
            kgEngine.update_prompts(
                {"response_synthesizer:text_qa_template": qa_prompt_str_tmpl}
            )
            vecEngine.update_prompts(
                {"response_synthesizer:text_qa_template": qa_prompt_str_tmpl}
            )

        kgResp = kgEngine.query(query)
        print("KG: ", kgResp.response)
        queryUp = query + "\n Additional Context: " + kgResp.response
        vecResp = vecEngine.query(queryUp)
        print("VEC: ", vecResp.response)
        return vecResp

    def query_index_chat(self, index_type: IndexType, query, response_mode="tree_summarize", embedding_mode="hybrid", similarity_top_k=150):
        if index_type in self.indices:
            index = self.indices[index_type]
            query_engine = index.as_chat_engine(
                chat_mode = "context",
                # verbose=True,
                # response_mode=response_mode,
                embedding_mode=embedding_mode,
                similarity_top_k=similarity_top_k
            )
            prompts_dict = query_engine.get_prompts()
            print(list(prompts_dict.keys()))
            response = query_engine.query(query)
            print(response)
            return response
        else:
            print(f"{index_type.name} index is not initialized.")
            return None

    def get_network_graph(self, filename="example.html"):
        if IndexType.KNOWLEDGE_GRAPH in self.indices:
            g = self.indices[IndexType.KNOWLEDGE_GRAPH].get_networkx_graph()
            net = Network(notebook=True, cdn_resources="in_line", directed=True)
            net.from_nx(g)
            html = net.generate_html()
            with open(filename, mode='w', encoding='utf-8') as fp:
                fp.write(html)
            print(f"Network graph saved to {filename}.")
            return html
        else:
            print("Knowledge graph index is not initialized.")

if __name__ == "__main__":
    repo_name = "REPO"
    document_dir = "test_repos/HR-REPO"
    llm_settings = {'temperature': 0.2, 'model': 'gpt-4', 'api_key': "API-KEY"}
    embedding_settings = {'model': "text-embedding-3-small", 'embed_batch_size': 100, 'api_key': "API-KEY"}

    index_manager = IndexManager(repo_name, document_dir, llm_settings, embedding_settings)
    # Create or load indices as needed
    index_manager.create_or_load_index(IndexType.KNOWLEDGE_GRAPH)
    index_manager.create_or_load_index(IndexType.VECTOR_STORE)
    # Optionally, combine indices
    index_manager.create_combined_index()

    # Example usage to query the combined index
    while True:
        question = input("Ask a question about the codebase (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        # index_manager.query_index(IndexType.KNOWLEDGE_GRAPH, question)  # Query the combined index
        # index_manager.query_index(IndexType.COMBINED, question)  # Query the combined index

        # index_manager.query_index_chat(IndexType.KNOWLEDGE_GRAPH, question)  # Query the combined index

        index_manager.query_index_special(question)  # Query the combined index

    # Generate a network graph from the knowledge graph index
    # index_manager.get_network_graph("network_graph.html")
