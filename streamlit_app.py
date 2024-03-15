import streamlit as st
from streamlit_chat import message
from src.indexManager import IndexManager, IndexType
import os

def main():
    st.set_page_config(page_title="CodePilot", layout="wide")
    st.title("CodePilot - Codebase Query Interface")
    st.sidebar.header("Setup")
    # Sidebar
    with st.sidebar:
        repo_name = st.text_input("Repository Name")
        project_dir = st.text_input("Project Directory")
        if st.button("Load"):
            if not repo_name or not project_dir:
                st.error("Please provide both repository name and project directory.")
            elif not os.path.exists(project_dir):
                st.error(f"Project directory '{project_dir}' does not exist.")
            else:
                load_indices(repo_name, project_dir)

    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message_dict in st.session_state.chat_history:
        # message(**message_dict)
        with st.chat_message(message_dict["role"]):
            st.markdown(message_dict["content"])

    if "indices_loaded" not in st.session_state:
        st.warning("Please load the indices before starting the chat.")
    else:
        # user_input = st.chat_input("Ask a question", key="user_input")
        # if st.button("Send") or (user_input and user_input[-1] == "\n"):
        #     query = user_input.strip()
        #     if query:
        #         st.session_state.chat_history.append({"role": "user", "content": query})
        #         with st.spinner("Thinking..."):
        #             try:
        #                 response = st.session_state['index_manager'].query_index_special(query)
        #                 st.session_state.chat_history.append({"role": "assistant", "content": response})
        #             except Exception as e:
        #                 st.error(f"Error generating response: {str(e)}")
        #         st.rerun()
        if prompt := st.chat_input("Ask a question"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # stream = client.chat.completions.create(
                #     model=st.session_state["openai_model"],
                #     messages=[
                #         {"role": m["role"], "content": m["content"]}
                #         for m in st.session_state.messages
                #     ],
                #     stream=True,
                # )
                with st.spinner("🤖 Codepilot is thinking..."):
                    response = st.session_state['index_manager'].query_index_special(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    # Knowledge graph toggle
    if "indices_loaded" in st.session_state and st.button("Toggle Knowledge Graph"):
        if "show_kg" not in st.session_state:
            st.session_state.show_kg = True
        else:
            st.session_state.show_kg = not st.session_state.show_kg

        if st.session_state.show_kg:
            try:
                kg_html = st.session_state.index_manager.get_network_graph()
                st.components.v1.html(kg_html, height=500)
            except Exception as e:
                st.error(f"Error displaying knowledge graph: {str(e)}")
        else:
            st.empty()

def load_indices(repo_name, project_dir):
    llm_settings = {
        'temperature': 0.2,
        'model': 'gpt-4',
        'api_key': "API-KEY"
    }
    embedding_settings = {
        'model': "text-embedding-3-small",
        'embed_batch_size': 100,
        'api_key': "API-KEY"
    }
    try:
        st.session_state.chat_history = []
        st.session_state['index_manager'] = IndexManager(repo_name, project_dir, llm_settings, embedding_settings)
        with st.spinner("Loading knowledge graph index..."):
            st.session_state['index_manager'].create_or_load_index(IndexType.KNOWLEDGE_GRAPH)
            st.success("Knowledge graph index loaded successfully!")
        
        with st.spinner("Loading vector index..."):
            st.session_state['index_manager'].create_or_load_index(IndexType.VECTOR_STORE)
            st.success("Vector index loaded successfully!")

        st.session_state.indices_loaded = True
    except Exception as e:
        st.error(f"Error loading indices: {str(e)}")

def generate_response(query):
    # Placeholder for generating response using the loaded indices
    return "This is a sample response to your query."

def show_kg():
    # Placeholder for generating HTML string representing the knowledge graph
    return "<html><body><h1>Knowledge Graph</h1><p>This is a sample knowledge graph.</p></body></html>"

if __name__ == "__main__":
    main()