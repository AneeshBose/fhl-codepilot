import os
from tree_sitter_languages import get_language, get_parser

class KnowledgeGraphGenerator:
    # Using a class variable as a cache to store knowledge graph data for project directories
    _cache = {}

    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.language, self.parser = get_language("c_sharp"), get_parser("c_sharp")
        self.scm_file = self.load_scm_file()

    def load_scm_file(self):
        scm_path = "app/scm_files/c_sharp_scm.scm"
        with open(scm_path, 'r') as file:
            return file.read()

    def parse_file(self, file_path):
        with open(file_path, 'r') as file:
            code = file.read()
        tree = self.parser.parse(bytes(code, 'utf8'))
        query = self.language.query(self.scm_file)
        captures = query.captures(tree.root_node)
        return self.extract_triplets(captures)

    def extract_triplets(self, captures):
        triplets = []
        current_class, current_method = None, None
        for capture in captures:
            if capture[1] == 'definition.class':
                current_class = capture[0].child_by_field_name('name').text.decode('utf-8')
                triplets.append((current_class, 'is a', 'class'))
            elif capture[1] == 'name.reference.class':
                nm = capture[0].child_by_field_name('name') or capture[0]
                referenced_class = nm.text.decode('utf-8')
                if current_class:
                    triplets.append((current_class, 'has dependency on', referenced_class))
            elif capture[1] == 'definition.method':
                current_method = capture[0].child_by_field_name('name').text.decode('utf-8')
                if current_class:
                    triplets.append((current_class, 'has method', current_method))
                triplets.append((current_method, 'is a', 'method'))
            elif capture[1] == 'name.reference.send' and capture[0].grammar_name != 'identifier':
                referenced_method = capture[0].text.decode('utf-8')
                if current_method:
                    triplets.append((current_method, 'calls', referenced_method))
        return triplets

    def generate_knowledge_graph(self):
        # Check if knowledge graph is already generated for the project
        if self.project_dir in KnowledgeGraphGenerator._cache:
            print("Using cached knowledge graph.")
            return KnowledgeGraphGenerator._cache[self.project_dir]

        knowledge_graph = []
        for root, dirs, files in os.walk(self.project_dir):
            dirs[:] = [d for d in dirs if d not in ['obj', 'bin']]
            for file in files:
                print("Current File: ", file)
                if file.endswith('.cs'):
                    file_path = os.path.join(root, file)
                    triplets = self.parse_file(file_path)
                    knowledge_graph.extend(triplets)

        # Cache the generated knowledge graph for future use
        KnowledgeGraphGenerator._cache[self.project_dir] = knowledge_graph
        return knowledge_graph

if __name__ == "__main__":
    project_directory = "graphData/repo/ASPNETCore-WebAPI-Sample"
    # while True:
    #     question = input("Ask: ")
    #     if question.lower() == 'quit':
    #         break
    #     kg_generator = KnowledgeGraphGenerator(project_directory)
    #     knowledge_graph = kg_generator.generate_knowledge_graph()
    #     print(knowledge_graph)
    kg_generator = KnowledgeGraphGenerator(project_directory)
    knowledge_graph = kg_generator.generate_knowledge_graph()
    print("Knowledge Graph:", knowledge_graph)
