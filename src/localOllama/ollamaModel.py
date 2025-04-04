from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from ollama import Client
import config.config as config
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma


class OllamaModel:
    
    modelName = None
    
    model = None
    
    client = Client(host=config.localOllamaClientHost)
    
    def setOllamaModel(self, isChangeModel = False):
        print("Installed Models: ")
        modelList = []
        for index, model in enumerate(self.client.list()['models']):
            modelList.append(model['model'])
            print(f"{index + 1}. {model['model']}")
            
        while True:
            value = self.getModelFromUser(self.client.list()['models'])
            match value:
                case "quit":
                    if not isChangeModel:
                        print("Exiting application...")
                        exit(0)
                    else:
                        break
                case _:
                    if value in modelList:
                        self.modelName = value
                        break
                    else:
                        print("Invalid model name")
                        
        print(f"Selected model: {self.modelName}")
        self.model = ChatOllama(model=self.modelName, temperature=0)
        # self.model._client = config.localOllamaClientHost
        
    def getModelFromUser(self, modelList):
        userInput = input("What model do you want to use? (quit to exit) ")
        if userInput.isdigit() and int(userInput) <= len(modelList):
            return modelList[int(userInput) - 1]['model']
        elif userInput in modelList:
            return userInput
        else:
            return userInput
        
    def ollamaEmbedding(self):
        embedding = OllamaEmbeddings(model=self.modelName)
        embedding._client = self.client
        return embedding

    def queryDatabase(
        self, queryText: str, db: Chroma, model: str, n_results: int = 5
    ):
        # Search the database
        try:
            results = db.similarity_search_with_relevance_scores(queryText, k=n_results)
        except Exception as e:
            print(f"Error: {e}")
            return f"Error: {e}"

        prompt = ChatPromptTemplate.from_template(config.PROMPT_TEMPLATE).format(
            context="\n\n---\n\n".join([doc.page_content for doc, _score in results]),
            question=queryText
        )
        
        # Get response with confidence check
        try:
            if model is not None:
                self.model = ChatOllama(model=model, temperature=0)
            response_text = self.model.invoke(prompt)
            print(f"Response: {response_text.content}")
            return response_text.content
            
        except Exception as e:
            print(f"Error: {e}")
            return f"Error: {e}"
            
