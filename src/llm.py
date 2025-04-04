import argparse
from ast import main
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from database.localDatabase import createDatabase, removeExistingDatabase
from localOllama.ollamaModel import OllamaModel
import uvicorn
import database.api as api
import config.config as config


# Load environment variables
load_dotenv()


def main():

    parser = argparse.ArgumentParser(description="Query the Chroma database")
    parser.add_argument(
        "--new-db", action="store_true", help="Force creation of a new database"
    )
    args = parser.parse_args()

    if args.new_db:
        print("Forcing creation of new database...")
        removeExistingDatabase()

    ollamaModel = OllamaModel()
    ollamaModel.setOllamaModel()
    db = createDatabase(ollamaModel)
    api.db = db
    api.ollamaModel = ollamaModel
  
    uvicorn.run(api.app, port=8000)
    
    # try:
    #     while db is not None:
    #         queryText = input("\nEnter your question (or 'quit' to exit): ")
    #         match queryText.lower():
    #             case "quit":
    #                 break
    #             case "cm":
    #                 ollamaModel.setOllamaModel(True)
    #                 db = createDatabase(ollamaModel)
    #                 continue
    #             case "model":
    #                 print(f"Installed models:")
    #                 for index, model in enumerate(ollamaModel.client.list()['models']):
    #                     print(f"{index + 1}. {model['model']}")
    #                 print(f"Current model: {ollamaModel.modelName}")
    #                 continue
    #             case _:
    #                 queryDatabase(queryText, db, ollamaModel)

    # except Exception as e:
    #     print(f"{e}")
    #     return


def queryDatabase(
    queryText: str, db: Chroma, ollamaModel: OllamaModel, n_results: int = 5
):
    # Search the database
    try:
        results = db.similarity_search_with_relevance_scores(queryText, k=n_results)
    except Exception as e:
        print(f"Error: {e}")
        return []

    prompt = ChatPromptTemplate.from_template(config.PROMPT_TEMPLATE).format(
        context="\n\n---\n\n".join([doc.page_content for doc, _score in results]),
        question=queryText
    )
    
    # Get response with confidence check
    try:
        response_text = ollamaModel.model.invoke(prompt)
        print(f"Response: {response_text.content}")
        
    except Exception as e:
        print(f"Error: {e}")
   
if __name__ == "__main__":
    main()
