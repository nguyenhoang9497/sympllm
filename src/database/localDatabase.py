import os
import shutil
import glob
import json
import config.config as config
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from localOllama.ollamaModel import OllamaModel

def getLatestDocumentTime():
    """Get the latest modification time of any document in the data directory"""
    latestTime = 0
    for file in getCurrentDocuments():
        mtime = os.path.getmtime(file)
        latestTime = max(latestTime, mtime)
    return latestTime

def getDatabaseTime():
    """Get the latest modification time of the database files"""
    if not os.path.exists(config.CHROMA_PATH):
        return 0
    latestTime = 0
    for root, _, files in os.walk(config.CHROMA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            mtime = os.path.getmtime(file_path)
            latestTime = max(latestTime, mtime)
    return latestTime

def getCurrentDocuments():
    """Get list of all PDF files currently in the data directory"""
    return [os.path.abspath(f) for f in glob.glob(os.path.join(config.DOCUMENT_PATH, "**/*.pdf"), recursive=True)]

def getProcessedDocuments():
    """Get list of documents that were processed in the last database creation"""
    if not os.path.exists(config.PROCESSED_FILES_PATH):
        return []
    try:
        with open(config.PROCESSED_FILES_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not read processed files list: {e}")
        return []

def saveProcessedDocuments(documents):
    """Save list of processed documents"""
    os.makedirs(os.path.dirname(config.PROCESSED_FILES_PATH), exist_ok=True)
    with open(config.PROCESSED_FILES_PATH, 'w') as f:
        json.dump(documents, f)

def checkIfDatabaseNeedUpdate():
    """Check if the database needs to be updated based on document changes or deletions"""
    if not os.path.exists(config.CHROMA_PATH):
        return True
    
    current_docs = set(getCurrentDocuments())
    processed_docs = set(getProcessedDocuments())
    
    # Check for add/remove documents
    if processed_docs != current_docs:
        return True
    
    # Check for new or modified documents
    doc_time = getLatestDocumentTime()
    db_time = getDatabaseTime()
    
    if doc_time > db_time:
        return True
    
    return False

def loadDocuments():
    return PyPDFDirectoryLoader(config.DOCUMENT_PATH).load()

def splitTextIntoChunk():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    ).split_documents(loadDocuments())

def removeExistingDatabase():   
    if os.path.exists(config.CHROMA_PATH):
        try:
            # Now try to remove the directory
            shutil.rmtree(config.CHROMA_PATH)
            print(f"Removed existing database at {config.CHROMA_PATH}")
        except Exception as e:
            print(f"Warning: Could not remove existing database: {e}")
            

def createDatabase(ollamaModel: OllamaModel) -> Chroma:
    if os.path.exists(config.CHROMA_PATH) and not checkIfDatabaseNeedUpdate():
        print(f"Loading existing database from {config.CHROMA_PATH}")
        return Chroma(
            persist_directory=config.CHROMA_PATH,
            embedding_function=ollamaModel.ollamaEmbedding()
        )
        
    else:
        print(f"Creating new database in {config.CHROMA_PATH}")
        if os.path.exists(config.CHROMA_PATH):
            removeExistingDatabase()
        
        # Create new database
        try:
            db = Chroma.from_documents(
                documents=splitTextIntoChunk(),
                embedding=ollamaModel.ollamaEmbedding(),
                persist_directory=config.CHROMA_PATH,
            )
        except Exception as e:
            print(f"Error creating database: {e}")
            return None
        
        # Save list of processed documents
        saveProcessedDocuments(getCurrentDocuments())
        
        return db
