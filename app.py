from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
#os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    max_new_tokens=512,
    temperature=0.3,
)

model = ChatHuggingFace(llm=llm)

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url=input("Enter the URL: ")
print("\nPage loaded and indexed. Ask questions about it (type 'exit' to quit).\n")

#loader = WebBaseLoader(url)
loader = PlaywrightURLLoader(urls=[url], remove_selectors=["header", "footer"])
docs = loader.load()
print(docs[0].page_content[:1500])

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


chain = prompt | model | parser


page_text=docs[0].page_content
while True:
    question = input("Q: ")
    if question.lower() in ('exit', 'quit'):
        break

    relevant_docs = retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in relevant_docs)

    answer = chain.invoke({'question': question, 'text': context_text})
    print(f"A: {answer}\n")


print("Thankyou🤗")