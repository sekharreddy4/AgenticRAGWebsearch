import os
from typing import TypedDict, List, Annotated, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
#from dotenv import load_dotenv

# Load environment variables (make sure to set GROQ_API_KEY in your .env file)
#load_dotenv()


# 1. Define State Schema
class ConversationState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], "chat_history"]
    pending_questions: Annotated[List[str], "questions_needed"]
    user_response: Annotated[str, "latest_user_input"]
    search_context: Annotated[List[str], "retrieved_info"]
    final_answer: Annotated[str, "final_response"]


groqapikey = os.environ.get('api-key')
print(groqapikey)

# 2. Initialize Components
llm = ChatGroq(temperature=0.1,
               model_name="llama-3.3-70b-versatile",
               api_key=groqapikey)
search_tool = DuckDuckGoSearchRun()

# Initialize document retriever (FAISS)
documents = [
    "Company policy states all employees must complete annual compliance training.",
    "GDPR requires explicit consent for data collection in the EU.",
    "HIPAA mandates strict protection of patient health information in the US."
]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_texts(documents, embeddings)
doc_retriever = vectorstore.as_retriever()

# 3. Build Workflow Graph
builder = StateGraph(ConversationState)


# Define Nodes
def input_processor(state: ConversationState):
    """Handle user input and update state"""
    if state["user_response"]:
        state["messages"].append(HumanMessage(content=state["user_response"]))
    return state


def information_check(state: ConversationState):
    """Determine if we need more info"""
    prompt = f"""
    Current conversation: {state['messages'][-2:]}
    Retrieved context: {state['search_context']}

    Should we:
    1. Ask clarifying questions (return 'questions')
    2. Provide final answer (return 'answer')
    3. Need to search more (return 'search')
    """
    decision = llm.invoke(prompt).content.lower()
    if 'question' in decision:
        return {"decision": "questions"}
    elif 'answer' in decision:
        return {"decision": "answer"}
    else:
        return {"decision": "search"}




def question_generator(state: ConversationState):
    """Generate follow-up questions"""
    prompt = f"""
    Based on: {state['messages']}
    Missing information: {state['pending_questions']}

    Generate 1-2 concise clarifying questions.
    """
    questions = llm.invoke(prompt).content.split("\n")
    state["pending_questions"] = questions
    return state


def search_handler(state: ConversationState):
    """Execute web/document search"""
    query = state["messages"][-1].content
    try:
        web_results = search_tool.run(query)
    except Exception as e:
        print(f"Web search error: {e}")
        web_results = ""

    try:
        doc_results = [doc.page_content for doc in doc_retriever.get_relevant_documents(query)]
    except Exception as e:
        print(f"Document retrieval error: {e}")
        doc_results = []

    state["search_context"] = [web_results] + doc_results
    return state


def answer_generator(state: ConversationState):
    """Synthesize final response"""
    context = "\n".join(state["search_context"])
    prompt = f"""
    Answer using: {context}
    Conversation history: {state['messages']}
    Provide a concise and informative response.
    """
    state["final_answer"] = llm.invoke(prompt).content
    return state


# Add Nodes
builder.add_node("process_input", input_processor)
builder.add_node("check_info", information_check)
builder.add_node("generate_questions", question_generator)
builder.add_node("perform_search", search_handler)
builder.add_node("generate_answer", answer_generator)

# Define Edges
builder.set_entry_point("process_input")
builder.add_edge("process_input", "check_info")

builder.add_conditional_edges(
    "check_info",
    lambda x: x["decision"],
    {
        "questions": "generate_questions",
        "answer": "generate_answer",
        "search": "perform_search"
    }
)


builder.add_edge("generate_questions", END)
builder.add_edge("perform_search", "check_info")
builder.add_edge("generate_answer", END)


# Add Feedback Loop
def response_handler(state: ConversationState):
    """Handle bot's response to user"""
    if state["final_answer"]:
        return {"messages": state["messages"] + [AIMessage(content=state["final_answer"])]}
    else:
        questions = "\n".join(state["pending_questions"])
        return {"messages": state["messages"] + [AIMessage(content=questions)]}


builder.add_node("handle_response", response_handler)
builder.add_edge("generate_questions", "handle_response")
builder.add_edge("generate_answer", "handle_response")

# Compile Graph
workflow = builder.compile()


# 4. Interactive Chat Loop
def chat():
    state = {
        "messages": [],
        "pending_questions": [],
        "user_response": "",
        "search_context": [],
        "final_answer": ""
    }

    print("Welcome to the Compliance Chatbot! (Type 'exit' to end the conversation)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for using the Compliance Chatbot. Goodbye!")
            break

        state["user_response"] = user_input
        result = workflow.invoke(state)

        if result["final_answer"]:
            print(f"Bot: {result['final_answer']}")
            state = {
                "messages": result["messages"],
                "pending_questions": [],
                "user_response": "",
                "search_context": [],
                "final_answer": ""
            }
        else:
            print(f"Bot: {result['messages'][-1].content}")
            state = result


if __name__ == "__main__":
    chat()
