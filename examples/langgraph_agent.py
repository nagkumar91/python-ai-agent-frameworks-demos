# https://github.com/JRAlexander/IntroToAgents1-Oxford/blob/main/intro-langgraph/time-travel.ipynb
import os

import azure.identity
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_azure_ai.callbacks.tracers import AzureAIInferenceTracer

load_dotenv(override=True)

# Determine the actual endpoint based on API_HOST
def get_endpoint_url():
    api_host = os.getenv("API_HOST", "github")
    if api_host == "azure":
        return os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    elif api_host == "github":
        return "https://models.inference.ai.azure.com"
    elif api_host == "ollama":
        return os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1")
    else:
        return "https://api.openai.com/v1"

# Configure Azure OpenAI tracing with proper values
azure_tracer = AzureAIInferenceTracer(
    connection_string=os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING"),
    enable_content_recording=os.getenv("OTEL_RECORD_CONTENT", "true").lower() == "true",
    name="Music Player Agent",
    id="music_agent_010",
    endpoint=get_endpoint_url(),
    scope="Entertainment Services"
)

@tool
def play_song_on_spotify(song: str):
    """Play a song on Spotify"""
    # Call the spotify API ...
    return f"Successfully played {song} on Spotify!"


@tool
def play_song_on_apple(song: str):
    """Play a song on Apple Music"""
    # Call the apple music API ...
    return f"Successfully played {song} on Apple Music!"


tools = [play_song_on_apple, play_song_on_spotify]
tool_node = ToolNode(tools)

# Setup the client to use either Azure OpenAI or GitHub Models
API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(
        azure.identity.DefaultAzureCredential(), 
        "https://cognitiveservices.azure.com/.default"
    )
    model = AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        azure_ad_token_provider=token_provider,
    )
elif API_HOST == "github":
    model = ChatOpenAI(
        model=os.getenv("GITHUB_MODEL", "gpt-4o"), 
        base_url="https://models.inference.ai.azure.com", 
        api_key=os.environ.get("GITHUB_TOKEN")
    )
elif API_HOST == "ollama":
    model = ChatOpenAI(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"), 
        base_url=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1"), 
        api_key="none"
    )
else:
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

model = model.bind_tools(tools, parallel_tool_calls=False)


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Set up memory
memory = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable

# We add in `interrupt_before=["action"]`
# This will add a breakpoint before the `action` node is called
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}, "callbacks": [azure_tracer]}
input_message = HumanMessage(content="Can you play Taylor Swift's most popular song?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()