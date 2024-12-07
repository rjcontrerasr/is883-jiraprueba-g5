import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from datetime import date
import pandas as pd
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain import hub

# Title
st.title("💬 Financial Support Chatbot")

# Load the dataset
url = "https://raw.githubusercontent.com/JeanJMH/Financial_Classification/main/Classification_data.csv"
st.write(f"Using dataset from: {url}")

try:
    df1 = pd.read_csv(url)
except Exception as e:
    st.error(f"An error occurred while loading the dataset: {e}")

# Extract unique product categories
product_categories = df1['Product'].unique().tolist()

# Initialize memory and the chatbot on the first run
if "memory" not in st.session_state:
    model_type = "gpt-4o-mini"

    # Initialize memory for the conversation
    max_number_of_exchanges = 10
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=max_number_of_exchanges, return_messages=True
    )

    # Initialize the language model
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

    # Tool: Today's Date
    from langchain.agents import tool

    @tool
    def classify_complaint(complaint: str) -> str:
        """Classifies a complaint based on the dataset."""
        # Classify by Product
        product_match = None
        for product in product_categories:
            if product.lower() in complaint.lower():
                product_match = product
                break

        if not product_match:
            return "I'm sorry, I couldn't classify the complaint into a product category. Please provide more details."

        # Classify by Sub-product
        subproduct_options = df1[df1['Product'] == product_match]['Sub-product'].unique()
        subproduct_match = None
        for subproduct in subproduct_options:
            if subproduct.lower() in complaint.lower():
                subproduct_match = subproduct
                break

        if not subproduct_match:
            subproduct_match = "No specific sub-product match found."

        # Classify by Issue
        issue_options = df1[(df1['Product'] == product_match) &
                            (df1['Sub-product'] == subproduct_match)]['Issue'].unique()
        issue_match = None
        for issue in issue_options:
            if issue.lower() in complaint.lower():
                issue_match = issue
                break

        if not issue_match:
            issue_match = "No specific issue match found."

        # Format the classification response
        return (
            f"Complaint classified as:\n"
            f"- **Product:** {product_match}\n"
            f"- **Sub-product:** {subproduct_match}\n"
            f"- **Issue:** {issue_match}"
        )

    tools = [classify_complaint]

    # Prompt for complaint classification
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are a financial support assistant. Begin by greeting the user warmly and asking them to describe their issue. "
                f"Once the issue is described, classify the complaint strictly based on these possible categories: {product_categories}. "
                f"Use the tool to classify complaints accurately. Inform the user that a ticket has been created and provide the classification. "
                f"Reassure them that the issue will be forwarded to the appropriate support team. "
                f"Maintain a professional and empathetic tone throughout."
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent with memory
    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=tools, memory=st.session_state.memory, verbose=True
    )

# Display chat history
for message in st.session_state.memory.buffer:
    st.chat_message(message.type).write(message.content)

# Chat input
if user_input := st.chat_input("How can I help?"):
    st.chat_message("user").write(user_input)

    # Generate response from the agent
    response = st.session_state.agent_executor.invoke({"input": user_input})["output"]

    # Display the assistant's response
    st.chat_message("assistant").write(response)


os.environ["JIRA_API_TOKEN"] = st.secrets["JIRA_API_TOKEN"]
os.environ["JIRA_USERNAME"] = "rich@bu.edu"
os.environ["JIRA_INSTANCE_URL"] = "https://is883-genai-r.atlassian.net/"
os.environ["JIRA_CLOUD"] = "True"
#os.environ["JIRA_PROJECT_KEY"] = "LLMTS"

#project_key = "LLMTS"

assigned_issue= "Managing an Account"
client_complaint = "I made a purchase and it was disputed"

question = (
    f"Create a task in my project with the key FST. Take into account tha the Key of this project is FST "
    f"The task's type is 'Task', assignee to rich@bu.edu,"
    f"The summary is '{assigned_issue}'."
    #f"with the priority '{priority}' and the description '{client_complaint}'. "
    f"Always assign 'Highest' priority if the '{assigned_issue}' is related to fraudulent activities. Fraudulent activities include terms or contexts like unauthorized access, theft, phishing, or stolen accounts. Be strict in interpreting fraud-related issues."
    f"with the priority 'High' for other type of issues"
    f"with the description '{client_complaint}'. "
    #f"with a status  'TO DO'. "
)

#agent_executor.invoke({"input": question}, handle_parsing_errors=True)

# Execute the agent to create the Jira task

# Initialize Jira API Wrapper and Toolkit
jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)

# Fix tool names and descriptions in the toolkit
for idx, tool in enumerate(toolkit.tools):
    toolkit.tools[idx].name = toolkit.tools[idx].name.replace(" ", "_")
    if "create_issue" in toolkit.tools[idx].name:
        toolkit.tools[idx].description += " Ensure to specify the project ID."

# Add tools for the agent
tools = toolkit.get_tools()

# LLM Setup for LangChain
chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model="gpt-4o-mini")

# Prepare the LangChain ReAct Agent
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(chat, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Create a question prompt for the Jira task
# priority = "Highest" if "fraud" in client_complaint.lower() else "High"


try:
    result = agent_executor.invoke({"input": question})
    print("Agent Output:", result)
except Exception as e:
    print(f"Error during Jira task creation: {e}")
