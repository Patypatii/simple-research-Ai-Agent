from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ReaserchResponse(BaseModel):
    topic: str
    summary: str
    aources: list[str]
    tools_used:list[str]


#llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

parser = PydanticOutputParser(pydantic_object=ReaserchResponse)

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

#search tools
tools = [search_tool, wiki_tool, save_tool]
#creating and running the agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
tools=tools
    
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[],
    verbose=True,
)
query = "What is the impact of climate change on polar bear populations?"
raw_response = agent_executor.invoke({"query": query})


try:
    structured_response = parser.parse(raw_response("output")[0]["text"])
    print(structured_response.topic)

except Exception as e:
    print("Error parsing response:" ,e, "\nRaw response:", raw_response)
