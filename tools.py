from langchain_community.tools import wikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import wikipendiaApiWrapper
from langchain.tools import Tool
from datetime import datetime

#saving file function
def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)


search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="search the web for information",
)

api_wrapper = wikipendiaApiWrapper(top_k_results=3, doc_content_length=100)
wiki_tool = wikipediaQueryRun(api_wrapper=api_wrapper)