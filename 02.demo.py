from config.config import Config
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_tool
import json
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

config = Config()
llm = config.get_client('ali', True)


class Joke(BaseModel):
    Q: str = Field(description='问题')
    A: str = Field(description='回答')


# parser = JsonOutputParser(pydantic_object=Joke)
#
# prompt = PromptTemplate(
#     template='回答问题. \n{format_tex} \n{query} \n',
#     input_variable=['query'],
#     partial_variables={'format_tex': parser.get_format_instructions()}
# )
#
# chain = prompt | llm | parser

# joke = '中国的首都是?'
# print(chain.invoke({'query': joke}))


def multiply(a, b):
    return a * b


# print(json.dumps(convert_to_openai_tool(multiply), indent=2))
#
# llm_with_tools = llm.bind_tools([multiply])
# print(llm_with_tools.invoke('2 * 3'))


set_llm_cache(InMemoryCache())

print(llm.invoke('写一首四言绝句'))

print(llm.invoke('写一首四言绝句'))
