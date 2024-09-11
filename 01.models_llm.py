from config.config import Config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

config = Config()
llm = config.get_client('siliconflow', 'chat')

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是瓜皮, 是一个专业的小丑演员.'),
    ('user', '{input}'),
])

parser = StrOutputParser()
chain = prompt | llm | parser
print(chain.invoke({'input': '你是谁?'}))

stream_response = chain.stream({'input': '你是谁?'})
for chunk in stream_response:
    print(chunk, end='', flush=True)

