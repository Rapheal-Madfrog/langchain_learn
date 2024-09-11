from abc import ABC

from langchain.llms.base import LLM
from openai import OpenAI
from typing import Any, List, Optional, Iterator, AsyncIterator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessageChunk, AIMessage
from langchain_core.outputs import ChatResult
from langchain_community.adapters.openai import convert_message_to_dict
from langchain_core.outputs import ChatGenerationChunk, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.runnables import run_in_executor

client: Optional[OpenAI] = None
model: str = None


def init(api_key, base_url, m):
    global client
    global model
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    model = m


class GptLikeLLM(LLM):
    # 基于OpenAi接口 自定义 LLM 类

    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.95
    max_tokens: Optional[float] = 500

    def __init__(self, model, api_key, base_url):
        super().__init__()
        print(f"正在从OpenAi Api加载client..., {model=}")
        init(api_key=api_key, base_url=base_url, m=model)
        print("完成client的加载")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 重写调用函数
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            # logprobs=logprobs,
            # top_logprobs=top_logprobs,
            # stream=stream,
            # stream_options=stream_options,
            **kwargs
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "GptLikeLLM"


class ChatGptLikeLLM(BaseChatModel, ABC):

    def __init__(self, model, api_key, base_url):
        super().__init__()
        print(f"正在从OpenAi Api加载client..., {model=}")
        init(api_key=api_key, base_url=base_url, m=model)
        print("完成client的加载")

    @property
    def _llm_type(self) -> str:
        return "ChatGptLikeLLM"

    def _gen(self, messages: List[Any]) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content

    def _gen_stream(self, messages: List[Any]) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        messages = [convert_message_to_dict(m) for m in messages]
        response = self._gen(messages)
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        messages = [convert_message_to_dict(m) for m in messages]
        tokens = self._gen_stream(messages)

        for token in tokens:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))

            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

    async def _astream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        result = await run_in_executor(
            None,
            self._stream,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        for chunk in result:
            yield chunk
