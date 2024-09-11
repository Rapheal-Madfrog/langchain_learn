from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import os
from utils.yaml_utils import load_yaml_with_env
from utils.models_gpt_like_llm import GptLikeLLM, ChatGptLikeLLM
from dacite import from_dict
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Tongyi, QianfanLLMEndpoint
from langchain_community.chat_models import ChatTongyi, QianfanChatEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
# import google.generativeai as genai


@dataclass
class GptLikeConfig:
    base_url: str = field(default='')
    keys: Dict[str, str] = field(default_factory=dict)
    base_models: None | List[str] = field(default_factory=list)
    chat_models: None | List[str] = field(default_factory=list)
    emb_models: None | List[str] = field(default_factory=list)


@dataclass
class QianFanConfig:
    keys: Dict[str, Dict[str, str]] = field(default_factory=dict)
    base_models: None | List[str] = field(default_factory=list)
    chat_models: None | List[str] = field(default_factory=list)
    emb_models: None | List[str] = field(default_factory=list)


@dataclass
class GeminiConfig:
    keys: Dict[str, str] = field(default_factory=dict)
    models: None | List[str] = field(default_factory=list)


@dataclass
class Config:
    """
    Configuration class for the project.
    This project is about langchain for language modeling.
    This config aims to provide config for openai client.
    """
    openai: GptLikeConfig
    ali: GptLikeConfig
    siliconflow: GptLikeConfig
    gemini: GeminiConfig
    qian_fan: QianFanConfig

    def __init__(self, config_path: str = None):
        default_dir = os.path.dirname(__file__)
        config_path = config_path or os.path.join(default_dir, "api_config.yml")
        config = load_yaml_with_env(config_path)

        models_config_path = config_path.replace("api_config.yml", "valid_models.yml")
        models_config = load_yaml_with_env(models_config_path)

        for platform in config:
            for model_g in models_config.get(platform, {}):
                config[platform][model_g] = models_config.get(platform, {}).get(model_g, [])

        self.openai = from_dict(GptLikeConfig, config["openai"])
        self.ali = from_dict(GptLikeConfig, config["ali"])
        self.siliconflow = from_dict(GptLikeConfig, config["siliconflow"])
        self.gemini = from_dict(GeminiConfig, config["gemini"])
        self.qian_fan = from_dict(QianFanConfig, config["qianfan"])

    def get_client(self, platform_name, mode: str = 'base', key_name=None, model_name=None, **kwargs):
        key_name = key_name or 'default'
        if platform_name == "openai":
            os.environ['OPENAI_API_KEY'] = self.openai.keys[key_name]
            if mode == 'base':
                return OpenAI()
            elif mode == 'chat':
                return ChatOpenAI()
            else:
                return OpenAIEmbeddings()
        elif platform_name == "ali":
            os.environ['DASHSCOPE_API_KEY'] = self.ali.keys[key_name]
            if mode == 'base':
                llm = Tongyi()
                ml = self.ali.base_models
            elif mode == 'chat':
                llm = ChatTongyi()
                ml = self.ali.chat_models
            else:
                raise ValueError("Ali platform does not support embeddings.")
            llm.model_name = model_name if (model_name is not None and model_name in ml) else ml[0]
            return llm
        elif platform_name == "siliconflow":

            if mode == 'base':
                model_name = model_name if (model_name is not None and model_name in self.siliconflow.base_models) else self.siliconflow.base_models[0]
                llm = GptLikeLLM(model_name, self.siliconflow.keys[key_name], self.siliconflow.base_url)
            elif mode == 'chat':
                model_name = model_name if (model_name is not None and model_name in self.siliconflow.chat_models) else self.siliconflow.chat_models[0]
                llm = ChatGptLikeLLM(model_name, self.siliconflow.keys[key_name], self.siliconflow.base_url)
            else:
                raise ValueError("Siliconflow platform does not support embeddings.")
            return llm
        elif platform_name == 'qianfan':
            os.environ['QIANFAN_AK'] = self.qian_fan.keys[key_name]['ak']
            os.environ['QIANFAN_SK'] = self.qian_fan.keys[key_name]['sk']
            if mode == 'base':
                model_name = model_name if (model_name is not None and model_name in self.qian_fan.base_models) else self.qian_fan.base_models[0]
                llm = QianfanLLMEndpoint(model=model_name, **kwargs)
            elif mode == 'chat':
                model_name = model_name if (model_name is not None and model_name in self.qian_fan.chat_models) else self.qian_fan.chat_models[0]
                llm = QianfanChatEndpoint(model=model_name, **kwargs)
            else:
                llm = QianfanEmbeddingsEndpoint()
            return llm
        # elif platform_name == 'gemini':
        #     genai.configure(api_key=self.gemini.keys[key_name])
        #     model_name = model_name if (model_name is not None and model_name in self.gemini.models) else self.gemini.models[0]
        #     return genai.GenerativeModel(model_name)
        else:
            raise ValueError(f"Unsupported platform: {platform_name}")



