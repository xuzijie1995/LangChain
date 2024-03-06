from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import BaseTool

class CalculatorTool(BaseTool):
    name = "Calculator"
    description = "如果问数学相关问题时，使用这个工具"
    return_direct = False  # 直接返回结果

    def _run(self, query: str) -> str:
        return eval(query)

from typing import Dict, Union, Any, List
import re
from langchain.output_parsers.json import parse_json_markdown
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.agents import AgentExecutor, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

# 自定义解析类
class CustomOutputParser(AgentOutputParser):

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        print(text)
        cleaned_output = text.strip()
        # 定义匹配正则
        action_pattern = r'"action":\s*"([^"]*)"'
        action_input_pattern = r'"action_input":\s*"([^"]*)"'
        # 提取出匹配到的action值
        action = re.search(action_pattern, cleaned_output)
        action_input = re.search(action_input_pattern, cleaned_output)
        if action:
            action_value = action.group(1)
        if action_input:
            action_input_value = action_input.group(1)
        
        # 如果遇到'Final Answer'，则判断为本次提问的最终答案了
        if action_value and action_input_value:
            if action_value == "Final Answer":
                return AgentFinish({"output": action_input_value}, text)
            else:
                return AgentAction(action_value, action_input_value, text)

        # 如果声明的正则未匹配到，则用json格式进行匹配
        response = parse_json_markdown(text)
        
        action_value = response["action"]
        action_input_value = response["action_input"]
        if action_value == "Final Answer":
            return AgentFinish({"output": action_input_value}, text)
        else:
            return AgentAction(action_value, action_input_value, text)
output_parser = CustomOutputParser()


from langchain.memory import ConversationBufferMemory
from langchain.agents.conversational_chat.base import ConversationalChatAgent 
from langchain.agents import AgentExecutor, AgentOutputParser

SYSTEM_MESSAGE_PREFIX = """尽可能用中文回答以下问题。您可以使用以下工具"""

# 初始化大模型实例，可以是本地部署的，也可是是ChatGPT
# llm = ChatGLM(endpoint_url="http://你本地的实例地址")
llm = ChatOpenAI(request_timeout=60)
# 初始化工具
tools = [CalculatorTool()]
# 初始化对话存储，保存上下文
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# 配置agent
chat_agent = ConversationalChatAgent.from_llm_and_tools(
    system_message=SYSTEM_MESSAGE_PREFIX, # 指定提示词前缀
    llm=llm, tools=tools, memory=memory, 
    verbose=True, # 是否打印调试日志，方便查看每个环节执行情况
    output_parser=output_parser # 
)
agent = AgentExecutor.from_agent_and_tools(
    agent=chat_agent, tools=tools, memory=memory, verbose=True,
    max_iterations=3 # 设置大模型循环最大次数，防止无限循环
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{text}"),
    ]
)
chain = _prompt | agent
# 判断 text 是否有值
# if "{text}":
#     chain = agent.run("{text}")  # 如果 text 有值，则调用 agent.run("{text}")
#     # 在这里可以添加处理 chain 的逻辑
#     print(chain)  # 示例：打印 chain 的结果
# else:
#     print("文本内容为空，不调用 agent.run()")

# 在用户输入后触发操作
# def wait_for_user_input():
#     while True:
#         user_input = input("请输入您的问题或指令：")  # 等待用户输入
#         if user_input.lower() == "exit":  # 如果用户输入"exit"，则退出循环
#             break
#         # 在此处添加您希望执行的操作，可以调用相应的函数或方法
#         chain = agent.run(user_input)  # 调用您的代码逻辑，例如运行代理程序
#         print(chain)  # 打印结果或进行其他操作

# # 调用等待用户输入函数
# wait_for_user_input()
# chain = agent.run("{text}")


# _prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "Translate user input into pirate speak",
#         ),
#         ("human", "{text}"),
#     ]
# )
# _model = ChatOpenAI()

# # if you update this, you MUST also update ../pyproject.toml
# # with the new `tool.langserve.export_attr`
# chain = _prompt | _model
