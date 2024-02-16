import autogen


config_list = [
    {
        "model":"llama2",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key" : "ollama",
    }
]

llm_config = {"config_list" : config_list}

assistant = autogen.AssistantAgent("assistant",llm_config = llm_config,system_message="You are helpfull assistant")

user_proxy = autogen.UserProxyAgent("user_proxy",code_execution_config = {"use_docker":False})

user_proxy.initiate_chat(assistant,message ="What is the name of the model you are based on?")


