import openai
import time


client = openai.OpenAI(
    base_url="https://yunwu.ai/v1",
    # 测o1，就用这个api
    # api_key=""
#     测3.5，4，4o，4omini之类的用这个
    api_key=""
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

prompt = "下面是一个算子的hip程序，我只提供了核函数代码。帮我完成以下任务：1.针对核函数代码进行优化，并给出对应的中文注释；2.只给我返回优化后的代码和中文优化建议。3.优化建议的小标题为优化建议,小标题大小与内容一致，加粗就行"

def improve_hip_code(code):
    user_input = prompt+code
    print(user_input)
    messages.append({"role": "user", "content": user_input})
    print(f"已发送请求")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=1
    )
    assistant_response = response.choices[0].message.content
    print(f"Assistant: {assistant_response}")

    messages.append({"role": "assistant", "content": assistant_response})
    return assistant_response


if __name__ == "__main__":
    with open("../question.txt", "r", encoding='utf-8') as file:
        user_input = file.read().strip()
    # user_input = input("You: ")
    print(user_input)
    messages.append({"role": "user", "content": user_input})
    print(f"已发送请求")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=1
    )
    assistant_response = response.choices[0].message.content
    print(f"Assistant: {assistant_response}")

    messages.append({"role": "assistant", "content": assistant_response})
    # time.sleep(1)