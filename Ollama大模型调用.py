import ollama

client = ollama.Client(host="http://127.0.0.1:11434")

print(client.list())

print(client.show("deepseek-r1:1.5b"))

print(client.ps())


while True:
    print("\n")
    input("请输入你的问题：")
    resp = client.chat(
        model="deepseek-r1:1.5b",
        messages=[{"role": "user", "content": "你是谁"}]
    )
    print(resp.message.content)
