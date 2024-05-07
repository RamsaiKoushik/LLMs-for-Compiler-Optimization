# %%

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "josh-ops"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

# # Prompt content: "hi"
messages = [
    {"role": "user", "content":"#include<iostream>\n\n#include<cstring>\n\nusing namespace std;\n\nint main()\n\n{\n\n\tint k;\n\n\tstring s;\n\n\tcin>>k>>s;\n\n\tif(s.length()>k)\n\n\t{\n\n\t\tfor(int i=0;i<k;i++)\n\n\t\tcout<<s[i];\n\n\t\tcout<<\"...\";\n\n\t}\n\n\telse\n\n\tcout<<s;\n\n}  \n\n# optimized version of the same code:", 
}
]

# messages = [
#     # {"role": "user", "content":"how can you help me?"}
# ]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt',max_length=200)
output_ids = model.generate(input_ids.to('cuda'),max_length=500)
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)

# %%



