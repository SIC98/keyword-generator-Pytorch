import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "bart-coco"
encoder_max_length = 512
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def greet(keywords):

    inputs = tokenizer(
        keywords,
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_str[0]


demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(lines=5, placeholder="Keywords Here..."),
    outputs=gr.Textbox(lines=5)
)

demo.launch()
