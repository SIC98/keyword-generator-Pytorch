from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr
import argparse
import urllib
from PIL import Image

from karlo_api import t2i
from utils import get_data_until_kth_comma


parser = argparse.ArgumentParser()

parser.add_argument('--model_name_or_path', type=str)
parser.add_argument('--device', type=str)
parser.add_argument('--fp16', action='store_true')

args = parser.parse_args()

tokenizer = GPT2Tokenizer.from_pretrained(
    args.model_name_or_path
)

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained(
    args.model_name_or_path, pad_token_id=tokenizer.eos_token_id
)

model = model.to(args.device)
if args.fp16:
    model = model.half()


def greet(keywords, total_keywords):

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--fp16', action='store_true')

    args = parser.parse_args()

    encodings_dict = tokenizer(
        keywords,
        max_length=1024,
        return_tensors="pt",
    )
    input_ids = encodings_dict.input_ids.to(args.device)
    output_sequences = model.generate(input_ids, max_length=1024)
    generated_text = tokenizer.decode(
        output_sequences[0], skip_special_tokens=True
    )
    generated_text = get_data_until_kth_comma(
        generated_text, total_keywords, False
    )

    response = t2i(generated_text, "")

    karlo_image = Image.open(urllib.request.urlopen(
        response.get("images")[0].get("image")))

    return generated_text, karlo_image


demo = gr.Interface(
    fn=greet,
    inputs=[gr.Textbox(lines=4, placeholder="Keywords Here"),
            gr.Slider(1, 20, step=1)],
    outputs=[gr.Textbox(lines=4),  gr.Image(shape=(200, 200))]
)

demo.launch()
