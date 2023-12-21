# keyword-generator-Pytorch

Generate keywords describing an image in an autoregressive manner.

| Input | Generated keywords |
| ----- | ------------------ |
| young, curly haired, | young, curly haired, redhead Natalie Portman as a heroine with a piercing gaze wearing a oversized t-shirt and oversized jeans, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus |
| molly millions, portrait of a beautiful cyberpunk woman, | molly millions, portrait of a beautiful cyberpunk woman, cyberpunk city background, megacity, gorgeous view, depth, painted by seb mckinnon, high detail, digital art, painted in the style of greg rutkowski, trending on artstation, 4k, vivid colors, ultra realistic, sharp focus, high definition, depth of field |
| cyborg sweating water, big drops of sweat, | cyborg sweating water, big drops of sweat, diffuse lighting, fantasy, intricate, elegant, highly detailed, lifelike, photorealistic, digital painting, artstation, illustration, concept art, smooth, sharp focus, art by John Collier and Albert Aublet and Krenz Cushart and Artem Demura and Alphonse Mucha and Giuseppe Arcimboldo and Bobby Chiu and Kevin Swartz and Greg Rutkowski and Alphons Mucha, masterpiece |

## Features
1. Use crawled Lexica dataset: [Gustavosta/Stable-Diffusion-Prompts](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts)
2. Fine-tuning GPT2 model based on `run_clm.py` script and run text generation by running `run_generation.py` script with some modifications.
    - The original code groups the dataset in chunks of 1,000 using the `group_texts` function and concatenates them. During this process, our dataset utilizes multiple prompts at once as input to the model. This is not the method I intended for training. I removed that code and implemented an appropriate token preprocessing function.
    - The GPT2 vocab has only one special token, which is the `eos_token` used for padding.
    - An accuracy metric function is used. Padding is not considered in the calculation.
    - I implemented a metrics function for evaluating using the ROUGE score. The function does not use the entire generated text up to the maximum length. I aligned the number of keywords in the generated text with the Test dataset, and then evaluation is performed.
 3. I achieved an evaluation accuracy of 81.01% from test dataset.

## Generation strategy

| ROUGE-L      | Greedy search | No repeat ngram size=3 | No repeat ngram size=3, beam size=3 | Temperature=0.7 | Top-k=50 | Top-p=0.9 |
| ------------ | ------------- | ---------------------- | ------------------------------------ | --------------- | -------- | --------- |
| One Keyword  | 39.60         | 40.23                  | 39.71                                | 37.20           | 34.10    | 35.86     |
| Two Keywords | 49.25         | 49.88                  | 49.65                                | 47.60           | 44.57    | 46.76     |

- In open-ended text generation, top-k or top-p sampling can be a better strategy than greedy or beam search, but greedy search is the best according to the ROUGE-L criterion in my task.
- If you want creative results, greedy search may not be appropriate.
- In my dataset, the proportion of data containing duplicated bigrams is 13.71%, and the proportion with duplicated trigrams is 2.6%. Therefore, I have set the 'No repeat ngram size' value to 3. The duplicated trigram helps prevent the repetition of keywords.

## Result

Using the Karlo API, it's possible to generate images using the keywords created by the model as prompts.

![result](https://github.com/SIC98/keyword-generator-Pytorch/assets/51232785/e474e912-374e-40ff-9f91-e2af45313f40)

## How to run my code

1. Fine-tuning GPT2 model
```
python run_clm.py \
	--model_name_or_path=gpt2 \
	--dataset_name=Gustavosta/Stable-Diffusion-Prompts \
	--per_device_train_batch_size=16 \
	--per_device_eval_batch_size=16 \
	--torch_dtype=bfloat16 \
	--num_train_epochs=10 \
	--learning_rate=4e-4 \
	--do_train \
	--do_eval \
	--output_dir=results \
	--save_steps=4000 \
	--evaluation_strategy=steps \
	--eval_steps=4000
```
2. Evaluate the model's ROUGE score. It should be run after executing run_clm.py. My code evaluates the model using beam search (beam size = 5). You can test other generation strategies by modifying the `batch_inference()` function in `utils.py`.
```
python eval.py \
	--model_name_or_path=<path_of_ckpt> \
	--input_type="two_keyword" \
	--batch_size=128 \
	--fp16 \
	--device="cuda"
```
3. Perform GPT2 model inference.
```
python run_generation.py \
	--model_name_or_path=<ckpt_path> \
	--fp16 \
  --model_type=gpt2 \
	--prompt=<prompt> \
	--length=<length> \
	--no_repeat_ngram_size=3
```
4. In `karlo_api.py`, after entering the `REST_API_KEY`, you can run the Gradio app.
```
python run_gradio.py \
	--model_name_or_path=<path_of_ckpt> \
	--fp16 \
	--device="cuda"
```
