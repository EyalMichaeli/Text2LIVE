import openai

openai.api_key = "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"

first_message = "I want you to learn a new task, read carefully:\nI'm going to give you a class name, like: 'tiger', 'dog', 'plane', 'flower' etc.\nAnd I want you to come up with an adjective for this class, that would not fit EXACTLY to the image, rather change it just a bit, just the style of the class. \nThis should be random each time I ask.\n\na few good examples:\ntiger: old tiger\ntiger: big tiger\ntiger: white tiger\ndog: scary dog\ndog: fluffy dog\ndog: happy dog\ndog: muscular dog\ndog: skinny dog\ndog: black dog\nplane: old plane\nplane: new plane\nplane: big plane\nplane: white plane\nplane: dirty plane\nflower: pretty flower\nflower: a dying flower\nflower: a beautiful flower\ncar: shiny car\ncar: luxurious car\ncar: sports car\ncar: dirty car\ncar: clean car\n\na few bad examples:\ntiger: tiny tiger (changes the object itself, not just the style)\ntiger: fake tiger (too far from the truth)\ndog: German shepherd dog (might be too far from the truth)\ncat: transparent cat (doesn't make sense)\nplane: tiny plane (doesn't make sense)\nyou got the idea.\nPlease respond with ONLY the edited text, no need for any other text, until i say 'you are free'"

BOT_MESSAGES_IMAGENET = [
           {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": first_message},
        {"role": "assistant", "content": "Sounds like a fun game. Let's play!"},
        {"role": "user", "content": "car"},
        {"role": "assistant", "content": "shiny car"},
        {"role": "user", "content": "Great!"},
        {"role": "assistant", "content": "Thank you! Let's continue. Please provide the next class name."},
        {"role": "user", "content": "tiger"},
        {"role": "assistant", "content": "golder tiger"},
        {"role": "user", "content": "Great one"},
        {"role": "assistant", "content": "I'm glad you liked it. Please provide the next class name."},
        {"role": "user", "content": "dog"},
        {"role": "assistant", "content": "spotted dog"},
        {"role": "user", "content": "not so great (how does this supposed to change the style of the dog?)"},
        {"role": "assistant", "content": "I apologize for that. Here's another one:\nsleepy dog"},
        {"role": "user", "content": "better. u can try: fluffy dog. lets try again\ncat"},
        {"role": "assistant", "content": "striped cat"},
        {"role": "user", "content": "table"},
        {"role": "assistant", "content": "round table"},    
        {"role": "user", "content": "too much changing the shape of it. we want to change mainly the style. try again\ntable"},
        {"role": "assistant", "content": "antique table"},
        {"role": "user", "content": "Great!\ntable"},
        {"role": "assistant", "content": "polished table"},
        {"role": "user", "content": "superb.\ntable"},
        {"role": "assistant", "content": "clean table"},
        {"role": "user", "content": "table"},
        {"role": "assistant", "content": "dirty table"},
        ]

def get_gpt_response(text, bot_messeges=BOT_MESSAGES_IMAGENET): 
    bot_messeges_copy = bot_messeges.copy()
    bot_messeges_copy.append({"role": "user", "content": text})

    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages= bot_messeges_copy,
        temperature=0.5
    )
    chat_response = completion.choices[0].message.content
    bot_messeges.append(
        {'role': 'assistant', 'content': chat_response})
    
    return chat_response


if __name__ == "__main__":
    print(get_gpt_response("plane"))