import os
import base64
import requests
import json
import time

file_directory = "/Users/anisamajhi/Downloads/ConceptARC_vision"
output_directory = "./outputs"
api_key = "YOUR_OPENAI_API_KEY"

train_data_initial_prompt = "Jenny likes to change pictures in a certain way. She changes the first picture into the second picture. Can you tell me what changed between the two pictures?"
train_data_followup_prompt = "Now Jenny changes the first object/picture into the second object/picture. Can you tell me what changed between the two pictures?"
general_rule_prompt = "Now can you tell me what the general rule for how Jenny changes all the pictures, based on the comparisons you made? Be as specific as possible, but remember that the rule must apply to all the pictures."
test_data_prompt = "Now Jenny sees a new picture. How exactly is she going to change it?"

requests_per_day = 100
daily_seconds_run = 24 * 60 * 60
delay = daily_seconds_run / requests_per_day # Adjust to speed up program

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_concept_names():
    prompt_names = set()
    for filename in os.listdir(file_directory):
        prompt_name = filename.split('_')[0]
        prompt_names.add(prompt_name)
    return list(prompt_names)

def get_file_tuples_for_concept(concept, type):
    files = []
    for filename in os.listdir(file_directory):
        if filename.startswith(concept + '_' + type):
            files.append(filename)

    files.sort()
    pairs = []
    for i in range(0, len(files), 2):
        pairs.append([files[i], files[i + 1]])

    return pairs

def initial_payload(input_image, output_image):
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": train_data_initial_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{input_image}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{output_image}"
                        },
                    },
                ], 
            },
        ],
        "max_tokens": 300
    }
    return payload

def update_payload(previous_payload, response, new_prompt):
    response_message = {"role": "assistant", 
                        "content": [
                            {
                                "type": "text",
                                "text": response
                                }
                        ]
    }

    previous_payload["messages"] += [response_message]
    previous_payload["messages"] += [new_prompt]

    return previous_payload

concepts = get_concept_names()

for i in range(3):
    output_file = output_directory + f"/iteration_{i + 1}.json"
    results = []

    for concept in concepts:
        print("---------------------")
        print(f"Beginning Concept {concept}")

        concept_result = {"concept_name": concept,
                  "local_detection": [],
                  "generalization": "",
                  "extrapolation": [],
                  }

        # Step 1: Local Detection.
        print("--------")
        print("Beginning Local Detection.")

        train_data = get_file_tuples_for_concept(concept, "train")

        initial_input_image = encode_image(file_directory + "/" + train_data[0][0])
        initial_output_image = encode_image(file_directory + "/" + train_data[0][1])

        payload = initial_payload(initial_input_image, initial_output_image)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()     
        response = response['choices'][0]['message']['content']
        concept_result["local_detection"] += [response]

        print(f"Completed Local Detection on Sample 0. Sleeping for {delay} seconds.")
        time.sleep(delay)

        for i, data in enumerate(train_data[1:]):
            input_image = encode_image(file_directory + "/" + data[0])
            output_image = encode_image(file_directory + "/" + data[1])
            new_prompt = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": train_data_followup_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{input_image}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{output_image}"
                            }
                        }
                    ]
                }
            
            payload = update_payload(payload, response, new_prompt)
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response = response.json()     
            response = response['choices'][0]['message']['content']
            concept_result["local_detection"] += [response]

            print(f"Completed Local Detection on Sample {i + 1}. Sleeping for {delay} seconds.")
            time.sleep(delay)
        
        # Step 2: Generalization. 
        print("--------")
        print("Beginning Generalization.")

        general_rule_prompt_message = {"role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": general_rule_prompt
                        }
                    ]
                }

        payload = update_payload(payload, response, general_rule_prompt_message)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()
        response = response['choices'][0]['message']['content']
        concept_result["generalization"] = response

        print(f"Completed Generalization. Sleeping for {delay} seconds.")
        time.sleep(delay)

        # Step 3: Extrapolation.
        print("--------")
        print("Beginning Extrapolation.")

        test_data = get_file_tuples_for_concept(concept, "test")

        for i, data in enumerate(test_data):
            input_image = encode_image(file_directory + "/" + data[0])
            new_prompt = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": test_data_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{input_image}"
                            }
                        }
                    ]
                }
            payload = update_payload(payload, response, new_prompt)
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response = response.json()
            response = response['choices'][0]['message']['content']
            concept_result["extrapolation"] += [response]

            print(f"Completed Extrapolation on Sample {i}. Sleeping for {delay} seconds.")
            time.sleep(delay)
            

        results.append(concept_result)
    
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)
