from typing import List, Tuple
import numpy as np
import itertools
from Levenshtein import distance, ratio
from openai import AzureOpenAI
from config import Config
from prompt import SPAN_DETECTION_SYSTEM_PROMPT

# Constants for entity parsing
BEGIN_ENTITY_TOKEN = '['
END_ENTITY_TOKEN = ']'
SEPARATOR_TOKEN = '|'
RELATION_SEPARATOR_TOKEN = '='


# Utility functions
def get_char_index_from_token_id(token_id, tokens):
    """Returns the character index of the token in the original text"""
    char_index = 0
    for i, token in enumerate(tokens):
        if i == token_id:
            return char_index, len(token) + char_index
        char_index += len(token) + 1
    return -1

def parse_output_sentence(tokens: list, output_sentence: str) -> Tuple[list, bool]:
    """Parse LLM output sentence and extract entities with tags"""
    output_tokens = []
    unmatched_predicted_entities = []
    
    padded_output_sentence = output_sentence
    for special_token in [BEGIN_ENTITY_TOKEN, END_ENTITY_TOKEN, SEPARATOR_TOKEN, RELATION_SEPARATOR_TOKEN]:
        padded_output_sentence = padded_output_sentence.replace(special_token, ' ' + special_token + ' ')
    
    entity_stack = []
    
    for token in padded_output_sentence.split():
        if len(token) == 0:
            continue
        elif token == BEGIN_ENTITY_TOKEN:
            start = len(output_tokens)
            entity_stack.append([start, "name", [], []])
        elif token == END_ENTITY_TOKEN and len(entity_stack) > 0:
            start, state, entity_name_tokens, entity_other_tokens = entity_stack.pop()
            entity_name = ' '.join(entity_name_tokens).strip()
            end = len(output_tokens)
            tags = []
            
            splits = [list(y) for x, y in itertools.groupby(entity_other_tokens, lambda z: z == SEPARATOR_TOKEN) if not x]
            
            if state == "other" and len(splits) > 0:
                for x in splits:
                    tags.append(tuple(' '.join(x).split(' ' + RELATION_SEPARATOR_TOKEN + ' ')))
            
            unmatched_predicted_entities.append((entity_name, tags, start, end))
        else:
            if len(entity_stack) > 0:
                if token == SEPARATOR_TOKEN:
                    x = entity_stack[-1]
                    if x[1] == "name":
                        x[1] = "other"
                    else:
                        x[3].append(token)
                else:
                    is_name_token = True
                    for x in reversed(entity_stack):
                        if x[1] == "name":
                            x[2].append(token)
                        else:
                            x[3].append(token)
                            is_name_token = False
                            break
                    if is_name_token:
                        output_tokens.append(token)
            else:
                output_tokens.append(token)
    
    wrong_reconstruction = (''.join(output_tokens) != ''.join(tokens))
    
    # Dynamic programming alignment
    cost = np.zeros((len(tokens) + 1, len(output_tokens) + 1))
    best = np.zeros_like(cost, dtype=int)
    
    for i in range(len(tokens) + 1):
        for j in range(len(output_tokens) + 1):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0 and j > 0:
                cost_pair = 1-ratio(tokens[i - 1], output_tokens[j - 1])
                candidates.append(((0 if tokens[i - 1] == output_tokens[j - 1] else cost_pair) + cost[i - 1, j - 1], 1))
            if i > 0:
                candidates.append((1 + cost[i - 1, j], 2))
            if j > 0:
                candidates.append((1 + cost[i, j - 1], 3))
            chosen_cost, chosen_option = min(candidates)
            cost[i, j] = chosen_cost
            best[i, j] = chosen_option
    
    # Reconstruct alignment
    matching = {}
    i, j = len(tokens) - 1, len(output_tokens) - 1
    
    while i >= 0 and j >= 0:
        chosen_option = best[i + 1, j + 1]
        if chosen_option == 1:
            matching[j] = i
            i, j = i - 1, j - 1
        elif chosen_option == 2:
            i -= 1
        else:
            j -= 1
    
    predicted_entities = []
    for entity_name, entity_tags, start, end in unmatched_predicted_entities:
        new_start = new_end = None
        for j in range(start, end):
            if j in matching:
                if new_start is None:
                    new_start = matching[j]
                new_end = matching[j]
        if new_start is not None:
            predicted_entities.append((entity_name, entity_tags, new_start, new_end + 1))
    
    return predicted_entities, wrong_reconstruction

class SpanDetection:
    def __init__(self, config: Config):
        self.config = config
        self.client = config.get_openai_client()
        self.system_prompt = SPAN_DETECTION_SYSTEM_PROMPT
    
    
    def detect_spans_from_text(self, text: str) -> str:
        """Detect spans in text using LLM"""
        preprocessed_text = text.replace('[', '{').replace(']', '}').replace('|', ',')
        
        response = self.client.chat.completions.create(
            model=self.config.azure_openai_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": preprocessed_text}
            ]
        )
        
        return response.choices[0].message.content.strip()
    
    @staticmethod
    def loop_no_punct(text: str, index: int, reverse: bool = False) -> int:
        """Loop through text and return index of first non-punctuation character"""
        if reverse:
            for i in range(index, -1, -1):
                if text[i].isalnum():
                    return i
        else:
            for i in range(index, len(text)):
                if text[i].isalnum():
                    return i
        return -1
    
    def get_tags_spans_from_construction(self, text: str, construction: str) -> List[Tuple[str, str, int, int]]:
        """Extract tags and spans from construction string"""
        tokens = text.split()
        tags_spans = []
        predicted_entities, wrong_reconstruction = parse_output_sentence(tokens, construction)
        
        for entity in predicted_entities:
            try:
                span = entity[0]
                tag = entity[1]
                first_token = entity[2]
                last_token = entity[3]
                
                first_char_index = get_char_index_from_token_id(first_token, tokens)
                last_char_index = get_char_index_from_token_id(last_token - 1, tokens)
                
                first_char_index = self.loop_no_punct(text, first_char_index[0], reverse=False)
                last_char_index = self.loop_no_punct(text, last_char_index[1], reverse=True)
                last_char_index += 1
                
            except Exception as e:
                print(f"Error processing entity {entity}: {e}, wrong_reconstruction: {wrong_reconstruction}")
                print(f"Text: {text}, start token: {first_token}, end token: {last_token}")
                print('entity:', entity)
                print(f"first_char_index: {first_char_index}, last_char_index: {last_char_index}")
                continue
                
            tags_spans.append((span, tag, first_char_index, last_char_index))
        
        return tags_spans