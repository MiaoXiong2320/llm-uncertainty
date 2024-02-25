import re, pdb
from typing import Tuple, Optional, Dict

word_to_digit = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

def word_to_int(word):
    try:
        return float(word)
    except ValueError:
        return word_to_digit.get(word.lower(), None)
    
def extract1(data):
    step_pattern = r"Step\d+:(.*?)(?=\nConfidence: \d+|\nFinal Answer and Confidence\(1-100\)|$)"
    step_matches = re.findall(step_pattern, data, re.DOTALL)

    # Extract confidence information
    confidence_pattern = r"Confidence: (\d+)"
    confidence_matches = re.findall(confidence_pattern, data)

    # Extract final answer and final confidence
    final_answer_pattern = r"Final Answer and Confidence\(0-100\): (\w+), (\d+)"
    final_answer_match = re.search(final_answer_pattern, data)
    final_answer = final_answer_match.group(1) if final_answer_match else None
    final_confidence = int(final_answer_match.group(2)) if final_answer_match else None

    # Create the result dictionary
    result = {}
    for i, step_match in enumerate(step_matches):
        step_key = f"step{i + 1}"
        result[step_key] = {
            "analysis": step_match.strip(),
            "confidence": int(confidence_matches[i]) if i < len(confidence_matches) else None
        }
    result["final_answer"] = final_answer
    result["final_confidence"] = final_confidence

    return result


def check_validity_answer_conf(extracted_answer, extracted_conf, task_type, error_log_file):
    
    # check the validity of the extracted confidence
    try:
        extracted_conf = float(extracted_conf)
        if extracted_conf > 100 or extracted_conf < 0:
            raise ValueError
    except:
        with open(error_log_file, 'a') as f:
            f.write(f"Confidence Invalid: {extracted_conf}\n")
        return None, None
    
    if extracted_answer is None:
        return None, extracted_conf
    
    # check whether the generated_answer_by_hint in A,B,C,D,E
    if task_type == "multi_choice_qa":
        # If a match is found, check whether the matched result meets the requirements
        if (extracted_answer not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') and (extracted_answer != "place_holder"):
            # If the matched result meets the requirements, print the result and exit the loop
            with open(error_log_file, 'a') as f:
                f.write(f"Answer Invalid: {extracted_answer}")
            extracted_answer = None

    elif task_type == "open_number_qa":
        try:
            extracted_answer = word_to_int(extracted_answer.lower())
        except:
            
            with open(error_log_file, 'a') as f:
                f.write(f"Number Answer Invalid: {extracted_answer}")
            extracted_answer = None
    else:
        raise ValueError(f"The task type {task_type} is not in the list: ['multi_choice_qa', 'open_number_qa']")
    
    
    return extracted_answer, extracted_conf

# we first ensure the logic is correct, then we refactor the code
# a legacy function. for multistep
def search_final_answer_for_multistep(text, options: Dict[str, str], task_type, error_log_file):
    
    def default_postprocess_match(match) -> Tuple[str, str]:
        assert match is not None
        answer = match.group(1) 
        conf = match.group(2 if len(match.groups()) == 2 else 3)
        # print(f"default_postprocess_match {answer} {conf}")
        return answer, conf
    
    def postprocess_match_without_option(match) -> Tuple[str, str]:
        assert match is not None
        answer_option_value = match.group(1).strip()
        
        conf = match.group(2)
        answer_option_key = None
        for option_key, option_value in options.items():
            option_value = option_value.strip().strip(".").lower()
            answer_option_value = answer_option_value.strip().strip(".").lower()
            if answer_option_value in option_value or \
                option_value in answer_option_value:
                answer_option_key = option_key
                
        if answer_option_key is None:
            # try to extract "None of" pattern
            if "none of" in answer_option_value.strip().lower():
                answer_option_key = "Z"
        
        # assert answer_option_key is not None, (match.group(0), answer_option_value, options)
        if answer_option_key is None:
            # print(match.group(0), answer_option_value, options)
            return None, None
        
        # print(f"default_postprocess_match_without_option {answer_option_key} {conf}")
        return answer_option_key, conf
    
    # Define five different regular expression patterns
    # print(text)
    patterns_multi_choice = [
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(0-100\):\s*([A-G])\.?\s?\w*,\s*(\d{1,3})%',
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(0-100\):\s*\(([A-G])\)\.?\s?\w*,\s*(\d{1,3})%',
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(0-100\):\s*([A-G])\.?\s?\w*,?\s*(\d{1,3})%',
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(0-100\):\s*([A-G])\.?\s?\w*,?\s*(\d{1,3})%',
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(0-100\):\s*([A-G])\.?\s*(?:\w+,)?\s*(\d{1,3})%',  
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d{1,3}-?\d{0,2}\): ([A-G](\.\s\w+)?), (\d{1,3})%',
        r"(?:Final )?Answer and (?:Overall )?Confidence\s*\(0-100\):\s*\(([A-G])\) .*? (\d+)%",
        r"(?:Final )?Answer and (?:Overall )?Confidence\s*\(0-100\):\s*Option \(?([A-G])\)?\s*.*?,\s*(\d{1,3})%",
        # r"(?:Final )?Answer and Overall Confidence\s*\(0-100\):\s*(?:(?:O|o)ption)?\s*\(?([A-G])\)?(?:,|\s+).*,\s*(\d+)%",
        r"(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*(?:(?:O|o)ption)?\s*\(?([A-G])\)?(?:,|\s+).*,\s*(\d+(?:\.\d+)?)%",
        r"(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*(?:(?:O|o)ption)?\s*\(?([A-G])\)?(?:[\s,]+)(\d+(?:\.\d+)?)%",
        
        r"(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*(?:(?:O|o)ption)?\s*\(?([A-G])\)?(?:,|\s+).*,\s*(\d+(?:\.\d+)?)(?:$|.|%)",
        r"(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*(?:(?:O|o)ption)?\s*\(?([A-G])\)?(?:[\s,]+)(\d+(?:\.\d+)?)(?:$|.|%)",
        
        r"(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*(?:(?:O|o)ption)?\s*\(?([A-G])\)?(?:,|\s+).*,\s*[cC]onfidence(?:\s|:)+(\d+(?:\.\d+)?)(?:$|.|%)",
        r"(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*(?:(?:O|o)ption)?\s*\(?([A-G])\)?(?:[\s,]+)[cC]onfidence(?:\s|:)+(\d+(?:\.\d+)?)(?:$|.|%)",
        r"(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):.*?(?:\s|,|\.|\()([A-G])(?:\s|,|:|\.|\)).*?(\d+)%",
        
    ]
    # sometimes the LLM will directly output the answer rather than the associated option key
    patterns_multi_choice_without_option = [
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*(.*)\s*,\s*(\d+)%',
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*(.*)\s*,\s*(\d+)(?:%|.|$)',
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*(.*)\s*,\s*[cC]onfidence(?:\s|:)+(\d+)(?:$|.|%)',
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*(.*)(?:\s|,)+\s*[cC]onfidence(?:\s|:)+(\d+)(?:$|.|%)',
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*(.*)(?:\s|,)+\s*[cC]onfidence(?:\s|:)+(\d+)(?:$|.|%)',
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*(.*)?(?:\s|,)+(.*)?(\d+)(?:%)',
        r'(?:Final )?Answer and (?:Overall )?Confidence\s*\(\d+-100\):\s*([a-zA-Z,]+)(\d+)(?:%)',
        
    ]
    patterns_and_postprocess_multi_choice = []
    patterns_and_postprocess_multi_choice.extend([(pat, default_postprocess_match) for pat in patterns_multi_choice])
    patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_without_option])
      
    # Define five different regular expression patterns
    patterns_open_number = [
        r'Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?:\s*(d+)\.?\s?\w*,\s*(\d{1,3})(?:%|$|.$)',
        r"Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?:.*?(\d+).*?(\d+)(?:%|$|.$)",
        r"Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?:.*(\d+).*, (\d+)(?:%|$|.$)",
        r"Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?:\s*(\d+) .*? (\d{1,3})(?:%|$|.$)",
        r"Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?: (\d+) .*? (\d{1,3})(?:%|$|.$)",
        r"Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?: (\d+), (\d{1,3})(?:%|$|.$)",
        r'Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?:\s*\((d+)\)\.?\s?\w*,\s*(\d{1,3})(?:%|$|.$)',
        r'Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?:\s*(d+)\.?\s?\w*,?\s*(\d{1,3})(?:%|$|.$)',
        r'Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?:\s*(d+)\.?\s?\w*,?\s*(\d{1,3})(?:%|$|.$)',
        r'Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?:\s*(d+)\.?\s*(?:\w+,)?\s*(\d{1,3})(?:%|$|.$)',  
        r'Final Answer and (?:Overall )?Confidence\s*\(\d{1,3}-?\d{0,2}\): (d+(\.\s\w+)?), (\d{1,3})(?:%|$|.$)',
        
        r"Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?:\s*\((d+)\) .*? (\d+)(?:%|$|.$)",
        # match the verbalized number 
        # 
        r"Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?: .*? ([Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine|[Tt]en|[Ee]leven|[Tt]welve|(?:[Tt]hir|fif|eigh|nine)teen|(?:[Tt]wen|[Tt]hir|[Ff]or|[Ff]if|[Ss]ix|[Ss]even|[Ee]igh|[Nn]ine)ty|(?:[Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine) hundred|(?:[Oo]ne) thousand|(?:[Oo]ne) million|[Oo]ther) (?:\w+, )*?(\d{1,3})(?:%|$|.$)",
        # Final Answer and (?:Overall )?Confidence ([01]-100): Nine musical instruments, 100(?:%|$|.$)'
        r"Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?:.*? ([Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine|[Tt]en|[Ee]leven|[Tt]welve|(?:[Tt]hir|fif|eigh|nine)teen|(?:[Tt]wen|[Tt]hir|[Ff]or|[Ff]if|[Ss]ix|[Ss]even|[Ee]igh|[Nn]ine)ty|(?:[Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine) hundred|(?:[Oo]ne) thousand|(?:[Oo]ne) million|[Oo]ther).*, (\d{1,3})%",
        
        r"Final Answer and (?:Overall )?Confidence\s*(?:\([01]-100\))?:(?:\s|\$)*(\d+(?:\.\d+)?)(?:\s|,)+(\d+)(?:%|$|.$)",
           
    ]
    
    patterns_and_postprocess_open_number = [(pat, default_postprocess_match) for pat in patterns_open_number]
    # pre-process
    text = text.replace("(1-100)", "(0-100)")
    

    # Try each regular expression pattern in turn, until a match is found or all patterns have been tried
    is_match = False
    
    if task_type == "multi_choice_qa":
        patterns_and_postprocess = patterns_and_postprocess_multi_choice
    elif task_type == "open_number_qa":
        patterns_and_postprocess = patterns_and_postprocess_open_number
    else:
        raise ValueError(f"task_type {task_type} is not supported")
    
    # begin to parse the final answer and confidence: e.g. Final Answer and Overall Confidence (0-100): (A) 05/01/2021, 100%
    answer, conf = None, None
    for pattern, match_processor in patterns_and_postprocess:
        match = re.search(pattern, text)
        if not match:
            continue
        answer, conf = match_processor(match)
        answer, conf = check_validity_answer_conf(answer, conf, task_type, error_log_file)
        if answer is not None and conf is not None:
            is_match = True
            break     
                
    if not is_match:
        # If no match is found, print a message
        print(f"No `final answer and confidence` match found for {text}")  
        with open(error_log_file, 'a') as f:
            f.write(f"No `final answer and confidence` match found. extracted_answer: {answer}, extracted_conf: {conf}\n")
        answer = None
        conf = None
    
    return answer, conf
    
def search_step_confidence(text, error_log_file):
    

    # Define five different regular expression patterns
    patterns = [
        r'[cC]onfidence\s*:\s*(\d{1,3})%',
        r'[cC]onfidence\s*\(0-100\):\s*(\d{1,3})%',
        r'[cC]onfidence\s*\(0-100\):\s*(\d{1,3})%?',
        r"[cC]onfidence:\s+(\d+)",
    ]
    
    # pre-process
    text = text.replace("(1-100)", "(0-100)")
    

    # Try each regular expression pattern in turn, until a match is found or all patterns have been tried
    is_match = False
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # If a match is found, check whether the matched result meets the requirements
            if 0 <= int(match.group(1)) <= 100:
                # If the matched result meets the requirements, print the result and exit the loop
                is_match = True
                break
    
    if is_match:
        conf = int(match.group(1))
    else:
        # If no match is found, print a message
        print(f"No `step confidence` match found for {text}")  
        with open(error_log_file, 'a') as f:
            # f.write(f"No `step confidence` match found for {text}\n")
            # conf = None

            # TODO: currently skip this step and set confidence to 100 (not sure if this is a good idea)
            f.write(f"No `step confidence` match found for {text}, set confidence to 100\n")
            f.write("-"*60 + "\n")
            conf = 100
        
    
    return conf

def extract_hint_response_multistep(data, options: Dict[str, str], task_type, error_log_file):
    
    # Extract step information
    try:
        step_pattern = r"Step\s*(\d)+:(.*?)(?=\s*\nStep\s+\d+|\s*Final Answer)"
        step_matches = re.findall(step_pattern, data, re.DOTALL)
    except Exception as e:
        print(e)
        # If no match is found, print a message
        print(f"No `step` match found for {data.strip()}")  
        with open(error_log_file, 'a') as f:
            f.write(e + f"No match found for {data}\n")
        return None, None, None
        

    # Extract final answer and final confidence
    final_answer, final_confidence = search_final_answer_for_multistep(data, options, task_type, error_log_file)

    if len(step_matches) == 0:
        with open(error_log_file, 'a') as f:
            f.write(f"No `step matches`\n")
    # Create the result dictionary
    step_result = {}
    for idx, (step_no, step_sentence) in enumerate(step_matches):
        if int(step_no) != idx + 1:
            with open(error_log_file, 'a') as f:
                f.write(f"STEP NUMBER MISMATCH: step_no {step_no} != idx {idx} in the sentence ({step_sentence})\n")
            return final_answer, final_confidence, step_result
        
        
        step_key = f"step{idx}"
        step_conf = search_step_confidence(step_sentence, error_log_file)
      
        step_result[step_key] = {
            "analysis": step_sentence.strip(),
            "confidence": step_conf
        }

    return final_answer, final_confidence, step_result

def extract_hint_response_vanilla(text, options: Dict[str, str], task_type, error_log_file):
    def default_postprocess_match(match) -> Tuple[str, str]:
        assert match is not None
        option_key, conf_scale = match.group(1), match.group(2)
        return option_key, conf_scale
    
    def postprocess_match_without_option(match) -> Tuple[str, str]:
        assert match is not None
        answer_option_value = match.group(1).strip()
        
        conf_scale = match.group(2)
        answer_option_key = None
        for option_key, option_value in options.items():
            option_value = option_value.strip().strip(".").lower()
            answer_option_value = answer_option_value.strip().strip(".").lower()
            if answer_option_value in option_value or \
                option_value in answer_option_value:
                answer_option_key = option_key
        
        # assert answer_option_key is not None, (match.group(0), answer_option_value, options)
        if answer_option_key is None:
            print(match.group(0), answer_option_value, options)
            # it returns an answer that does not belong to any of the option values
            return "Z", conf_scale
        
        return answer_option_key, conf_scale
    
    def postprocess_match_open_number(match) -> Tuple[str, str]:
        assert match is not None
        numerical_answer, conf_scale = match.group(1), match.group(2)
        numerical_answer = numerical_answer.replace(",", "") # 1,000 -> 1000
        
        return numerical_answer, conf_scale
        
        
    # Define five different regular expression patterns
    patterns_multi_choice = [
        r"Answer and Confidence\s*(?:\(0-100\))?:\s*[\(\[]?([A-Z])[\)\]]?,\s*(\d+)%*",
        r"Answer and Confidence\s*(?:\(0-100\))?:\s*[\(\[]?([A-Z])[\)\]]?[,]?\s*(\d+)%?",
        r"Answer and Confidence\s*(?:\(0-100\))?:\s*[\(\[]?([A-Z])[\)\]]?[,]?\s*[\(\[]?(\d+)%?[\(\[]?",
        r"Answer and Confidence\s*(?:\(0-100\))?:\s*[\(\[]?([A-Z])[\)\]]?[,.]?\s*.*[\(\[]?(\d+)%?[\(\[]?",
    ]
    # sometimes the LLM will directly output the answer rather than the associated option key
    patterns_multi_choice_without_option = [
        r"Answer and Confidence\s*(?:\(0-100\))?:\s*(.*?)\s*,\s*(\d+)%*"
    ]
    
    # [\(\[]?([A-Z])[\)\]]?  -> [\(\[]? matches optional ([
    # [\)\]]? matches optional )]
    # most appears in vicuna
    # Note: .* can match any character (except for a newline character) zero or more times
    patterns_multi_choice_werid = [
        r"Answer: [\(\[]?([A-Z])[\)\]]?[,.]?\s+Confidence level: (\d+%)",
        r"Answer: [\(\[]?([A-Z])[\)\]]?[,.]?.*\s+Confidence(?: level)?: (\d+%)",
        r"Answer:\s*[\(\[]?([A-Z])[\)\]]?[,.]?\s+Confidence level:\s*(\d+%)"
    ]
    
    patterns_and_postprocess_multi_choice = []
    patterns_and_postprocess_multi_choice.extend([(pat, default_postprocess_match) for pat in patterns_multi_choice])
    patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_without_option])
    patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_werid])
      
    # Define five different regular expression patterns
    patterns_open_number = [
        r"Answer and Confidence\s*(?:\(0-100\))?:\s*.*?([0-9,.]+).*?,\s*(\d+)%*",
        r"Answer and Confidence\s*(?:\(0-100\))?:\s*.*?([0-9,.]+).*?;\s*(\d+)%*"
        
    ]
    
    
    patterns_and_postprocess_open_number = [(pat, postprocess_match_open_number) for pat in patterns_open_number]
    # pre-process
    text = text.replace("(1-100)", "(0-100)")
    
    # Try each regular expression pattern in turn, until a match is found or all patterns have been tried
    is_match = False
    
    if task_type == "multi_choice_qa":
        patterns_and_postprocess = patterns_and_postprocess_multi_choice
    elif task_type == "open_number_qa":
        patterns_and_postprocess = patterns_and_postprocess_open_number
    else:
        raise ValueError(f"task_type {task_type} is not supported")
    
    answer, conf = None, None
    for pattern, match_processor in patterns_and_postprocess:
        match = re.search(pattern, text)
        if not match:
            continue
        answer, conf = match_processor(match)
        answer, conf = check_validity_answer_conf(answer, conf, task_type, error_log_file)
        if answer is not None and conf is not None:
            is_match = True
            break     
                
    if not is_match:
        # If no match is found, print a message
        print(f"No `final answer and confidence` match found for {text}")  
        with open(error_log_file, 'a') as f:
            f.write(f"No `final answer and confidence` match found. extracted_answer: {answer}, extracted_conf: {conf}\n")
        answer = None
        conf = None
    
    return answer, conf


def extract_hint_response_top_k(text, K, options: Dict[str, str], task_type, error_log_file):
    def default_postprocess_match(match) -> Tuple[str, str]:
        assert match is not None
        option_key, conf_scale = match.group(1), match.group(2)
        return option_key, conf_scale
    
    def postprocess_match_without_option(match) -> Tuple[str, str]:
        assert match is not None
        answer_option_value = match.group(1).strip()
        
        conf_scale = match.group(2)
        answer_option_key = None
        for option_key, option_value in options.items():
            option_value = option_value.strip().strip(".").lower()
            answer_option_value = answer_option_value.strip().strip(".").lower()
            if answer_option_value in option_value or \
                option_value in answer_option_value:
                answer_option_key = option_key
        
        # assert answer_option_key is not None, (match.group(0), answer_option_value, options)
        if answer_option_key is None:
            print(match.group(0), answer_option_value, options)
            # it returns an answer that does not belong to any of the option values
            return "Z", conf_scale
        
        return answer_option_key, conf_scale
    
    def postprocess_match_open_number(match) -> Tuple[str, str]:
        assert match is not None
        numerical_answer, conf_scale = match.group(1), match.group(2)
        numerical_answer = numerical_answer.replace(",", "") # 1,000 -> 1000
        
        return numerical_answer, conf_scale
        
    
    def process_pipeline(ith):
        # Define five different regular expression patterns
        patterns_multi_choice = [
            rf"(?:G{ith}|Guess {ith}):\s*[\(\[]?([A-Z])[\)\]]?\s*(?:P{ith}|Probability {ith}):\s*(\d+)%*",
            rf"(?:G{ith}|Guess {ith}):\s*[\(\[]?([A-Z])[\)\]]?\s*(?:P{ith}|Probability {ith}):\s*(\d+)%?",
            rf"(?:G{ith}|Guess {ith}):\s*[\(\[]?([A-Z])[\)\]]?\s*(?:P{ith}|Probability {ith}):\s*[\(\[]?(\d+)%?[\(\[]?",
            rf"(?:G{ith}|Guess {ith}):\s*[\(\[]?([A-Z])[\)\]]?\s*.*\s*(?:P{ith}|Probability {ith}):\s*[\(\[]?(\d+)%?[\(\[]?",
        ]
        # sometimes the LLM will directly output the answer rather than the associated option key
        patterns_multi_choice_without_option = [
            rf"(?:G{ith}|Guess {ith}):\s*(.*?)\s*(?:P{ith}|Probability {ith}):\s*(\d+)%*"
        ]
        
        # [\(\[]?([A-Z])[\)\]]?  -> [\(\[]? matches optional ([
        # [\)\]]? matches optional )]
        # most appears in vicuna
        # Note: .* can match any character (except for a newline character) zero or more times
        patterns_multi_choice_werid = [
            r"Answer: [\(\[]?([A-Z])[\)\]]?[,.]?\s+Confidence level: (\d+%)",
            r"Answer: [\(\[]?([A-Z])[\)\]]?[,.]?.*\s+Confidence(?: level)?: (\d+%)",
            r"Answer:\s*[\(\[]?([A-Z])[\)\]]?[,.]?\s+Confidence level:\s*(\d+%)"
        ]
        
        patterns_and_postprocess_multi_choice = []
        patterns_and_postprocess_multi_choice.extend([(pat, default_postprocess_match) for pat in patterns_multi_choice])
        patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_without_option])
        # patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_werid])
        
        # Define five different regular expression patterns
        patterns_open_number = [
            rf"G{ith}:\s*.*?([0-9,.]+)\s*P{ith}:\s*(\d+)%*",
            rf"G{ith}:\s*.*?([0-9,.]+)\s*.*\s+P{ith}:\s*(\d+)%*",
            
        ]
        
        
        patterns_and_postprocess_open_number = [(pat, postprocess_match_open_number) for pat in patterns_open_number]
        
        
        if task_type == "multi_choice_qa":
            patterns_and_postprocess = patterns_and_postprocess_multi_choice
        elif task_type == "open_number_qa":
            patterns_and_postprocess = patterns_and_postprocess_open_number
        else:
            raise ValueError(f"task_type {task_type} is not supported")
        
        return patterns_and_postprocess
       
    # pre-process
    text = text.replace("(1-100)", "(0-100)")
    text = text.replace("\n", " ")
         
    answers, confs = {}, {}
    match_error = []
    for ith in range(0, K):
        # Try each regular expression pattern in turn, until a match is found or all patterns have been tried
        is_match = False
        patterns_and_postprocess = process_pipeline(ith+1)
        for pattern, match_processor in patterns_and_postprocess:
            match = re.search(pattern, text)
            if not match:
                continue
            else:
                answer, conf = match_processor(match)
                answer, conf = check_validity_answer_conf(answer, conf, task_type, error_log_file)
            if answer is not None and conf is not None:
                is_match = True
                break   
            
                    
        if not is_match:
            match_error.append(ith)
            answer = None
            conf = None
            

        answers[ith] = answer
        confs[ith] = conf

    if answers[0] is None or confs[0] is None:
        # If no match is found, print a message
        print(f"\n\nTop-1 ERROR: {match_error}.\nReponse: {text}")  
        with open(error_log_file, 'a') as f:
            f.write(f"\n\nTop-1 ERROR for: {match_error}.\nReponse: {text}\n\nExtracted_answer: {answers}\n extracted_conf: {confs}\n") 
        answers = None
        confs = None
        
        return answers, confs

    if len(match_error) > 0:
        # If no match is found, print a message
        print(f"\n\nMatch error for: {match_error}.\nReponse: {text}")  
        with open(error_log_file, 'a') as f:
            f.write(f"\n\nMatch error for: {match_error}.\nReponse: {text}\n\nExtracted_answer: {answers}\n extracted_conf: {confs}\n")  

    
    return answers, confs

def extract_hint_response_self_evaluate(text, options: Dict[str, str], task_type, error_log_file):
    def default_postprocess_match(match) -> Tuple[str, str]:
        assert match is not None
        conf_scale = match.group(1)
        return conf_scale
    
    def postprocess_match_without_option(match) -> Tuple[str, str]:
        assert match is not None

        conf_scale = match.group(1)
        # for option_key, option_value in options.items():
        #     option_value = option_value.strip().strip(".").lower()
        #     answer_option_value = answer_option_value.strip().strip(".").lower()
        #     if answer_option_value in option_value or \
        #         option_value in answer_option_value:
        #         answer_option_key = option_key
        
        # assert answer_option_key is not None, (match.group(0), answer_option_value, options)
        # if answer_option_key is None:
        #     print(match.group(0), answer_option_value, options)
        #     # it returns an answer that does not belong to any of the option values
        #     return "Z", conf_scale
        
        return conf_scale
    
    def postprocess_match_open_number(match) -> Tuple[str, str]:
        assert match is not None
        conf_scale = match.group(1)
        
        return conf_scale
        
        
    # Define five different regular expression patterns
    patterns_multi_choice = [
        r"Confidence:\s*(\d+)%*",
        r"Confidence:\s*(\d+)%?",
        r"Confidence:\s*[\(\[]?(\d+)%?[\(\[]?",
        r"Confidence:\s*.*[\(\[]?(\d+)%?[\(\[]?",
        r"Confidence:\s*.*[\(\[]?(\d+)[\(\[]?%?",
    ]
    # sometimes the LLM will directly output the answer rather than the associated option key
    patterns_multi_choice_without_option = [
        r"Confidence:\s*(.*?)\s*,\s*(\d+)%*"
    ]
    
    # [\(\[]?([A-Z])[\)\]]?  -> [\(\[]? matches optional ([
    # [\)\]]? matches optional )]
    # most appears in vicuna
    # Note: .* can match any character (except for a newline character) zero or more times
    patterns_multi_choice_werid = [
        r"Confidence level: (\d+%)",
        r"Confidence(?: level)?: (\d+%)",
        r"Confidence level:\s*(\d+%)"
    ]
    
    patterns_and_postprocess_multi_choice = []
    patterns_and_postprocess_multi_choice.extend([(pat, default_postprocess_match) for pat in patterns_multi_choice])
    patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_without_option])
    patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_werid])
      
    # Define five different regular expression patterns
    patterns_open_number = [
        r"Confidence\s*:\s*(\d+)%*",
        r"Confidence:\s*(\d+)%*",
        r"Confidence:\s*(\d+)%?",
        r"Confidence:\s*[\(\[]?(\d+)%?[\(\[]?",
        r"Confidence:\s*.*[\(\[]?(\d+)%?[\(\[]?",
        r"Confidence:\s*.*[\(\[]?(\d+)[\(\[]?%?",
    ]
    
    
    patterns_and_postprocess_open_number = [(pat, postprocess_match_open_number) for pat in patterns_open_number]
    # pre-process
    text = text.replace("(1-100)", "(0-100)")
    
    # Try each regular expression pattern in turn, until a match is found or all patterns have been tried
    is_match = False
    
    if task_type == "multi_choice_qa":
        patterns_and_postprocess = patterns_and_postprocess_multi_choice
    elif task_type == "open_number_qa":
        patterns_and_postprocess = patterns_and_postprocess_open_number
    else:
        raise ValueError(f"task_type {task_type} is not supported")
    
    answer, conf = None, None
    for pattern, match_processor in patterns_and_postprocess:
        match = re.search(pattern, text)
        if not match:
            continue
        conf = match_processor(match)
        answer = "place_holder"
        _, conf = check_validity_answer_conf(answer, conf, task_type, error_log_file)
        if answer is not None and conf is not None:
            is_match = True
            break     
                
    if not is_match:
        # If no match is found, print a message
        print(f"No `final answer and confidence` match found for {text}")  
        with open(error_log_file, 'a') as f:
            f.write(f"No `final answer and confidence` match found. extracted_answer: {answer}, extracted_conf: {conf}\n")
        answer = None
        conf = None
    
    return answer, conf

# TODO: refactor these functions
def extract_hint_response_confidence_scale(text, options: Dict[str, str], task_type, error_log_file):
    def parse_confidence_scale(conf_scale: str) -> str:
        # (A) Very certain/almost certain (100%)
        # (B) Probable (75-99%)
        # (C) Somewhat confident (50-74%)
        # (D) Unsure/fifty-fifty (50%)
        # (E) Very uncertain (25-49%)
        # (F) Improbable (1-24%)
        # (G) Almost impossible (0%)
        # return {
        #     'A': '100',
        #     'B': f"{(75 + 99) // 2}",
        #     'C': f"{(50 + 74) // 2}",
        #     'D': '50',
        #     'E': f"{(25 + 49) // 2}",
        #     'F': f"{(1 + 24) // 2}",
        #     'G': '0'
        # }.get(conf_scale, None)
        conf_scale = conf_scale.lower().strip()
        # return {
        #     "very certain/almost certain": "100",
        #     "probable": f"{(75 + 99) // 2}",
        #     "somewhat confident": f"{(50 + 74) // 2}",
        #     "unsure/fifty-fifty": "50",
        #     "very uncertain": f"{(25 + 49) // 2}",
        #     "improbable": f"{(1 + 24) // 2}",
        #     "almost impossible": "0"
        # }.get(conf_scale, None)
        
        # - Very certain
        # - Almost certain
        # - Probable
        # - Somewhat confident
        # - Unsure
        # - Very uncertain
        
        return {
            "very certain": "100",
            "almost certain": "90",
            "probable": "80",
            "somewhat confident": "70",
            "unsure": "50",
            "very uncertain": "20",
        }.get(conf_scale, None)
    
    def default_postprocess_match(match) -> Tuple[str, str]:
        assert match is not None
        option_key, conf_scale = match.group(1), match.group(2)
        return option_key, parse_confidence_scale(conf_scale)
    
    def postprocess_match_without_option(match) -> Tuple[str, str]:
        assert match is not None
        answer_option_value = match.group(1).strip()
        
        conf_scale = match.group(2)
        conf_scale = parse_confidence_scale(conf_scale)
        answer_option_key = None
        for option_key, option_value in options.items():
            option_value = option_value.strip().strip(".").lower()
            answer_option_value = answer_option_value.strip().strip(".").lower()
            if answer_option_value in option_value or \
                option_value in answer_option_value:
                answer_option_key = option_key
        
        # assert answer_option_key is not None, (match.group(0), answer_option_value, options)
        if answer_option_key is None:
            print(match.group(0), answer_option_value, options)
            # return an invalid answer
            return "Z", conf_scale
        
        return answer_option_key, conf_scale
    
    def postprocess_match_open_number(match) -> Tuple[str, str]:
        assert match is not None
        numerical_answer, conf_scale = match.group(1), match.group(2)
        numerical_answer = numerical_answer.replace(",", "") # 1,000 -> 1000
        conf_scale = parse_confidence_scale(conf_scale)
        
        return numerical_answer, conf_scale
        
        
    # Define five different regular expression patterns
    patterns_multi_choice = [
        r"Answer and Confidence:\s*[\(\[]?([A-Z])[\)\]]?,\s*[\(\[]?([A-Z])[\)\]]?",
        r"Answer and Confidence:\s*[\(\[]?([A-Z])[\)\]]?,\s*\*(.*?)\s*(?:\(\d+\-\d+%\))?\*",
        
    ]
    # sometimes the LLM will directly output the answer rather than the associated option key
    patterns_multi_choice_without_option = [
        r"Answer and Confidence:\s*(.*?)\s*,\s*[\(\[]?([A-Z])[\)\]]?",
        r"Answer and Confidence:\s*(.*?)\s*,\s*\*(.*?)\s*(?:\(\d+\-\d+%\))?\*"
        
    ]
    patterns_and_postprocess_multi_choice = []
    patterns_and_postprocess_multi_choice.extend([(pat, default_postprocess_match) for pat in patterns_multi_choice])
    patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_without_option])
      
    # Define five different regular expression patterns
    patterns_open_number = [
        r"Answer and Confidence:\s*.*?([0-9,.]+).*?,\s*[\(\[]?([A-Z])[\)\]]?"
        
    ]
    
    patterns_and_postprocess_open_number = [(pat, postprocess_match_open_number) for pat in patterns_open_number]
    # pre-process
    text = text.replace("(1-100)", "(0-100)")
    
    # Try each regular expression pattern in turn, until a match is found or all patterns have been tried
    is_match = False
    
    if task_type == "multi_choice_qa":
        patterns_and_postprocess = patterns_and_postprocess_multi_choice
    elif task_type == "open_number_qa":
        patterns_and_postprocess = patterns_and_postprocess_open_number
    else:
        raise ValueError(f"task_type {task_type} is not supported")
    
    answer, conf = None, None
    for pattern, match_processor in patterns_and_postprocess:
        match = re.search(pattern, text)
        if not match:
            continue
        answer, conf = match_processor(match)
        answer, conf = check_validity_answer_conf(answer, conf, task_type, error_log_file)
        if answer is not None and conf is not None:
            is_match = True
            break     
                
    if not is_match:
        # If no match is found, print a message
        print(f"No `final answer and confidence` match found for {text}")  
        with open(error_log_file, 'a') as f:
            f.write(f"No `final answer and confidence` match found. extracted_answer: {answer}, extracted_conf: {conf}\n")
        answer = None
        conf = None
    
    return answer, conf

def extract3(data):
    # Extract step information
    step_pattern = r"Step\s*(\d+):([^:]+)(?=\nStep|$)"
    step_matches = re.findall(step_pattern, data, re.DOTALL)

    # Extract confidence information
    confidence_pattern = r"Confidence: (\d+)"
    confidence_matches = re.findall(confidence_pattern, data)

    # Extract final answer and final confidence
    final_answer_pattern = r"Final Answer and Confidence\(0-100\): (\w+), (\d+)"
    final_answer_match = re.search(final_answer_pattern, data)
    final_answer = final_answer_match.group(1) if final_answer_match else None
    final_confidence = int(final_answer_match.group(2)) if final_answer_match else None

    # Create the result dictionary
    result = {}
    for step_match in step_matches:
        step_key = f"step{step_match[0]}"
        step_analysis = step_match[1].strip()
        step_confidence = None
        for confidence_match in confidence_matches:
            if step_analysis.find(confidence_match) != -1:
                step_confidence = int(confidence_match)
                break
        result[step_key] = {
            "analysis": step_analysis,
            "confidence": step_confidence
        }
    result["final_answer"] = final_answer
    result["final_confidence"] = final_confidence
    return result

def extract4(data):
    # Extract step information
    step_pattern = r"Step\s+(\d+):([^:]+)"
    step_matches = re.findall(step_pattern, data, re.DOTALL)

    # Extract confidence information for each step
    confidence_pattern = r"Confidence:\s+(\d+)"
    confidence_matches = re.findall(confidence_pattern, data)

    # Extract final answer and final confidence
    final_answer_pattern = r"Final Answer and Confidence\(0-100\):\s+(\w+),\s+(\d+)"
    final_answer_match = re.search(final_answer_pattern, data)
    final_answer = final_answer_match.group(1) if final_answer_match else None
    final_confidence = int(final_answer_match.group(2)) if final_answer_match else None

    # Create the result dictionary
    result = {}
    for i, step_match in enumerate(step_matches):
        step_key = f"step{i+1}"
        step_analysis = step_match[1].strip()
        step_confidence = int(confidence_matches[i]) if i < len(confidence_matches) else None
        result[step_key] = {
            "analysis": step_analysis,
            "confidence": step_confidence
        }
    result["final_answer"] = final_answer
    result["final_confidence"] = final_confidence
    return result


def extract5(data):
    # Extract step information
    step_pattern = r"Step(\d+):([^:]+)"
    step_matches = re.findall(step_pattern, data, re.DOTALL)

    # Extract confidence information for each step
    confidence_pattern = r"Confidence:\s+(\d+)"
    confidence_matches = re.findall(confidence_pattern, data)

    # Extract final answer and final confidence
    final_answer_pattern = r"Final Answer and Confidence\(1-100\):\s+(\w+),\s+(\d+)"
    final_answer_match = re.search(final_answer_pattern, data)
    final_answer = final_answer_match.group(1) if final_answer_match else None
    final_confidence = int(final_answer_match.group(2)) if final_answer_match else None

    # Create the result dictionary
    result = {}
    for i, step_match in enumerate(step_matches):
        step_key = f"step{i+1}"
        step_analysis = step_match[1].strip()
        step_confidence = int(confidence_matches[i]) if i < len(confidence_matches) else None
        result[step_key] = {
            "analysis": step_analysis,
            "confidence": step_confidence
        }
    result["final_answer"] = final_answer
    result["final_confidence"] = final_confidence
    return result

