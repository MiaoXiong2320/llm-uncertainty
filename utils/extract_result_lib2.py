import re

def extract1(data):
    step_info = re.findall(r"Step (\d+): (.+?)\. Confidence: (\d+)", data)
    steps = {}
    for step in step_info:
        step_num = step[0]
        step_analysis = step[1]
        step_confidence = int(step[2])
        steps[f"step{step_num}"] = {
            "analysis": step_analysis,
            "confidence": step_confidence
        }

    # Extracting Final Answer and Confidence information
    final_info = re.search(r"Final Answer and Confidence\(0-100\): \$([\d,]+), (\d+)", data)
    final_answer = final_info.group(1).replace(",", "") if final_info.group(1)[0] == "$" else final_info.group(1)
    final_confidence = int(final_info.group(2))

    # Creating the result dictionary
    result = {
        **steps,
        "final_answer": final_answer,
        "final_confidence": final_confidence
    }
    return result

def extract2(data):
    step_info = re.findall(r"Step (\d+): (.+?)\n(.*?)(?=Step \d+|$)", data, flags=re.DOTALL)
    steps = {}
    for step in step_info:
        step_num = step[0]
        step_analysis = step[1]
        step_confidence = re.search(r"Confidence: (\d+)", step[2]).group(1)
        steps[f"step{step_num}"] = {
            "analysis": step_analysis.strip(),
            "confidence": int(step_confidence)
        }

    # Extracting Final Answer and Confidence information
    final_info = re.search(r"Final Answer and Confidence\(0-100\): (\d+(?:\.\d+)?), (\d+)", data)
    final_answer = final_info.group(1)
    final_confidence = int(final_info.group(2))

    # Creating the result dictionary
    result = {
        **steps,
        "final_answer": final_answer,
        "final_confidence": final_confidence
    }
    return result

def extract3(data):
    step_info = re.findall(r"Step (\d+): (.+?)\nConfidence: (\d+)", data)
    print(step_info)
    steps = {}
    #print(step_info)
    for step in step_info:
        step_num = step[0]
        step_analysis = step[1]
        step_confidence = int(step[2])
        steps[f"step{step_num}"] = {
            "analysis": step_analysis.strip(),
            "confidence": step_confidence
        }

    # Extracting Final Answer and Confidence information
    final_info = re.search(r"Final Answer and Confidence\(0-100\): \$([\d,]+), (\d+)", data)
    #print(steps)
    #print(final_info)
    final_answer = final_info.group(1)
    final_confidence = int(final_info.group(2))
    #print(final_answer)

    # Creating the result dictionary
    result = {
        **steps,
        "final_answer": final_answer,
        "final_confidence": final_confidence
    }
    return result

def extract4(data):
    step_pattern = r'Step (\d+): (.+?)\nConfidence: (\d+)'
    final_answer_pattern = r'Final Answer and Confidence\(0-100\): \$([\d.]+), (\d+)'

    # Extract the data using regex
    step_matches = re.findall(step_pattern, data)
    final_answer_match = re.search(final_answer_pattern, data)

    # Create the result dictionary
    result = {}
    for match in step_matches:
        step_number = match[0]
        step_content = match[1]
        step_confidence = match[2]
        result[f'step{step_number}'] = {'analysis': step_content, 'confidence': step_confidence}

    if final_answer_match:
        final_answer = final_answer_match.group(1)
        if '.' in final_answer:
            final_answer = float(final_answer)
        else:
            final_answer = int(final_answer)
        final_confidence = int(final_answer_match.group(2))
        result['final_answer'] = final_answer
        result['final_confidence'] = final_confidence
    return result


def extract5(data):
    step_info = re.findall(r"Step (\d+): (.+?)\n(.*?)(?=Step \d+|$)", data, flags=re.DOTALL)
    steps = {}
    for step in step_info:
        step_num = step[0]
        step_analysis = step[1]
        step_confidence = re.search(r"Confidence: (\d+)", step[2]).group(1)
        steps[f"step{step_num}"] = {
            "analysis": step_analysis.strip(),
            "confidence": int(step_confidence)
        }

    # Extracting Final Answer and Confidence information
    final_info = re.search(r"Final Answer and Confidence\(0-100\): ([\d,]+), (\d+)", data)
    if final_info:
        final_answer = final_info.group(1).replace(",", "")  # 提取数字并移除逗号
    #final_answer = final_info.group(1)
    final_confidence = int(final_info.group(2))

    # Creating the result dictionary
    result = {
        **steps,
        "final_answer": final_answer,
        "final_confidence": final_confidence
    }
    return result

def extract6(data):
    step_pattern = r'Step(\d+): (.+?), confidence: (\d+);'
    final_answer_pattern = r'Final Answer and Confidence\(0-100\): (\d+(?:\.\d+)?), (\d+)'

    # Extract the data using regex
    step_matches = re.findall(step_pattern, data)
    final_answer_match = re.search(final_answer_pattern, data)

    # Create the result dictionary
    result = {}
    for match in step_matches:
        step_number = match[0]
        step_content = match[1]
        step_confidence = match[2]
        result[f'step{step_number}'] = {'analysis': step_content, 'confidence': step_confidence}

    if final_answer_match:
        final_answer = final_answer_match.group(1)
        if '.' in final_answer:
            final_answer = float(final_answer)
        else:
            final_answer = int(final_answer)
        final_confidence = int(final_answer_match.group(2))
        result['final_answer'] = final_answer
        result['final_confidence'] = final_confidence
    return result

def extract7(data):
    step_pattern = r'Step(\d+): (.+?), confidence: (\d+);'
    final_answer_pattern = r'Final Answer and Confidence: (\d+(?:\.\d+)?), (\d+)'

    # Extract the data using regex
    step_matches = re.findall(step_pattern, data)
    final_answer_match = re.search(final_answer_pattern, data)

    # Create the result dictionary
    result = {}
    for match in step_matches:
        step_number = match[0]
        step_content = match[1]
        step_confidence = match[2]
        result[f'step{step_number}'] = {'analysis': step_content, 'confidence': step_confidence}

    if final_answer_match:
        final_answer = final_answer_match.group(1)
        if '.' in final_answer:
            final_answer = float(final_answer)
        else:
            final_answer = int(final_answer)
        final_confidence = int(final_answer_match.group(2))
        result['final_answer'] = final_answer
        result['final_confidence'] = final_confidence
    return result

def extract_v2(data):
    functions_to_try = [extract1, extract2, extract3, extract4, extract5,extract6,extract7]
    for function in functions_to_try:
        try:
            result = function(data)
            assert result['step1'] is not None,'step1 can not be none'
            assert result['final_answer'] is not None, "final_answer 不能为 None"
            assert result['final_confidence'] is not None, "final_confidence 不能为 None"
            assert result['final_answer'] is not None, "final_answer 不能为 None"
            assert result['step1']['confidence'] is not None, "confidence 不能为 None"
            assert result['step2']['confidence'] is not None, "confidence 不能为 None"
            #assert result['step2']['confidence'] is not None, "confidence 不能为 None"            

            if result is not None:
                print(function)
                return result
        except Exception as e:
            print(function)
            pass
    return None