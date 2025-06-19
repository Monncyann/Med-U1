import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from collections import defaultdict
nltk.data.path.append("/rsch/xiaotian/l1/verl/verl/utils/punkt")

contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]

def split_sentence_list(sentence, n): # for Rough-L
    words = []
    tmp_sentence = sentence.lower().strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words.append(tmp_words)
    return words

def longest_common_subsequence(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def compute_rouge_l(candidate, reference):
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence_list(candidate, 1)
    reference_words = split_sentence_list(reference, 1)

    lcs = longest_common_subsequence(candidate_words, reference_words)

    m = len(candidate_words)
    n = len(reference_words)

    if m == 0 or n == 0:
        return 0

    precision = lcs / m
    recall = lcs / n

    if precision + recall == 0:
        return 0

    beta = 1.0
    f1 = (1 + beta ** 2) * precision * recall / (recall + beta ** 2 * precision)

    return 100 * f1

def normalize_word(token):
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (
            re.search(comma_strip, token) != None
        ):
            _token = _token.replace(p, "")
        else:
            _token = _token.replace(p, " ")
    token = period_strip.sub("", _token, re.UNICODE)

    _token = []
    temp = token.lower().split()
    for word in temp:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            _token.append(word)
    for i, word in enumerate(_token):
        if word in contractions:
            _token[i] = contractions[word]
    token = " ".join(_token)
    token = token.replace(",", "")
    return token

def extract_solution(solution_str: str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str: # base
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str: # qwen and tower
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str: # llama3
        processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def extract_answer(solution_str: str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, solution_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, solution_str

def split_sentence(sentence, n):
    words = defaultdict(int)
    # tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = sentence
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words

def calculate_exactmatch(candidate, reference):

    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]
        
    if total == 0:
        return 0 # "0 (warning: length of candidate's words is 0)"
    else:
        return 100*(count / total)
    
def compute_bleu(ref, pred):
    pred = pred if isinstance(pred, str) else ""

    ref_tokens = ref.split()
    pred_tokens = pred.split()

    weights = [(1, 0, 0, 0),  # 1-gram
               (0, 1, 0, 0),  # 2-gram
               (0, 0, 1, 0)]  # 3-gram

    bleu_scores = []
    smoothing = SmoothingFunction().method1

    for w in weights:
        score = sentence_bleu( [ref_tokens], pred_tokens, weights=w, smoothing_function=smoothing)
        bleu_scores.append(round(score, 4))
    
    print(f"[BLEU Scores] 1-gram: {bleu_scores[0]:.4f}, "
          f"2-gram: {bleu_scores[1]:.4f}, "
          f"3-gram: {bleu_scores[2]:.4f}")
    final_score = (bleu_scores[0] + bleu_scores[1] + bleu_scores[2]) / 3
    return float(final_score)