""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import _silences, symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")
"""这段代码中包含了两个主要部分：

1. `_symbol_to_id` 和 `_id_to_symbol`：这两个字典分别用于将符号映射到数字 ID 和将数字 ID 映射回符号。它们是通过将符号列表 `symbols` 中的每个符号与其相应的索引值关联而创建的。例如，
如果 `symbols` 中包含了 `{'a', 'b', 'c'}`，那么`_symbol_to_id` 可能是 `{'a': 0, 'b': 1, 'c': 2}`，而 `_id_to_symbol` 则可能是 `{0: 'a', 1: 'b', 2: 'c'}`。

2. `_curly_re`：这是一个正则表达式对象，用于匹配被花括号 `{}` 包围的文本。正则表达式 `r"(.*?)\{(.+?)\}(.*)"` 中的各个部分表示：
   - `(.*?)`：匹配零个或多个任意字符（非贪婪），直到下一个部分（花括号 `{`）出现。
   - `\{(.+?)\}`：匹配被花括号包围的任意字符（非贪婪）。
   - `(.*)`：匹配零个或多个任意字符（贪婪）。
这个正则表达式可以用于从文本中提取出被花括号包围的内容，例如，将字符串 `"This is {a} test"` 与这个正则表达式匹配，将得到三个匹配组 `("This is ", "a", " test")`。
"""


def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)

        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence
"""这个函数将文本字符串转换为对应的符号序列（列表形式），符号序列中的每个符号都对应于文本中的一个字符或者是 ARPAbet 序列。

函数的主要步骤如下：

1. 首先，通过循环遍历文本字符串中的每个字符，直到整个文本被处理完毕。
2. 在循环中，使用正则表达式 `_curly_re` 来匹配文本中的花括号 `{}`，如果找到了匹配项，则意味着文本中包含 ARPAbet 序列。如果未找到匹配项，则说明文本中不包含 ARPAbet 序列，直接将文本进行清洗并转换
为符号序列。
3. 如果找到了匹配项，将花括号之前的文本部分进行清洗，并转换为符号序列，然后将清洗后的 ARPAbet 序列转换为相应的符号序列，并将其连接到之前的符号序列中。
4. 继续处理剩余的文本，直到整个文本字符串被处理完毕。
5. 返回最终的符号序列。

在这个过程中，使用了其他函数 `_clean_text` 和 `_arpabet_to_sequence` 来对文本进行清洗和 ARPAbet 序列的转换，但这两个函数的具体实现在提供的代码中没有给出。
"""


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def sil_phonemes_ids():
    return [_symbol_to_id[sil] for sil in _silences]


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"
