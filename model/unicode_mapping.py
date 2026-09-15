"""
unicode_mapping.py - Authoritative Class-to-Unicode mapping for uTHCD dataset.

Derived strictly from the IEEE Access research paper:
"uTHCD: A New Benchmarking for Tamil Handwritten OCR"
Authors: Noushath Shaffi and Faizal Hajamohideen (IEEE Access, Vol. 9, 2021)
Figures 11, 28, and 29.

156 classes total, indexed from 0 to 155.
UTF-8 encoding throughout.
"""

from typing import Dict, Any, List, Optional

# Raw mapping table: class_id -> (Tamil Character, Hex codepoints space-separated)
_RAW_MAPPINGS: Dict[int, tuple] = {
    0: ('ா', '0BBE'),
    1: ('அ', '0B85'),
    2: ('ஆ', '0B86'),
    3: ('இ', '0B87'),
    4: ('ஈ', '0B88'),
    5: ('உ', '0B89'),
    6: ('ஊ', '0B8A'),
    7: ('எ', '0B8E'),
    8: ('ஏ', '0B8F'),
    9: ('ஐ', '0B90'),
    10: ('ஒ', '0B92'),
    11: ('ஓ', '0B93'),
    12: ('ஔ', '0B94'),
    13: ('ஃ', '0B83'),
    14: ('க்', '0B95 0BCD'),
    15: ('க', '0B95'),
    16: ('கி', '0B95 0BBF'),
    17: ('கீ', '0B95 0BC0'),
    18: ('கு', '0B95 0BC1'),
    19: ('கூ', '0B95 0BC2'),
    20: ('ச்', '0B9A 0BCD'),
    21: ('ச', '0B9A'),
    22: ('சி', '0B9A 0BBF'),
    23: ('சீ', '0B9A 0BC0'),
    24: ('சு', '0B9A 0BC1'),
    25: ('சூ', '0B9A 0BC2'),
    26: ('ங்', '0B99 0BCD'),
    27: ('ங', '0B99'),
    28: ('ஙி', '0B99 0BBF'),
    29: ('ஙீ', '0B99 0BC0'),
    30: ('ஙு', '0B99 0BC1'),
    31: ('ஙூ', '0B99 0BC2'),
    32: ('ஞ்', '0B9E 0BCD'),
    33: ('ஞ', '0B9E'),
    34: ('ஞி', '0B9E 0BBF'),
    35: ('ஞீ', '0B9E 0BC0'),
    36: ('ஞு', '0B9E 0BC1'),
    37: ('ஞூ', '0B9E 0BC2'),
    38: ('ட்', '0B9F 0BCD'),
    39: ('ட', '0B9F'),
    40: ('டி', '0B9F 0BBF'),
    41: ('டீ', '0B9F 0BC0'),
    42: ('டு', '0B9F 0BC1'),
    43: ('டூ', '0B9F 0BC2'),
    44: ('ண்', '0BA3 0BCD'),
    45: ('ண', '0BA3'),
    46: ('ணி', '0BA3 0BBF'),
    47: ('ணீ', '0BA3 0BC0'),
    48: ('ணு', '0BA3 0BC1'),
    49: ('ணூ', '0BA3 0BC2'),
    50: ('த்', '0BA4 0BCD'),
    51: ('த', '0BA4'),
    52: ('தி', '0BA4 0BBF'),
    53: ('தீ', '0BA4 0BC0'),
    54: ('து', '0BA4 0BC1'),
    55: ('தூ', '0BA4 0BC2'),
    56: ('ந்', '0BA8 0BCD'),
    57: ('ந', '0BA8'),
    58: ('நி', '0BA8 0BBF'),
    59: ('நீ', '0BA8 0BC0'),
    60: ('நு', '0BA8 0BC1'),
    61: ('நூ', '0BA8 0BC2'),
    62: ('ப்', '0BAA 0BCD'),
    63: ('ப', '0BAA'),
    64: ('பி', '0BAA 0BBF'),
    65: ('பீ', '0BAA 0BC0'),
    66: ('பு', '0BAA 0BC1'),
    67: ('பூ', '0BAA 0BC2'),
    68: ('ம்', '0BAE 0BCD'),
    69: ('ம', '0BAE'),
    70: ('மி', '0BAE 0BBF'),
    71: ('மீ', '0BAE 0BC0'),
    72: ('மு', '0BAE 0BC1'),
    73: ('மூ', '0BAE 0BC2'),
    74: ('ய்', '0BAF 0BCD'),
    75: ('ய', '0BAF'),
    76: ('யி', '0BAF 0BBF'),
    77: ('யீ', '0BAF 0BC0'),
    78: ('யு', '0BAF 0BC1'),
    79: ('யூ', '0BAF 0BC2'),
    80: ('ர்', '0BB0 0BCD'),
    81: ('ர', '0BB0'),
    82: ('ரி', '0BB0 0BBF'),
    83: ('ரீ', '0BB0 0BC0'),
    84: ('ரு', '0BB0 0BC1'),
    85: ('ரூ', '0BB0 0BC2'),
    86: ('ல்', '0BB2 0BCD'),
    87: ('ல', '0BB2'),
    88: ('லி', '0BB2 0BBF'),
    89: ('லீ', '0BB2 0BC0'),
    90: ('லு', '0BB2 0BC1'),
    91: ('லூ', '0BB2 0BC2'),
    92: ('ள்', '0BB3 0BCD'),
    93: ('ள', '0BB3'),
    94: ('ளி', '0BB3 0BBF'),
    95: ('ளீ', '0BB3 0BC0'),
    96: ('ளு', '0BB3 0BC1'),
    97: ('ளூ', '0BB3 0BC2'),
    98: ('ற்', '0BB1 0BCD'),
    99: ('ற', '0BB1'),
    100: ('றி', '0BB1 0BBF'),
    101: ('றீ', '0BB1 0BC0'),
    102: ('று', '0BB1 0BC1'),
    103: ('றூ', '0BB1 0BC2'),
    104: ('வ்', '0BB5 0BCD'),
    105: ('வ', '0BB5'),
    106: ('வி', '0BB5 0BBF'),
    107: ('வீ', '0BB5 0BC0'),
    108: ('வு', '0BB5 0BC1'),
    109: ('வூ', '0BB5 0BC2'),
    110: ('ழ்', '0BB4 0BCD'),
    111: ('ழ', '0BB4'),
    112: ('ழி', '0BB4 0BBF'),
    113: ('ழீ', '0BB4 0BC0'),
    114: ('ழு', '0BB4 0BC1'),
    115: ('ழூ', '0BB4 0BC2'),
    116: ('ன்', '0BA9 0BCD'),
    117: ('ன', '0BA9'),
    118: ('னி', '0BA9 0BBF'),
    119: ('னீ', '0BA9 0BC0'),
    120: ('னு', '0BA9 0BC1'),
    121: ('ஷி', '0BB7 0BBF'),
    122: ('ஷீ', '0BB7 0BC0'),
    123: ('ஷு', '0BB7 0BC1'),
    124: ('ஷூ', '0BB7 0BC2'),
    125: ('க்ஷ', '0B95 0BCD 0BB7'),
    126: ('க்ஷ்', '0B95 0BCD 0BB7 0BCD'),
    127: ('க்ஷி', '0B95 0BCD 0BB7 0BBF'),
    128: ('க்ஷீ', '0B95 0BCD 0BB7 0BC0'),
    129: ('ஜு', '0B9C 0BC1'),
    130: ('ஜூ', '0B9C 0BC2'),
    131: ('ஹ', '0BB9'),
    132: ('ஹ்', '0BB9 0BCD'),
    133: ('ஹி', '0BB9 0BBF'),
    134: ('ஹீ', '0BB9 0BC0'),
    135: ('ஹு', '0BB9 0BC1'),
    136: ('ஹூ', '0BB9 0BC2'),
    137: ('ஸ', '0BB8'),
    138: ('ஸ்', '0BB8 0BCD'),
    139: ('ஸி', '0BB8 0BBF'),
    140: ('ஸீ', '0BB8 0BC0'),
    141: ('ஸு', '0BB8 0BC1'),
    142: ('ஸூ', '0BB8 0BC2'),
    143: ('ஷ', '0BB7'),
    144: ('ஷ்', '0BB7 0BCD'),
    145: ('னூ', '0BA9 0BC2'),
    146: ('ஸ்ரீ', '0BB8 0BCD 0BB0 0BC0'),
    147: ('க்ஷூ', '0B95 0BCD 0BB7 0BC2'),
    148: ('ஜ', '0B9C'),
    149: ('ஜ்', '0B9C 0BCD'),
    150: ('ஜி', '0B9C 0BBF'),
    151: ('ஜீ', '0B9C 0BC0'),
    152: ('க்ஷு', '0B95 0BCD 0BB7 0BC1'),
    153: ('ெ', '0BC6'),
    154: ('ே', '0BC7'),
    155: ('ை', '0BC8'),
}

# Precompute structured dictionaries
CLASS_TO_UNICODE: Dict[int, Dict[str, Any]] = {}
UNICODE_TO_CLASS: Dict[str, int] = {}

for class_id, (char, hex_str) in _RAW_MAPPINGS.items():
    codepoint_list = ['U+' + h for h in hex_str.split()]
    formatted_codepoints = ' '.join(codepoint_list)
    CLASS_TO_UNICODE[class_id] = {
        'class_id': class_id,
        'char': char,
        'hex': hex_str,
        'codepoints': formatted_codepoints,
    }
    UNICODE_TO_CLASS[char] = class_id

NUM_CLASSES = len(CLASS_TO_UNICODE)

def get_unicode_char(class_id: int) -> str:
    """Return the Tamil Unicode character string for a given class ID."""
    if class_id not in CLASS_TO_UNICODE:
        raise KeyError(f"Invalid class_id {class_id}. Must be between 0 and {NUM_CLASSES - 1}.")
    return CLASS_TO_UNICODE[class_id]['char']

def get_unicode_codepoints(class_id: int) -> str:
    """Return the formatted Unicode code point string (e.g., 'U+0B95') for a class ID."""
    if class_id not in CLASS_TO_UNICODE:
        raise KeyError(f"Invalid class_id {class_id}. Must be between 0 and {NUM_CLASSES - 1}.")
    return CLASS_TO_UNICODE[class_id]['codepoints']

def get_class_id(unicode_char: str) -> Optional[int]:
    """Return the class ID corresponding to a given Tamil character, or None if not found."""
    return UNICODE_TO_CLASS.get(unicode_char)

def get_mapping_info(class_id: int) -> Dict[str, Any]:
    """Return the complete metadata dict for a class ID."""
    if class_id not in CLASS_TO_UNICODE:
        raise KeyError(f"Invalid class_id {class_id}. Must be between 0 and {NUM_CLASSES - 1}.")
    return CLASS_TO_UNICODE[class_id]

def get_all_mappings() -> List[Dict[str, Any]]:
    """Return a list of all 156 class mappings sorted by class_id."""
    return [CLASS_TO_UNICODE[i] for i in range(NUM_CLASSES)]
