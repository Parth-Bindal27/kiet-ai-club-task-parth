"""Validation filters."""

def validate_example(example: dict) -> tuple[bool, str]:
    if "instruction" not in example or "response" not in example:
        return False, "missing_fields"
        
    response = example["response"].strip()
    if not response:
        return False, "empty_response"
        
    if response == example["instruction"].strip():
        return False, "repeated_instruction"
        
    return True, "valid"
