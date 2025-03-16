#!/usr/bin/env python3

from marginaleffects.docs import DocsModels
from marginaleffects.model_sklearn import docs_sklearn

def evaluate_sklearn_docs():
    """
    Evaluates the docs_sklearn string from model_sklearn.py by using
    the same components and string concatenation.
    """ 
    
    print("Evaluated docs_sklearn:")
    print("=" * 80)
    print(docs_sklearn)
    print("=" * 80)
    
    return docs_sklearn

def inject_docstring_to_sklearn(docs):
    """
    Injects the evaluated docstring into model_sklearn.py right after
    the fit_sklearn function signature.
    """
    # Read the current content of model_sklearn.py
    with open('marginaleffects/model_sklearn.py', 'r') as f:
        lines = f.readlines()
    
    # Find the line with the fit_sklearn function definition
    for i, line in enumerate(lines):
        if line.strip().startswith('def fit_sklearn('):
            # Find the line with the closing parenthesis and return type
            # works when column is not on same line as function definition
            # works when column is on same line as function definition
            while not lines[i].strip().endswith(':'): 
                i += 1
            inject_position = i + 1
            break
    else:
        raise ValueError("Could not find fit_sklearn function definition")
    
    # Get the docstring with proper indentation
    docstring = f'    """{docs}"""'
    docstring_lines = docstring.split('\n')
    
    # Insert the docstring lines after the function definition
    lines[inject_position:inject_position] = [line + '\n' for line in docstring_lines]
    
    # Write the modified content back to the file
    with open('marginaleffects/model_sklearn.py', 'w') as f:
        f.writelines(lines)
    
    print("Successfully injected docstring into model_sklearn.py")

def clean_sklearn_docstring():
    # Read the current content of model_sklearn.py
    with open('marginaleffects/model_sklearn.py', 'r') as f:
        lines = f.readlines()
    
    # Find the line with the fit_sklearn function definition
    for i, line in enumerate(lines):
        if line.strip().startswith('def fit_sklearn('):
            # Find the line with the closing parenthesis and return type
            # works when column is not on same line as function definition
            # works when column is on same line as function definition
            while not lines[i].strip().endswith(':'): 
                i += 1
            inject_position = i + 1
            break
    else:
        raise ValueError("Could not find fit_sklearn function definition")
    # Does this line contain a docstring?
    if '"""' in lines[inject_position]:
        # Find the end of the docstring
        for j in range(inject_position + 1, len(lines)):
            if '"""' in lines[j]:
                docstring_end = j
                break
    else:
        raise ValueError("Could not find docstring")

    # Remove the existing docstring
    lines[inject_position:docstring_end + 1] = []

    # Write the modified content back to the file
    with open('marginaleffects/model_sklearn.py', 'w') as f:
        f.writelines(lines)

    print("Successfully cleaned docstring from model_sklearn.py")

if __name__ == "__main__":
    # First evaluate the docstring
    docs = evaluate_sklearn_docs()

    # Then clean the docstring
    clean_sklearn_docstring()
    
    # Then inject it into the source code
    inject_docstring_to_sklearn(docs) 