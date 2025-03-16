import marginaleffects


def inject_docstring_to_func(func_dict: dict):
    """
    Injects the evaluated docstring into the function right after
    the function signature.
    """
    # Read the current content of model_sklearn.py
    with open(f"marginaleffects/{func_dict['file_name']}.py", "r") as f:
        lines = f.readlines()

    # Find the line with the fit_sklearn function definition
    for i, line in enumerate(lines):
        if line.strip().startswith(f"def {func_dict['func_name']}("):
            # Find the line with the closing parenthesis and return type
            # works when column is not on same line as function definition
            # works when column is on same line as function definition
            while not lines[i].strip().endswith(":"):
                i += 1
            inject_position = i + 1
            break
    else:
        raise ValueError(f"Could not find {func_dict['func_name']} function definition")

    # Get the docstring with proper indentation
    docstring = f'    """{docs}"""'
    docstring_lines = docstring.split("\n")

    # Insert the docstring lines after the function definition
    lines[inject_position:inject_position] = [line + "\n" for line in docstring_lines]

    # Write the modified content back to the file
    with open(f"marginaleffects/{func_dict['file_name']}.py", "w") as f:
        f.writelines(lines)

    print(f"Successfully injected docstring into {func_dict['file_name']}.py")


def clean_func_docstring(func_dict: dict):
    # Read the current content of model_sklearn.py
    with open(f"marginaleffects/{func_dict['file_name']}.py", "r") as f:
        lines = f.readlines()

    # Find the line with the fit_sklearn function definition
    for i, line in enumerate(lines):
        if line.strip().startswith(f"def {func_dict['func_name']}("):
            # Find the line with the closing parenthesis and return type
            # works when column is not on same line as function definition
            # works when column is on same line as function definition
            while not lines[i].strip().endswith(":"):
                i += 1
            inject_position = i + 1
            break
    else:
        raise ValueError(f"Could not find {func_dict['func_name']} function definition")
    # Does this line contain a docstring?
    if '"""' in lines[inject_position]:
        # Find the end of the docstring
        for j in range(inject_position + 1, len(lines)):
            if '"""' in lines[j]:
                docstring_end = j
                break
    else:
        print(f"Could not find docstring for {func_dict['func_name']}")
        return

    # Remove the existing docstring
    lines[inject_position : docstring_end + 1] = []

    # Write the modified content back to the file
    with open(f"marginaleffects/{func_dict['file_name']}.py", "w") as f:
        f.writelines(lines)

    print(f"Successfully cleaned docstring from {func_dict['file_name']}.py")


if __name__ == "__main__":
    # create a dictionary mapping function names to file names
    func_to_file = {}
    for func in marginaleffects.__all__:
        # get function from marginaleffects package
        func = getattr(marginaleffects, func)
        func_to_file[func] = {"module_name": func.__module__ + "." + func.__name__}
        # the value of each item looks like: marginaleffects.comparisons.avg_comparisons
        # we want to extract the file and the function name from the value
        file_name = func_to_file[func]["module_name"].split(".")[-2]
        func_name = func_to_file[func]["module_name"].split(".")[-1]
        func_to_file[func]["file_name"] = file_name
        func_to_file[func]["func_name"] = func_name

    for func in func_to_file:
        print(func_to_file[func])
        docs = func.__doc__
        clean_func_docstring(func_to_file[func])
        inject_docstring_to_func(func_to_file[func])
