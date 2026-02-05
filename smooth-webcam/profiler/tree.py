"""
Tree Printing Utility
Uses: nothing
Used by: profiler

ASCII tree visualization for nested data structures.
"""


def tree_dict(data):
    """Normalizes data into a nested dict schema for tree display."""
    if isinstance(data, (list, tuple)) and data:
        label = f"[0...{len(data) - 1}]"
        merged = {}
        for item in data:
            res = tree_dict(item)
            if isinstance(res, dict):
                merged.update(res)
        return {label: merged} if merged else str(type(data).__name__)

    if isinstance(data, dict):
        if not data:
            return "(empty)"
        return {k: tree_dict(v) for k, v in data.items()}

    return str(data)


def tree_string(
    schema,
    prefix="",
):
    """Recursively builds the ASCII tree from the schema dict."""
    if not isinstance(schema, dict):
        return str(schema)

    lines = []
    items = list(schema.items())
    for i, (key, value) in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        if isinstance(value, dict):
            lines.append(f"{prefix}{connector}{key}")
            lines.append(tree_string(value, child_prefix))
        else:
            lines.append(f"{prefix}{connector}{key}: {value}")

    return "\n".join(lines)


def tree(data):
    """Generates a tree visualization of any object/dictionary."""
    schema = tree_dict(data)
    return tree_string(schema)
