import re


def read_file():
    with open("qlib/__init__.py", "r", encoding="utf-8") as f:
        content = f.read()
    return content


def write_file(content: str):
    with open("qlib/__init__.py", "w", encoding="utf-8") as f:
        f.write(content)


def update_version(version_num: list):
    if len(version_num) == 3:
        new_version = f"{version_num[0]}.{version_num[1]}.{version_num[2]}.{1}"
    if len(version_num) == 4:
        new_version = f"{version_num[0]}.{version_num[1]}.{version_num[2]}.{int(version_num[3]) + 1}"
    return new_version


def main():
    content = read_file()
    pattern = r"__version__ = \"(\d+(\.\d+)*(\.\d+)*)\""
    match = re.search(pattern, content)
    old_version = match.group(1)
    old_version_num = old_version.split(".")
    new_version = update_version(old_version_num)
    new_content = content.replace(old_version, new_version)
    write_file(new_content)


if __name__ == "__main__":
    main()
