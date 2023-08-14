def read_number_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            number = int(file.read())
            return number
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except ValueError:
        print(f"File '{file_path}' does not contain a valid number.")
        return None

def write_number_to_file(file_path, number):
    with open(file_path, 'w') as file:
        file.write(str(number))

def update_version(file_path):
    last_version = read_number_from_file(file_path)
    write_number_to_file(file_path, last_version + 1)
    return last_version + 1

f = open("test.csv", "a")
s = ""
for i in range(400):
    s = s + f"{i}, {i}, {i}, {i}\n"

f.write(s)
f.close()